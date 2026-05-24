"""End-to-end Multimodal RAG pipeline for defect diagnosis.

Flow:
  1. Embed the query image with SigLIP.
  2. Retrieve top-K similar historical cases from Qdrant.
  3. Build a multimodal prompt: [query_img, case_1_img, case_1_meta, ..., user_question].
  4. Send to vLLM-served VLM and return the structured answer.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings
from .embedding import embed_image
from .llm_client import VLMClient
from .vector_store import RetrievedCase, VectorStore

logger = logging.getLogger(__name__)

# vLLM is launched with --limit-mm-per-prompt image=3 (1 query + 2 retrieved).
# Clamping here keeps the RAG layer in sync without requiring callers to know
# the vLLM flag value.
_VLLM_MAX_IMAGES = 3


@dataclass
class DiagnosisResult:
    """Structured result of a single diagnosis call — easier to log / serialize."""
    answer: str
    retrieved_cases: list[dict]  # payload + score for each retrieved case

    def to_dict(self) -> dict:
        return asdict(self)


class DefectDiagnosisRAG:
    """High-level orchestrator. Inject dependencies in tests if needed."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_client: VLMClient | None = None,
    ) -> None:
        self.vs = vector_store or VectorStore()
        self.llm = llm_client or VLMClient()

    def diagnose(
        self,
        query_image: str | Path,
        question: str,
        top_k: int | None = None,
    ) -> DiagnosisResult:
        top_k = top_k or settings.retrieval_top_k
        max_retrieved = _VLLM_MAX_IMAGES - 1  # reserve slot 0 for the query image
        if top_k > max_retrieved:
            logger.warning(
                "top_k=%d exceeds vLLM image limit (%d images total); "
                "clamping to %d retrieved cases.",
                top_k, _VLLM_MAX_IMAGES, max_retrieved,
            )
            top_k = max_retrieved

        # 1. Retrieve
        query_vec = embed_image(query_image)
        hits = self.vs.search(query_vector=query_vec, top_k=top_k)

        # 2. Build prompt
        images, prompt = self._build_prompt(query_image, question, hits)

        # 3. Generate
        answer = self.llm.chat(images=images, text=prompt)

        return DiagnosisResult(
            answer=answer,
            retrieved_cases=[
                {"score": h.score, **h.payload} for h in hits
            ],
        )

    @staticmethod
    def _build_prompt(
        query_image: str | Path,
        question: str,
        hits: list[RetrievedCase],
    ) -> tuple[list[str | Path], str]:
        """Assemble the image list and text portion for the VLM.

        The image list MUST stay in the same order as the textual references
        ([Case 1], [Case 2], ...). The model relies on positional grounding.
        """
        images: list[str | Path] = [query_image]
        prompt_parts = [
            "You are an industrial defect-diagnosis assistant.",
            "Image 1 above is the PART UNDER INSPECTION.",
            "",
            f"Below are {len(hits)} similar historical cases retrieved from the case base:",
        ]

        for i, hit in enumerate(hits, start=1):
            img_path = hit.payload.get("image_path")
            if img_path and Path(img_path).is_file():
                images.append(img_path)
            else:
                logger.warning("Case %d has no valid image_path; skipping image.", i)
            prompt_parts.append("")
            prompt_parts.append(
                f"[Case {i}] cosine similarity = {hit.score:.3f}\n"
                f"  Defect type: {hit.payload.get('defect_type', 'unknown')}\n"
                f"  Description: {hit.payload.get('description', 'n/a')}\n"
                f"  Recommended SOP: {hit.payload.get('sop', 'n/a')}"
            )

        prompt_parts += [
            "",
            f"User question: {question}",
            "",
            "Provide a concise diagnosis with:",
            "  1) Most likely defect type",
            "  2) Reasoning grounded in the retrieved cases",
            "  3) Recommended next action",
        ]
        return images, "\n".join(prompt_parts)
