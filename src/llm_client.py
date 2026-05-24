"""Client for the vLLM server.

vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint, so we use the
official openai SDK and don't have to maintain any custom HTTP code.

Key vLLM features we are relying on:
  - PagedAttention: KV cache is paged like OS virtual memory → near-zero fragmentation.
  - Continuous batching: new requests slot into free positions without waiting.
  - Prefix caching: identical system prompts are KV-cached across requests.
"""
from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from .config import settings


def _encode_image_to_data_url(image_path: str | Path) -> str:
    """Convert a local image into a base64 data URL the VLM can consume."""
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    # JPEG can be referenced as either "jpg" or "jpeg" — normalize.
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


class VLMClient:
    """Wrapper around the vLLM OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )
        self.model = settings.vlm_model_id

    def chat(
        self,
        images: list[str | Path],
        text: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a multimodal message and return the assistant's reply text.

        Args:
            images: list of local image paths. Order matters — they appear before
                    the text in the message content.
            text: the textual portion of the user turn.
            max_tokens / temperature: overrides for the defaults in settings.

        Returns:
            The assistant's response string.
        """
        content: list[dict] = []
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image_to_data_url(img_path)},
            })
        content.append({"type": "text", "text": text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens or settings.generation_max_tokens,
            temperature=temperature if temperature is not None
                        else settings.generation_temperature,
        )
        return response.choices[0].message.content or ""
