"""FastAPI service exposing the RAG pipeline.

Endpoints:
  GET  /health     — liveness probe (used by K8s)
  POST /diagnose   — multipart upload (image file + question form field)

Production patterns shown:
  - Dependency-injected RAG instance (testable, easy to mock).
  - Pydantic response model (auto OpenAPI docs at /docs).
  - Health endpoint with downstream checks.
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .rag import DefectDiagnosisRAG, DiagnosisResult

_rag_instance: DefectDiagnosisRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_instance
    _rag_instance = DefectDiagnosisRAG()
    yield
    _rag_instance = None


app = FastAPI(title="Defect-MMRAG", version="0.1.0", lifespan=lifespan)


# ---- Dependency injection -----------------------------------------------------

def get_rag() -> DefectDiagnosisRAG:
    assert _rag_instance is not None, "RAG not initialized"
    return _rag_instance


# ---- Schemas ------------------------------------------------------------------

class DiagnoseResponse(BaseModel):
    answer: str
    retrieved_cases: list[dict]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    vllm: bool


# ---- Endpoints ----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health(rag: DefectDiagnosisRAG = Depends(get_rag)) -> HealthResponse:
    """Basic liveness + downstream checks."""
    qdrant_ok = True
    vllm_ok = True
    try:
        rag.vs.client.get_collections()
    except Exception:
        qdrant_ok = False
    try:
        rag.llm.client.models.list()
    except Exception:
        vllm_ok = False
    return HealthResponse(
        status="ok" if qdrant_ok and vllm_ok else "degraded",
        qdrant=qdrant_ok,
        vllm=vllm_ok,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    image: UploadFile = File(...),
    question: str = Form("What defect is shown and what should we do?"),
    rag: DefectDiagnosisRAG = Depends(get_rag),
) -> DiagnoseResponse:
    """Upload an image and get a multimodal RAG-grounded diagnosis."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Persist the upload to a temp file (the VLM client reads from disk).
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result: DiagnosisResult = rag.diagnose(tmp_path, question)
        return DiagnoseResponse(**result.to_dict())
    finally:
        Path(tmp_path).unlink(missing_ok=True)
