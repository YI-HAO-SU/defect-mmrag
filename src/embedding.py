"""SigLIP-based multimodal embedder.

SigLIP (Zhai et al., 2023) replaces CLIP's softmax contrastive loss with a sigmoid loss,
which decouples the loss from batch size — yielding stronger retrieval at small batches.
Both image and text are projected into the same 768-dim space (for siglip-base).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .config import settings


def _resolve_device() -> str:
    """Use the configured device, but fall back to CPU if CUDA is unavailable."""
    if settings.device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return settings.device


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoProcessor, AutoModel]:
    """Cached loader so the model is initialized exactly once per process."""
    device = _resolve_device()
    processor = AutoProcessor.from_pretrained(settings.embedding_model_id)
    model = (
        AutoModel.from_pretrained(settings.embedding_model_id)
        .to(device)
        .eval()
    )
    return processor, model


@torch.no_grad()
def embed_image(image: str | Path | Image.Image) -> list[float]:
    """Embed an image into the joint multimodal space.

    Args:
        image: file path or PIL Image.

    Returns:
        L2-normalized 768-dim vector as a plain list (Qdrant-ready).
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt").to(_resolve_device())
    out = model.get_image_features(**inputs)
    features = out if isinstance(out, torch.Tensor) else out.pooler_output

    # L2-normalize so cosine similarity == dot product downstream.
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()


@torch.no_grad()
def embed_text(text: str) -> list[float]:
    """Embed a text query into the same multimodal space."""
    processor, model = _load_model()
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ).to(_resolve_device())
    out = model.get_text_features(**inputs)
    features = out if isinstance(out, torch.Tensor) else out.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()
