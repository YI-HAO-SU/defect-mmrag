"""Smoke tests. Run from repo root: pytest tests/

These don't require the vLLM server but DO require GPU + model download.
Mark them appropriately so CI can skip on CPU-only runners.
"""
from __future__ import annotations

import pytest
import torch

from src.config import settings


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_image_embedding_shape(tmp_path) -> None:
    from PIL import Image

    from src.embedding import embed_image

    # Create a dummy 224x224 RGB image.
    img_path = tmp_path / "dummy.png"
    Image.new("RGB", (224, 224), color=(128, 64, 200)).save(img_path)

    vec = embed_image(img_path)
    assert isinstance(vec, list)
    assert len(vec) == settings.embedding_dim
    # L2 norm should be ~1.0 after normalization.
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_text_embedding_shape() -> None:
    from src.embedding import embed_text

    vec = embed_text("a scratch on a metal surface")
    assert len(vec) == settings.embedding_dim
