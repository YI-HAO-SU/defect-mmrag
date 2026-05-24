"""Centralized configuration.

Using pydantic-settings so values can be overridden by env vars or a .env file
without touching code — a common MLOps pattern.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ----- Vector DB -----
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "defect_cases"

    # ----- Embedding model -----
    # SigLIP-base: 768-dim, sigmoid loss, better retrieval than CLIP at small batches.
    # Swap to "google/siglip-large-patch16-384" for higher quality (more VRAM).
    embedding_model_id: str = "google/siglip-base-patch16-224"
    embedding_dim: int = 768

    # ----- VLM (vLLM server) -----
    # On 12GB VRAM use Qwen2.5-VL-3B-Instruct (bf16).
    # On 24GB+ swap to Qwen/Qwen2.5-VL-7B-Instruct-AWQ (4-bit) for better quality.
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "dummy"  # vLLM ignores it but openai SDK requires non-empty

    # ----- RAG -----
    retrieval_top_k: int = 2  # How many historical cases to inject into the prompt
    generation_max_tokens: int = 512
    generation_temperature: float = 0.3  # Low temp for industrial diagnostics

    # ----- Runtime -----
    device: str = "cuda"  # "cuda" | "cpu"


settings = Settings()
