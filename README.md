# Defect-MMRAG: Industrial Multimodal RAG for Defect Diagnosis

End-to-end multimodal RAG system for industrial defect diagnosis, built to demonstrate
production-grade GenAI engineering skills for AI/ML Application Engineer roles.

## Architecture

```
            ┌─────────────────────────────────────────┐
            │  Client (FastAPI / Streamlit / curl)    │
            └────────────────────┬────────────────────┘
                                 │ POST /diagnose
                                 │ {image, question}
                                 ▼
            ┌─────────────────────────────────────────┐
            │  RAG Orchestrator (src/rag.py)          │
            └───┬──────────────────────────────┬──────┘
                │ 1. embed(query_image)        │ 3. generate(prompt + retrieved)
                ▼                              ▼
        ┌───────────────┐              ┌────────────────────┐
        │ SigLIP        │              │ vLLM Server        │
        │ (CLIP-like)   │              │ Qwen2.5-VL-3B      │
        └───────┬───────┘              └────────────────────┘
                │ 2. search(top_k)
                ▼
        ┌───────────────┐
        │ Qdrant        │  ← seeded with MVTec AD cases (scripts/ingest.py)
        │ (vector DB)   │
        └───────────────┘
```

## Why these choices

| Component         | Choice                       | Reason                                                                                         |
| ----------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Inference engine  | **vLLM**                     | PagedAttention + continuous batching → 10-20× throughput vs HF transformers                    |
| VLM               | **Qwen2.5-VL-3B-Instruct**   | Fits 12GB VRAM at bf16; strong CN/EN; SOTA among small open VLMs                               |
| Embedding         | **SigLIP-base-patch16-224**  | Sigmoid loss > softmax for retrieval; small-batch friendly                                     |
| Vector DB         | **Qdrant**                   | Rust, single-binary, HNSW index, payload filtering, production-ready                           |
| Demo data         | **MVTec AD**                 | Industry-standard defect dataset; transferable to semiconductor wafer scenarios                |
| Distance metric   | **Cosine**                   | Embeddings are L2-normalized; equivalent to dot product but [-1, 1] is more interpretable      |

## Hardware

- **Tested on:** RTX 3060 12GB
- **VRAM budget:** ~7GB for Qwen2.5-VL-3B (bf16) + ~1GB for SigLIP + ~2GB KV cache
- **For 24GB+ GPUs:** swap to `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` in `src/config.py`

## Quickstart

**Prerequisites:** Docker + NVIDIA Container Toolkit (GPU)

```bash
git clone <repo> && cd defect-mmrag
```

**Linux / Mac / WSL2** — requires GNU `make` and `bash`
```bash
make up     # build → qdrant → vllm → ingest data → app
```

**Windows (PowerShell, no WSL2)**
```powershell
./start.ps1
```

Once the stack is up:

| Service | URL |
|---|---|
| REST API | http://localhost:8080/diagnose |
| Interactive docs (Swagger) | http://localhost:8080/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Common `make` targets

```bash
make up-cpu     # no GPU — skips vLLM (point VLLM_BASE_URL in .env to an external server)
make ingest     # re-seed the vector DB from data/mvtec
make down       # stop and remove all containers
make logs       # tail all service logs
make health     # call /health and pretty-print the JSON
make help       # list all targets
```

No `make`? Run `bash start.sh` instead — it does the same thing.

### Manual step-by-step (reference)

```bash
# Infrastructure
docker compose -f docker/docker-compose.yml up -d qdrant
docker compose -f docker/docker-compose.yml --profile gpu up -d vllm

# Data (generate synthetic if you don't have MVTec AD)
python scripts/generate_sample_data.py
python scripts/ingest.py --data_dir ./data/mvtec

# App
docker compose -f docker/docker-compose.yml --profile full up -d app
# or locally: uvicorn src.app:app --host 0.0.0.0 --port 8080
```

## Repo layout

```
wafer-mmrag/
├── Makefile                     # one-click deployment: make up
├── start.sh                     # bash alternative to make
├── README.md
├── requirements.txt
├── docker/
│   ├── docker-compose.yml       # Qdrant + vLLM (gpu) + app (full) + ingest profiles
│   └── Dockerfile.app           # FastAPI + ingest image (CPU torch)
├── src/
│   ├── config.py                # Centralized config (model IDs, dims, hosts)
│   ├── embedding.py             # SigLIP image/text embedder
│   ├── vector_store.py          # Qdrant wrapper
│   ├── llm_client.py            # vLLM (OpenAI-compatible) client
│   ├── rag.py                   # End-to-end pipeline
│   └── app.py                   # FastAPI service
├── scripts/
│   ├── download_mvtec.sh        # Pull a small subset of MVTec AD
│   ├── generate_sample_data.py  # Synthetic data (no download needed)
│   ├── ingest.py                # Build the Qdrant case DB
│   ├── serve_vllm.sh            # vLLM launch with tuned flags
│   └── demo.py                  # CLI demo
└── tests/
    └── test_embedding.py        # Smoke tests
```

## Deployment architecture

The stack runs as three independent services, already separated by Docker Compose profiles:

```
[Ingress / load balancer]
        │
        ▼
[FastAPI app]  ──→  [Qdrant]
        │
        ▼
[vLLM / Qwen2.5-VL]  (GPU)
```

| Service | Docker Compose | K8s equivalent | Scales how |
|---|---|---|---|
| FastAPI app | `profile: full` | `Deployment` + HPA | Horizontal — stateless |
| vLLM | `profile: gpu` | `Deployment` + GPU nodeSelector | Add GPU nodes + LoadBalancer |
| Qdrant | always-on | `StatefulSet` + PVC | Vertical / Qdrant Cloud |

All services communicate over HTTP and are configured via env vars (12-factor), so the same images run unchanged in Compose, K8s, or cloud-managed environments.


