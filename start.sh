#!/usr/bin/env bash
# start.sh — one-click launcher for defect-mmrag (alternative to `make up`)
set -euo pipefail

COMPOSE="docker compose -f docker/docker-compose.yml --env-file .env"
DATA_DIR="${DATA_DIR:-data/mvtec}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")"

# ── .env ──────────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ".env created — edit HF_TOKEN if your model is gated"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo "==> Building app image..."
$COMPOSE build app

# ── Qdrant ────────────────────────────────────────────────────────────────────
echo "==> Starting Qdrant..."
$COMPOSE up -d qdrant

echo "==> Waiting for Qdrant to become healthy..."
until [[ "$(docker inspect --format='{{.State.Health.Status}}' defect-mmrag-qdrant 2>/dev/null)" == "healthy" ]]; do
  sleep 2
done

# ── vLLM (GPU) ────────────────────────────────────────────────────────────────
echo "==> Starting vLLM (first run downloads ~7 GB)..."
$COMPOSE --profile gpu up -d vllm

echo "==> Waiting for vLLM HTTP endpoint..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 5; printf '.'
done
echo ""

# ── Ingest ────────────────────────────────────────────────────────────────────
echo "==> Ingesting ${DATA_DIR} into Qdrant..."
$COMPOSE --profile ingest run --rm ingest \
  python scripts/ingest.py --data_dir "/app/${DATA_DIR}"

# ── App ───────────────────────────────────────────────────────────────────────
echo "==> Starting FastAPI app..."
$COMPOSE --profile full up -d app

echo ""
echo "Stack is up!"
echo "  API:              http://localhost:8080/diagnose"
echo "  Interactive docs: http://localhost:8080/docs"
echo "  Qdrant dashboard: http://localhost:6333/dashboard"