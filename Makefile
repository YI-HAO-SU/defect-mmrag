SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE    := docker compose -f docker/docker-compose.yml --env-file .env
DATA_DIR   ?= data/mvtec

# Set HF_CACHE_DIR so the vLLM volume mount is never blank.
ifeq ($(OS),Windows_NT)
    export HF_CACHE_DIR ?= $(subst \,/,$(USERPROFILE))/.cache/huggingface
else
    export HF_CACHE_DIR ?= $(HOME)/.cache/huggingface
endif

.PHONY: help up up-cpu build ingest down logs ps health

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  up        Full stack: Qdrant + vLLM (GPU) + app + ingest data"
	@echo "  up-cpu    Qdrant + app only (no GPU required)"
	@echo "  build     (Re)build the app container image"
	@echo "  ingest    Load defect images into Qdrant  [DATA_DIR=data/mvtec]"
	@echo "  down      Stop and remove all containers"
	@echo "  logs      Tail logs from all running services"
	@echo "  ps        Show container status"
	@echo "  health    Hit /health and pretty-print the response"

# ── Full GPU stack ─────────────────────────────────────────────────────────────
up: .env build
	@echo "==> Starting Qdrant..."
	$(COMPOSE) up -d qdrant
	@echo "==> Waiting for Qdrant to become healthy..."
	@until [[ "$$(docker inspect --format='{{.State.Health.Status}}' defect-mmrag-qdrant 2>/dev/null)" == "healthy" ]]; do sleep 2; done
	@echo "==> Starting vLLM  (first run downloads ~7 GB — grab a coffee)..."
	$(COMPOSE) --profile gpu up -d vllm
	@echo "==> Waiting for vLLM HTTP endpoint..."
	@until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 5; printf '.'; done && echo ""
	@$(MAKE) --no-print-directory ingest
	@echo "==> Starting FastAPI app..."
	$(COMPOSE) --profile full up -d app
	@echo ""
	@echo "Stack is up!"
	@echo "  API:              http://localhost:8080/diagnose"
	@echo "  Interactive docs: http://localhost:8080/docs"
	@echo "  Qdrant dashboard: http://localhost:6333/dashboard"

# ── CPU-only (no GPU / vLLM) ──────────────────────────────────────────────────
up-cpu: .env build
	@echo "==> Starting Qdrant + app (CPU mode, no vLLM)..."
	$(COMPOSE) up -d qdrant
	@until [[ "$$(docker inspect --format='{{.State.Health.Status}}' defect-mmrag-qdrant 2>/dev/null)" == "healthy" ]]; do sleep 2; done
	@$(MAKE) --no-print-directory ingest
	$(COMPOSE) --profile full up -d app
	@echo "App: http://localhost:8080/docs  (set VLLM_BASE_URL in .env for inference)"

build: .env
	$(COMPOSE) --profile ingest build app ingest

# ── Ingest defect images into Qdrant ──────────────────────────────────────────
ingest: .env
	@echo "==> Ingesting $(DATA_DIR) into Qdrant..."
	$(COMPOSE) --profile ingest run --rm ingest \
		python scripts/ingest.py --data_dir /app/$(DATA_DIR)

# ── Ops ───────────────────────────────────────────────────────────────────────
down:
	$(COMPOSE) --profile gpu --profile full --profile ingest down

logs:
	$(COMPOSE) --profile gpu --profile full logs -f

ps:
	$(COMPOSE) --profile gpu --profile full ps

health:
	@curl -s http://localhost:8080/health | python3 -m json.tool

# Auto-create .env from example when missing
.env:
	cp .env.example .env
	@echo ".env created — edit HF_TOKEN if your model is gated"