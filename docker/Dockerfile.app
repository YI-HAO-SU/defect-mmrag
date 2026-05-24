# Lightweight CPU image for the FastAPI app.
# The heavy lifting (vLLM) lives in a separate GPU process / container.
# SigLIP can run on CPU at low QPS; switch to a CUDA base image if you need
# embedding throughput.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-app.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

EXPOSE 8080

# Healthcheck so K8s/orchestrators can detect a wedged process.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]