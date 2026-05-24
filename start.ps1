#Requires -Version 5.1
<#
.SYNOPSIS
    One-click launcher for defect-mmrag on Windows PowerShell.
    Equivalent to `make up` on Linux/Mac.

.PARAMETER DataDir
    Relative path to the defect image dataset. Default: data/mvtec

.PARAMETER CpuOnly
    Skip vLLM startup (no GPU required). Set VLLM_BASE_URL in .env manually.

.PARAMETER Down
    Stop and remove all containers.

.PARAMETER Ingest
    Re-seed the Qdrant vector DB only (Qdrant must already be running).

.PARAMETER Health
    Call /health and print the result.

.EXAMPLE
    .\start.ps1              # full GPU stack
    .\start.ps1 -CpuOnly    # no GPU
    .\start.ps1 -Down        # stop everything
    .\start.ps1 -Ingest      # re-ingest data
    .\start.ps1 -Health      # check health
#>
param(
    [string]$DataDir = "data/mvtec",
    [switch]$CpuOnly,
    [switch]$Down,
    [switch]$Ingest,
    [switch]$Health
)

$ErrorActionPreference = "Stop"
$COMPOSE = "docker compose -f docker/docker-compose.yml --env-file .env"

# HuggingFace cache — reuse host cache so model downloads survive restarts.
# Also patch .env if HF_CACHE_DIR is left blank (common on first run).
if (-not $env:HF_CACHE_DIR) {
    $env:HF_CACHE_DIR = ("$env:USERPROFILE\.cache\huggingface") -replace '\\', '/'
}
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "HF_CACHE_DIR=\s*`n") {
        $envContent = $envContent -replace "HF_CACHE_DIR=", "HF_CACHE_DIR=$($env:HF_CACHE_DIR)"
        $envContent | Set-Content ".env" -Encoding utf8 -NoNewline
    }
}

Set-Location $PSScriptRoot

function Wait-ContainerHealthy($name) {
    Write-Host "==> Waiting for $name to become healthy..." -NoNewline
    while ($true) {
        $status = docker inspect --format='{{.State.Health.Status}}' $name 2>$null
        if ($status -eq "healthy") { Write-Host " ready."; return }
        Write-Host "." -NoNewline
        Start-Sleep 2
    }
}

function Wait-HttpReady($url) {
    Write-Host "==> Waiting for $url ..." -NoNewline
    while ($true) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($r.StatusCode -lt 400) { Write-Host " ready."; return }
        } catch {}
        Write-Host "." -NoNewline
        Start-Sleep 5
    }
}

# ── down ──────────────────────────────────────────────────────────────────────
if ($Down) {
    Write-Host "==> Stopping all services..."
    Invoke-Expression "$COMPOSE --profile gpu --profile full --profile ingest down"
    exit
}

# ── health ────────────────────────────────────────────────────────────────────
if ($Health) {
    $r = Invoke-RestMethod -Uri "http://localhost:8080/health"
    $r | ConvertTo-Json
    exit
}

# ── .env ──────────────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env created from .env.example — edit HF_TOKEN if your model is gated"
}

# ── ingest only ───────────────────────────────────────────────────────────────
if ($Ingest) {
    Write-Host "==> Ingesting $DataDir into Qdrant..."
    Invoke-Expression "$COMPOSE --profile ingest run --rm ingest python scripts/ingest.py --data_dir /app/$DataDir"
    exit
}

# ── build ─────────────────────────────────────────────────────────────────────
Write-Host "==> Building app image..."
Invoke-Expression "$COMPOSE build app"

# ── qdrant ────────────────────────────────────────────────────────────────────
Write-Host "==> Starting Qdrant..."
Invoke-Expression "$COMPOSE up -d qdrant"
Wait-ContainerHealthy "defect-mmrag-qdrant"

# ── vllm (skip in cpu-only mode) ──────────────────────────────────────────────
if (-not $CpuOnly) {
    Write-Host "==> Starting vLLM (first run downloads ~7 GB — this will take a while)..."
    Invoke-Expression "$COMPOSE --profile gpu up -d vllm"
    Wait-HttpReady "http://localhost:8000/health"
}

# ── ingest ────────────────────────────────────────────────────────────────────
Write-Host "==> Ingesting $DataDir into Qdrant..."
Invoke-Expression "$COMPOSE --profile ingest run --rm ingest python scripts/ingest.py --data_dir /app/$DataDir"

# ── app ───────────────────────────────────────────────────────────────────────
Write-Host "==> Starting FastAPI app..."
Invoke-Expression "$COMPOSE --profile full up -d app"

Write-Host ""
Write-Host "Stack is up!"
Write-Host "  API:              http://localhost:8080/diagnose"
Write-Host "  Interactive docs: http://localhost:8080/docs"
Write-Host "  Qdrant dashboard: http://localhost:6333/dashboard"
