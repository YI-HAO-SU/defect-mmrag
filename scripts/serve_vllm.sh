#!/usr/bin/env bash
# Launch vLLM serving Qwen2.5-VL-3B on RTX 3060 (12GB).
#
# Key flags explained (interview gold):
#   --dtype bfloat16              bf16 has wider dynamic range than fp16 → fewer NaNs
#                                 with VLMs. 3060 supports it natively.
#   --max-model-len 4096          Caps prompt+gen length. Lower = less KV-cache VRAM.
#   --limit-mm-per-prompt image=3 We send query + up to 2 retrieved images = 3.
#   --gpu-memory-utilization 0.85 Leave headroom; default 0.90 OOMs on 12GB cards.
#   --enable-prefix-caching       Reuse KV for shared system prompt across requests.
#   --max-num-seqs 4              Cap concurrency to avoid OOM on small GPUs.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --max-num-seqs 4 \
    --port "$PORT"
