#!/usr/bin/env bash
set -euo pipefail

# Dedicated Qwen3.5 + vLLM multimodal serving profile for SWE-Vision.
# This keeps the request format OpenAI-compatible while enabling the
# Qwen/vLLM-specific reasoning and tool-calling knobs the model expects.

MODEL="${QWEN35_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
HOST="${QWEN35_HOST:-0.0.0.0}"
PORT="${QWEN35_PORT:-8000}"
TP_SIZE="${QWEN35_TP_SIZE:-8}"
MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${QWEN35_GPU_MEMORY_UTILIZATION:-0.95}"
DOWNLOAD_DIR="${QWEN35_DOWNLOAD_DIR:-/mnt/localssd/hf-cache}"
VLLM_BIN="${QWEN35_VLLM_BIN:-/mnt/localssd/.venv/bin/vllm}"

export HF_TOKEN="${HF_TOKEN:-}"

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --download-dir "$DOWNLOAD_DIR"
