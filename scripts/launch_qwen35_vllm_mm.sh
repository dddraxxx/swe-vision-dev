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
RUNTIME_DIR="${QWEN35_RUNTIME_DIR:-/mnt/localssd/runtime/qwen35-vllm}"
TMP_DIR="${QWEN35_TMPDIR:-${RUNTIME_DIR}/tmp}"
TRITON_CACHE_DIR="${QWEN35_TRITON_CACHE_DIR:-${RUNTIME_DIR}/triton-cache}"
TORCHINDUCTOR_CACHE_DIR="${QWEN35_TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_DIR}/torchinductor-cache}"

export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p "$DOWNLOAD_DIR" "$TMP_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

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
