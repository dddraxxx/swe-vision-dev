#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${GEMMA4_MODEL:-google/gemma-4-31B-it}"
HOST="${GEMMA4_HOST:-0.0.0.0}"
PORT="${GEMMA4_PORT:-8001}"
TP_SIZE="${GEMMA4_TP_SIZE:-2}"
DP_SIZE="${GEMMA4_DP_SIZE:-1}"
DP_SIZE_LOCAL="${GEMMA4_DP_SIZE_LOCAL:-}"
DP_BACKEND="${GEMMA4_DP_BACKEND:-}"
DIST_EXEC_BACKEND="${GEMMA4_DIST_EXEC_BACKEND:-}"
MAX_MODEL_LEN="${GEMMA4_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GEMMA4_GPU_MEMORY_UTILIZATION:-0.90}"
DOWNLOAD_DIR="${GEMMA4_DOWNLOAD_DIR:-/mnt/localssd/hf-cache}"
VLLM_BIN="${GEMMA4_VLLM_BIN:-/mnt/localssd/gemma4-vllm-host/.venv/bin/vllm}"
CHAT_TEMPLATE="${GEMMA4_CHAT_TEMPLATE:-${REPO_DIR}/assets/tool_chat_template_gemma4.jinja}"
CHAT_TEMPLATE_URL="${GEMMA4_CHAT_TEMPLATE_URL:-https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja}"
CUDA_DEVICES="${GEMMA4_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
LIMIT_MM_PER_PROMPT="${GEMMA4_LIMIT_MM_PER_PROMPT:-}"
MM_PROCESSOR_KWARGS="${GEMMA4_MM_PROCESSOR_KWARGS:-}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${GEMMA4_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
RUNTIME_DIR="${GEMMA4_RUNTIME_DIR:-/mnt/localssd/runtime/gemma4-vllm}"
TMP_DIR="${GEMMA4_TMPDIR:-${RUNTIME_DIR}/tmp}"
TRITON_CACHE_DIR="${GEMMA4_TRITON_CACHE_DIR:-${RUNTIME_DIR}/triton-cache}"
TORCHINDUCTOR_CACHE_DIR="${GEMMA4_TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_DIR}/torchinductor-cache}"

export HF_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p "${DOWNLOAD_DIR}" "${TMP_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  mkdir -p "$(dirname "${CHAT_TEMPLATE}")"
  if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
    curl -fsSL "${CHAT_TEMPLATE_URL}" -o "${CHAT_TEMPLATE}"
  fi
fi

args=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-auto-tool-choice
  --reasoning-parser gemma4
  --tool-call-parser gemma4
  --enable-prefix-caching
  --download-dir "$DOWNLOAD_DIR"
)

if [[ "${DP_SIZE}" != "1" ]]; then
  args+=(--data-parallel-size "$DP_SIZE")
fi

if [[ -n "${DP_SIZE_LOCAL}" ]]; then
  args+=(--data-parallel-size-local "$DP_SIZE_LOCAL")
fi

if [[ -n "${DP_BACKEND}" ]]; then
  args+=(--data-parallel-backend "$DP_BACKEND")
fi

if [[ -n "${DIST_EXEC_BACKEND}" ]]; then
  args+=(--distributed-executor-backend "$DIST_EXEC_BACKEND")
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  args+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
  args+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi

if [[ -n "${MM_PROCESSOR_KWARGS}" ]]; then
  args+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
fi

if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
  args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
fi

exec "$VLLM_BIN" "${args[@]}"
