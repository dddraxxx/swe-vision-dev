#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

SESSION_NAME="${GEMMA4_TMUX_SESSION:-swevision-gemma4}"
LOG_DIR="${GEMMA4_LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${GEMMA4_LOG_FILE:-${LOG_DIR}/gemma4_host_vllm.log}"
GPU_BUSY_THRESHOLD_MIB="${GEMMA4_GPU_BUSY_THRESHOLD_MIB:-70000}"

mkdir -p "$LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  total_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  busy_gpus="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v threshold="${GPU_BUSY_THRESHOLD_MIB}" '$1 + 0 >= threshold {count++} END {print count + 0}')"
  if [[ "${total_gpus}" != "0" ]] && [[ "${busy_gpus}" == "${total_gpus}" ]]; then
    echo "All ${total_gpus} GPUs already look occupied (>= ${GPU_BUSY_THRESHOLD_MIB} MiB each)."
    echo "Stop the current host model first, or set GEMMA4_CUDA_VISIBLE_DEVICES and GEMMA4_TP_SIZE for a free subset."
    exit 1
  fi
fi

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

LAUNCH_ENV=()
for var in \
  HF_TOKEN \
  GEMMA4_MODEL \
  GEMMA4_HOST \
  GEMMA4_PORT \
  GEMMA4_TP_SIZE \
  GEMMA4_DP_SIZE \
  GEMMA4_DP_SIZE_LOCAL \
  GEMMA4_DP_BACKEND \
  GEMMA4_DIST_EXEC_BACKEND \
  GEMMA4_MAX_MODEL_LEN \
  GEMMA4_GPU_MEMORY_UTILIZATION \
  GEMMA4_DOWNLOAD_DIR \
  GEMMA4_VLLM_BIN \
  GEMMA4_CHAT_TEMPLATE \
  GEMMA4_CHAT_TEMPLATE_URL \
  GEMMA4_CUDA_VISIBLE_DEVICES \
  GEMMA4_LIMIT_MM_PER_PROMPT \
  GEMMA4_MM_PROCESSOR_KWARGS \
  GEMMA4_DEFAULT_CHAT_TEMPLATE_KWARGS
do
  if [[ -n "${!var:-}" ]]; then
    LAUNCH_ENV+=("${var}=$(printf '%q' "${!var}")")
  fi
done

tmux new-session -d -s "$SESSION_NAME" \
  "bash -lc '${LAUNCH_ENV[*]} ${SCRIPT_DIR}/launch_gemma4_vllm_mm.sh 2>&1 | tee \"${LOG_FILE}\"'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
