#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

SESSION_NAME="${QWEN35_TMUX_SESSION:-swevision-qwen35}"
LOG_DIR="${QWEN35_LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${QWEN35_LOG_FILE:-${LOG_DIR}/qwen35_host_vllm.log}"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME" \
  "bash -lc '${SCRIPT_DIR}/launch_qwen35_vllm_mm.sh 2>&1 | tee \"${LOG_FILE}\"'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
