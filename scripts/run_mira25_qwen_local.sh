#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

mkdir -p eval_runs logs

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTDIR:-eval_runs/mira25_qwen_local_${STAMP}}"
LOG_FILE="${LOG_FILE:-logs/${STAMP}-mira25-qwen-local.log}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.5-local}"
export OPENAI_REASONING_BACKEND="${OPENAI_REASONING_BACKEND:-qwen3}"
export VLM_RUNTIME="${VLM_RUNTIME:-podman}"
export VLM_ATTACH_IMAGES_TO_LLM="${VLM_ATTACH_IMAGES_TO_LLM:-true}"
export VLM_HOST_WORK_DIR="${VLM_HOST_WORK_DIR:-/mnt/localssd/tmp/vlm_docker_workdir}"
export VLM_PODMAN_ROOT="${VLM_PODMAN_ROOT:-/mnt/localssd/podman-root}"
export VLM_PODMAN_RUNROOT="${VLM_PODMAN_RUNROOT:-/mnt/localssd/podman-runroot}"

{
  echo "[run] output_dir=${OUTDIR}"
  echo "[run] log_file=${LOG_FILE}"
  echo "[run] base_url=${OPENAI_BASE_URL}"
  echo "[run] model=${OPENAI_MODEL}"
  echo "[run] reasoning_backend=${OPENAI_REASONING_BACKEND}"
  echo "[run] runtime=${VLM_RUNTIME}"
  echo "[run] concurrency=${CONCURRENCY:-1}"

  .venv/bin/python scripts/eval_jsonl.py \
    --input eval_examples/mira25_mixed/mira_eval.jsonl \
    --output-dir "$OUTDIR" \
    --model "$OPENAI_MODEL" \
    --max-iterations "${MAX_ITERATIONS:-50}" \
    --concurrency "${CONCURRENCY:-1}" \
    "$@"
} 2>&1 | tee "$LOG_FILE"
