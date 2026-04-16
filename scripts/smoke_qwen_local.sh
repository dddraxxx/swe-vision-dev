#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs trajectories

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/${STAMP}-smoke-qwen-local.log}"
TRAJ_DIR="${TRAJ_DIR:-trajectories/qwen_local_smoke}"
IMAGE_PATH="${1:-${REPO_DIR}/assets/test_image.png}"
QUERY="${2:-Use Python to inspect this image, then answer the question.}"

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
  echo "[smoke] image=${IMAGE_PATH}"
  echo "[smoke] trajectory_dir=${TRAJ_DIR}"
  echo "[smoke] runtime=${VLM_RUNTIME}"
  echo "[smoke] base_url=${OPENAI_BASE_URL}"
  echo "[smoke] model=${OPENAI_MODEL}"
  echo "[smoke] reasoning_backend=${OPENAI_REASONING_BACKEND}"
  .venv/bin/python -m swe_vision.cli \
    --image "$IMAGE_PATH" \
    --model "$OPENAI_MODEL" \
    --save-trajectory "$TRAJ_DIR" \
    --reasoning \
    "$QUERY"
} 2>&1 | tee "$LOG_FILE"
