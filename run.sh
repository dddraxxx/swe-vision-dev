#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Environment ────────────────────────────────────────────────
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set." >&2
  exit 1
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-openai/gpt-5.2}"
export VLM_RUNTIME="${VLM_RUNTIME:-local_sandbox}"
export VLM_ATTACH_IMAGES_TO_LLM="${VLM_ATTACH_IMAGES_TO_LLM:-true}"
export OPENAI_REASONING_BACKEND="${OPENAI_REASONING_BACKEND:-auto}"

# ─── Run agent with an image question ───────────────────────────
IMAGE_PATH="${1:-./assets/test_image.png}"
QUERY="${2:-What is the gap between GPT5.2 and 6-year-olds from the chart?}"

.venv/bin/python -m swe_vision.cli \
    --image "$IMAGE_PATH" \
    --model "$OPENAI_MODEL" \
    --reasoning \
    "$QUERY"
