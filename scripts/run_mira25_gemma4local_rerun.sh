#!/usr/bin/env bash
set -euo pipefail
cd /mnt/localssd/swe-vision
source /home/colligo/.bashrc >/dev/null 2>&1
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="eval_runs/mira25_gemma4local_${STAMP}"
LOG="logs/${STAMP}-mira25-gemma4local.log"
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL='http://127.0.0.1:8001/v1'
export OPENAI_MODEL='google/gemma-4-31B-it'
export OPENAI_REASONING_BACKEND='gemma4'
export VLM_RUNTIME='podman'
export VLM_HOST_WORK_DIR='/mnt/localssd/tmp/vlm_docker_workdir'
.venv/bin/python scripts/eval_jsonl.py \
  --input eval_examples/mira25_mixed/mira_eval.jsonl \
  --output-dir "$OUTDIR" \
  --model 'google/gemma-4-31B-it' \
  --max-iterations 50 \
  2>&1 | tee "$LOG"
