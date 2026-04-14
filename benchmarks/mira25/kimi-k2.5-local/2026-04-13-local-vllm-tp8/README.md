# Kimi-K2.5 Local MIRA-25 Snapshot

- Date: `2026-04-13`
- Task: `MIRA-25 mixed`
- Model: `kimi-k2.5-local`
- Score: `5/25 = 0.20 exact match`
- Serving stack: local `vLLM nightly`, `tp=8`
- Runtime: `Podman`
- Reasoning backend: `kimi_k2`

This directory preserves the stable outputs from the completed run
`eval_runs/mira25_kimi_local_full_20260413-174140/`.

Command shape used for the eval:

```bash
OUTDIR=eval_runs/mira25_kimi_local_full_20260413-174140 \
LOG_FILE=logs/20260413-174140-mira25-kimi-full.log \
CONCURRENCY=1 \
MAX_ITERATIONS=50 \
bash scripts/run_mira25_kimi_local.sh
```

Tracked files:

- `summary.json` for aggregate metrics
- `predictions.jsonl` for per-case outputs

Intentionally omitted from this tracked snapshot:

- raw logs under `logs/`
- trajectories under `trajectories/`
- the original generated run directory under `eval_runs/`
