# Run MIRA25 on another machine

This note documents the 25-case MIRA eval bundle that already lives in this repo,
and the exact steps used to run it on another host.

## What the 25 cases are

The case list is committed here:

- `eval_examples/mira25_mixed/mira_eval.jsonl`

The matching image files are committed here:

- `eval_examples/mira25_mixed/images/`

So for these 25 cases, a fresh clone of the repo is enough. No separate dataset
download is needed.

The 25 case IDs are:

```text
billiards-uid1
convex_hull-uid0
cubes_count-uid1
cubes_missing-uid1
defuse_a_bomb-uid1
electric_charge-uid1
gear_rotation-uid1.1
localizer-uid1
mirror_clock-uid0
mirror_pattern-uid0
multi_piece_puzzle-uid1
overlap-uid0
paper_airplane-uid1
puzzle-uid1
rolling_dice_sum-uid1
rolling_dice_top-uid1
trailer_cubes_count-uid1
trailer_cubes_missing-uid1
unfolded_cube-uid0
billiards-uid2
convex_hull-uid1
cubes_count-uid2
cubes_missing-uid2
defuse_a_bomb-uid2
electric_charge-uid2
```

## Fresh machine setup

```bash
git clone git@github.com:dddraxxx/swe-vision-dev.git
cd swe-vision
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Podman runtime setup

Build the Podman runtime image once:

```bash
export VLM_RUNTIME=podman
export VLM_HOST_WORK_DIR=/mnt/localssd/tmp/vlm_docker_workdir
export VLM_PODMAN_ROOT=/mnt/localssd/podman-root
export VLM_PODMAN_RUNROOT=/mnt/localssd/podman-runroot

sudo podman --root "$VLM_PODMAN_ROOT" --runroot "$VLM_PODMAN_RUNROOT" \
  build --isolation=chroot -t swe-vision:latest ./env
```

On the machine used during development, putting both the host workdir and
Podman storage on `/mnt/localssd` avoided the root-filesystem quota problems
that showed up with `/home` and `/var/lib/containers`.

## OpenRouter Qwen 3.6 setup

```bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="qwen/qwen3.6-plus"
```

## Run the 25-case eval

```bash
python scripts/eval_jsonl.py \
  --input eval_examples/mira25_mixed/mira_eval.jsonl \
  --output-dir eval_runs/mira25_qwen36plus \
  --model qwen/qwen3.6-plus \
  --max-iterations 50 \
  --reasoning \
  --concurrency 4
```

## Notes

- `--concurrency 4` was stable on the original machine.
- `--concurrency 25` launched all cases but stalled workers and left many
  long-lived containers, so it is not the recommended setting.
- The eval runner is resumable. Re-run the same command with the same
  `--output-dir` and it skips any rows already written to `predictions.jsonl`.

For long runs, prefer `tmux` and a log file:

```bash
tmux new -s mira25-qwen36plus \
  "cd /path/to/swe-vision && source .venv/bin/activate && python scripts/eval_jsonl.py \
    --input eval_examples/mira25_mixed/mira_eval.jsonl \
    --output-dir eval_runs/mira25_qwen36plus \
    --model qwen/qwen3.6-plus \
    --max-iterations 50 \
    --reasoning \
    --concurrency 4 | tee logs/mira25-qwen36plus.log"
```
