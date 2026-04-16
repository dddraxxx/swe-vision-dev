# SWE-Vision

<div align="center">
  <picture>
      <img src="./assets/logo_swev.png" width="30%">
  </picture>
</div>


<div align="center" style="line-height: 1;">

[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/UniPat-AI/SWE-Vision)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://unipat.ai/blog/SWE-Vision)

</div>

An agentic VLM (Vision Language Model) framework that gives a language model access to a **stateful notebook runtime**. The agent can iteratively write and execute Python code to process images, run computations, and produce visualizations. This local fork supports a safer `local_sandbox` mode by default, a plain `local` kernel for debugging, and optional Docker or Podman container runtimes when available.

`local_sandbox` is safer than plain local execution, but it is **not** equivalent to Docker. It uses `setpriv + unshare` for reduced privileges and user/pid/mount namespace isolation.

## Project Structure

```
SWE-Vision/
├── swe_vision/                  # Core library
│   ├── __init__.py              # Package exports
│   ├── config.py                # Constants, logging, tool definitions, system prompt
│   ├── kernel.py                # JupyterNotebookKernel — local, docker, podman runtimes
│   ├── image_utils.py           # Image encoding, MIME detection, OpenAI content parts
│   ├── file_manager.py          # NotebookFileManager — host ↔ container file sharing
│   ├── trajectory.py            # TrajectoryRecorder — saves full agent traces to disk
│   ├── agent.py                 # VLMToolCallAgent — agentic loop with tool calling
│   ├── cli.py                   # CLI entry point
│   └── eval_utils.py            # LLM judge prompt, answer extraction utilities
│
├── apps/                        # Standalone applications
│   ├── web_app.py               # ChatGPT-style web UI (Flask + SSE streaming)
│   └── trajectory_viewer.py     # Trajectory visualization dashboard (Flask)
│
├── env/                         # Docker environment (Dockerfile for the kernel)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"   # custom API endpoint
export OPENAI_MODEL="openai/gpt-5.2"                          # default model
export VLM_RUNTIME="local_sandbox"                            # default / recommended here
export VLM_ATTACH_IMAGES_TO_LLM="true"                        # set false for text-only host models
export OPENAI_REASONING_BACKEND="auto"                        # auto/openai/qwen3/gemma4/kimi_k2
```

### 3. Prepare the runtime

Recommended in this environment: use the local sandboxed kernel.

```bash
export VLM_RUNTIME="local_sandbox"
```

Debugging option: use the plain local kernel.

```bash
export VLM_RUNTIME="local"
```

Optional: if you have Docker available, you can still use the original container runtime:

```bash
export VLM_RUNTIME="docker"
docker build -t swe-vision -f ./env/Dockerfile ./env
```

Optional: if you have Podman available on this host, you can use the Podman runtime:

```bash
export VLM_RUNTIME="podman"
sudo podman build --isolation=chroot -t swe-vision ./env
```

On this machine, Podman requires `crun` and disabled cgroups, so the runtime uses:
`--runtime=crun --cgroups=disabled --network=host`.

### CharXiv sample workspace

To stage one official CharXiv validation reasoning example locally:

```bash
.venv/bin/python scripts/setup_charxiv_reasoning.py
```

This creates `workspaces/charxiv_reasoning/` with:
- `sample.json`
- `sample.jsonl`
- `prompt.txt`
- `answer.txt`
- the chart image for that question

To enter the same Podman/IPython environment the model-facing runtime uses:

```bash
scripts/podman_ipython.sh workspaces/charxiv_reasoning
```

### Qwen3.5 + vLLM setup

For a dedicated local host-model setup with Qwen3.5 and vLLM, use the bundled launch scripts:

```bash
cd /mnt/localssd/swe-vision
chmod +x scripts/launch_qwen35_vllm_mm.sh scripts/start_qwen35_vllm_tmux.sh
scripts/start_qwen35_vllm_tmux.sh
```

This launches:
- `Qwen/Qwen3.5-397B-A17B-FP8`
- OpenAI-compatible API on `http://127.0.0.1:8000/v1`
- Qwen-specific reasoning parser: `qwen3`
- Qwen tool-calling parser: `qwen3_coder`
- multimodal vLLM options: `--mm-encoder-tp-mode data` and `--mm-processor-cache-type shm`

Use these client-side environment variables with SWE-Vision:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=Qwen/Qwen3.5-397B-A17B-FP8
export OPENAI_REASONING_BACKEND=qwen3
export VLM_ATTACH_IMAGES_TO_LLM=true
export VLM_RUNTIME=podman
```

Then run the CharXiv sample:

```bash
bash run.sh workspaces/charxiv_reasoning/figure_0000.jpg "$(cat workspaces/charxiv_reasoning/prompt.txt)"
```

Useful checks:

```bash
curl http://127.0.0.1:8000/v1/models
tail -f logs/qwen35_host_vllm.log
tmux attach -t swevision-qwen35
```

### Gemma 4 + vLLM setup

Gemma 4 support in vLLM is moving quickly, so this setup uses a dedicated uv
environment with a recent vLLM nightly build.

```bash
cd /mnt/localssd
mkdir -p gemma4-vllm-host
cd gemma4-vllm-host
uv venv .venv
uv pip install --python .venv/bin/python -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python transformers==5.5.0
```

Then use the bundled launch scripts:

```bash
cd /mnt/localssd/swe-vision
chmod +x scripts/launch_gemma4_vllm_mm.sh scripts/start_gemma4_vllm_tmux.sh
scripts/start_gemma4_vllm_tmux.sh
```

Default local serving profile:
- model: `google/gemma-4-31B-it`
- OpenAI-compatible API on `http://127.0.0.1:8001/v1`
- tensor parallelism: `2`
- Gemma 4 reasoning + tool-calling parsers enabled

The launcher auto-caches the official Gemma 4 chat template into
`assets/tool_chat_template_gemma4.jinja` on first run.

If another host model already occupies the GPUs, the start script exits early
with a preflight warning instead of failing later during model load.

Use these client-side environment variables with SWE-Vision:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_MODEL=google/gemma-4-31B-it
export OPENAI_REASONING_BACKEND=gemma4
export VLM_ATTACH_IMAGES_TO_LLM=true
export VLM_RUNTIME=podman
```

Useful checks:

```bash
curl http://127.0.0.1:8001/v1/models
tail -f logs/gemma4_host_vllm.log
tmux attach -t swevision-gemma4
```

### Kimi K2.5 + local vLLM setup

When using the local Kimi host from `/mnt/localssd/kimi-k2.5-local`, keep the
model API local and run notebook tool execution in Podman:

```bash
export OPENAI_API_KEY=local
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_MODEL=kimi-k2.5-local
export OPENAI_REASONING_BACKEND=kimi_k2
export VLM_ATTACH_IMAGES_TO_LLM=true
export VLM_RUNTIME=podman
export VLM_HOST_WORK_DIR=/mnt/localssd/tmp/vlm_docker_workdir
export VLM_PODMAN_ROOT=/mnt/localssd/podman-root
export VLM_PODMAN_RUNROOT=/mnt/localssd/podman-runroot
```

Build the runtime image once:

```bash
sudo podman --root "$VLM_PODMAN_ROOT" --runroot "$VLM_PODMAN_RUNROOT" \
  build --isolation=chroot -t swe-vision:latest ./env
```

Then use the Kimi helper scripts:

```bash
bash scripts/smoke_kimi_local.sh
bash scripts/run_mira25_kimi_local.sh --max-items 1
```

### Qwen3.5 + local vLLM eval helpers

When the local Qwen vLLM host is already serving on `http://127.0.0.1:8001/v1`,
use the served model id and keep the notebook runtime in Podman:

```bash
export OPENAI_API_KEY=local
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_MODEL=qwen3.5-local
export OPENAI_REASONING_BACKEND=qwen3
export VLM_ATTACH_IMAGES_TO_LLM=true
export VLM_RUNTIME=podman
export VLM_HOST_WORK_DIR=/mnt/localssd/tmp/vlm_docker_workdir
export VLM_PODMAN_ROOT=/mnt/localssd/podman-root
export VLM_PODMAN_RUNROOT=/mnt/localssd/podman-runroot
```

Quick checks:

```bash
curl http://127.0.0.1:8001/v1/models
tail -f logs/*-smoke-qwen-local.log
```

Use the local Qwen helper scripts:

```bash
bash scripts/smoke_qwen_local.sh
bash scripts/run_mira25_qwen_local.sh --max-items 1
```

For the first full MIRA25 run, keep the wrapper default of `CONCURRENCY=1`.
Only raise concurrency after validating stability on this host.

Directory conventions for these helpers:
- wrappers live in `scripts/`
- one-off smoke and eval logs live in `logs/`
- eval outputs live in `eval_runs/`
- agent trajectories live in `trajectories/`


### 4. Run the agent (CLI)


We provide a script to run the agent with a single command.
```bash
bash run.sh
```


You can also run the agent manually.
```bash
# Single query with an image
python -m swe_vision.cli --image photo.png "What objects are in this image?"

# Multiple images
python -m swe_vision.cli -i img1.png -i img2.png "What is the difference between these two images?"
```


### 5. Run the Web UI

A ChatGPT-style interface with real-time streaming of the agent's reasoning, code execution, and results:

```bash
python apps/web_app.py --port 8080
# Open http://localhost:8080
```

![Web App Screenshot](./assets/web_app_screenshot.png)

### 6. Run batch evaluation

Prepare a JSONL file with fields:
- `id`
- `question`
- `answer`
- `image` or `images`

Example:

```json
{"id":"sample-chart-gap","question":"What is the performance gap between GPT5.2 and 6-year-olds from the chart? Answer with a number only.","answer":"46","image":"assets/test_image.png"}
```

Run:

```bash
cd /mnt/localssd/swe-vision
source .venv/bin/activate
python scripts/eval_jsonl.py --input eval_examples/test_image_eval.jsonl --output-dir eval_runs/sample
```

When the run has a known reference answer, include it in the saved trajectory
metadata when feasible. In practice, prefer storing both `eval_id` and
`ground_truth` for eval rows so the trajectory viewer and any later debugging
stay self-contained even if the original JSONL is not nearby. The bundled eval
runner already does this for batch evals.

For the 25-case MIRA recipe on a fresh machine, see
[`docs/mira25_other_machine.md`](./docs/mira25_other_machine.md).

The repo also carries two committed MIRA eval bundles for repeated local testing:

- [`eval_examples/mira25_mixed/mira_eval.jsonl`](./eval_examples/mira25_mixed/mira_eval.jsonl):
  the first 25 answer-bearing round-robin examples from MIRA.
- [`eval_examples/mira50_round2/mira_eval.jsonl`](./eval_examples/mira50_round2/mira_eval.jsonl):
  the next 50 answer-bearing round-robin examples, with zero overlap against
  `mira25_mixed`.

The staging helper supports deterministic continuation:

```bash
export HF_TOKEN=...
.venv/bin/python scripts/setup_mira_eval.py \
  --output-dir eval_examples/mira50_round2 \
  --count 50 \
  --skip 25
```

`rolling_dice_two` is not included in either committed bundle because its
current upstream rows have null `answer` values, and the staging rule keeps
only answer-bearing examples.

### 7. View trajectories

Every agent run saves a trajectory (JSON + images) to `./trajectories/`. Browse them with the viewer:

```bash
python apps/trajectory_viewer.py --port 5050
# Open http://localhost:5050
```

![Trajectory Viewer Screenshot](./assets/traj_viewer_screenshot.png)



## Architecture

```
                                User Query (+ images)
                                        │
                                        ▼
                                ┌──────────────────────┐
                                │   LLM (e.g. GPT-5.2) │◄───────────────────────┐
                                │                      │                        │
                                │   Tool Calls:        │                        │
                                │   ┌────────────────┐ │     ┌──────────────┐   │
                                │   │  execute_code  │─┼────►│Jupyter Kernel│   │
                                │   └────────────────┘ │     │  (Docker)    │   │
                                │   ┌────────────────┐ │     └──────┬───────┘   │
                                │   │    finish      │─┼──► Answer  │ (Output)  │
                                │   └────────────────┘ │            │           │
                                └──────────────────────┘    text + images ──────┘
```

**Key components:**

| Module | Responsibility |
|---|---|
| `config.py` | All constants, tool schemas, system prompt |
| `kernel.py` | Builds Docker image, starts container, manages Jupyter kernel via ZMQ |
| `agent.py` | Orchestrates the agentic loop: LLM calls → tool dispatch → result collection |
| `trajectory.py` | Records every step with timestamps, code, images; saves to JSON |
| `image_utils.py` | Base64 encoding, compression, OpenAI content part builders |
| `file_manager.py` | Copies files into the Docker mount so the kernel can access them |

## CLI Options

```
usage: python -m swe_vision.cli [-h] [--image IMAGE] [--interactive]
                                [--model MODEL] [--api-key API_KEY]
                                [--base-url BASE_URL]
                                [--max-iterations MAX_ITERATIONS]
                                [--save-trajectory SAVE_TRAJECTORY]
                                [--verbose] [--quiet]
                                [--reasoning | --no-reasoning]
                                [query]
```

| Flag | Description |
|---|---|
| `--image, -i` | Image file path (repeatable) |
| `--interactive` | Multi-turn interactive mode |
| `--model, -m` | Model name (default: `gpt-4o` or `$OPENAI_MODEL`) |
| `--reasoning / --no-reasoning` | Enable/disable extended reasoning |
| `--save-trajectory` | Custom trajectory output directory |
| `--quiet, -q` | Minimal console output |

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key for the LLM provider | *(required)* |
| `OPENAI_BASE_URL` | Custom API base URL | OpenAI default |
| `OPENAI_MODEL` | Default model name | `gpt-4o` |
| `VLM_RUNTIME` | Runtime backend: `local_sandbox`, `local`, `docker`, or `podman` | `local_sandbox` |
| `VLM_DOCKER_IMAGE` | Docker image name for the kernel | `swe-vision:latest` |
| `VLM_DOCKERFILE_DIR` | Path to the Dockerfile directory | `./env/` |
| `VLM_HOST_WORK_DIR` | Host-side working directory for file sharing | `/mnt/localssd/tmp/vlm_docker_workdir` |
| `VLM_PODMAN_ROOT` | Podman storage root (rootful) | `/mnt/localssd/podman-root` |
| `VLM_PODMAN_RUNROOT` | Podman runroot (rootful) | `/mnt/localssd/podman-runroot` |
| `VLM_WEB_SESSION_DIR` | Session storage for the web app | `/tmp` |

## Programmatic Usage

```python
import asyncio
from swe_vision import VLMToolCallAgent

async def main():
    agent = VLMToolCallAgent(
        model="openai/gpt-5.2",
        api_key="sk-...",
        reasoning=True,
    )
    try:
        answer = await agent.run(
            "Analyze this chart and summarize the trends",
            image_paths=["chart.png"],
        )
        print(answer)
    finally:
        await agent.cleanup()

asyncio.run(main())
```

## License

MIT
