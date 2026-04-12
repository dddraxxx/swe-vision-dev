#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


DATASET_REPO = "YiyangAiLab/MIRA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a local MIRA eval subset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument(
        "--strategy",
        choices=["round_robin"],
        default="round_robin",
        help="How to sample examples across tasks.",
    )
    return parser.parse_args()


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to stage MIRA locally.")
    return token


def repo_files(token: str) -> list[str]:
    return list_repo_files(DATASET_REPO, repo_type="dataset", token=token)


def task_names(files: list[str]) -> list[str]:
    tasks = sorted(
        {
            path.split("/")[0]
            for path in files
            if "/" in path and not path.startswith(".")
        }
    )
    return tasks


def task_jsonl_path(task: str, files: list[str]) -> str:
    candidates = [
        path
        for path in files
        if path.startswith(f"{task}/") and path.endswith(".jsonl")
    ]
    if not candidates:
        raise FileNotFoundError(f"No JSONL found for task {task}")
    return sorted(candidates)[0]


def resolve_image_repo_path(task: str, image_path: str, uid: int, files: list[str]) -> str:
    normalized = image_path.lstrip("./")
    if normalized in files:
        return normalized

    candidate_dir = f"{task}/image/"
    task_images = [path for path in files if path.startswith(candidate_dir)]
    image_name = Path(normalized).name
    stem = Path(image_name).stem

    exact_stem = [path for path in task_images if Path(path).stem == stem]
    if exact_stem:
        return sorted(exact_stem)[0]

    uid_stem = [path for path in task_images if Path(path).stem in {str(uid), f"{task}{uid}"}]
    if uid_stem:
        return sorted(uid_stem)[0]

    raise FileNotFoundError(f"Could not resolve image for task={task} uid={uid} path={image_path}")


def extract_answer(row: dict) -> str | None:
    answer = row.get("answer")
    if answer is None:
        return None
    return str(answer)


def load_rows(task: str, token: str, files: list[str]) -> list[dict]:
    jsonl_path = hf_hub_download(
        DATASET_REPO,
        task_jsonl_path(task, files),
        repo_type="dataset",
        token=token,
    )
    rows = []
    with open(jsonl_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if extract_answer(row) is not None:
                    rows.append(row)
    return rows


def build_round_robin_selection(
    tasks: list[str], count: int, token: str, files: list[str]
) -> list[tuple[str, dict]]:
    task_rows = {task: load_rows(task, token, files) for task in tasks}
    positions = {task: 0 for task in tasks}
    selected: list[tuple[str, dict]] = []

    while len(selected) < count:
        made_progress = False
        for task in tasks:
            idx = positions[task]
            rows = task_rows[task]
            if idx >= len(rows):
                continue
            selected.append((task, rows[idx]))
            positions[task] += 1
            made_progress = True
            if len(selected) >= count:
                break
        if not made_progress:
            break

    return selected


def stage_selection(
    output_dir: Path, selection: list[tuple[str, dict]], token: str, files: list[str]
) -> None:
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    manifest = []
    for task, row in selection:
        uid = row["uid"]
        image_repo_path = resolve_image_repo_path(task, row["image_path"], uid, files)
        image_cache_path = hf_hub_download(
            DATASET_REPO,
            image_repo_path,
            repo_type="dataset",
            token=token,
        )
        local_name = f"{task}_uid{uid}{Path(image_repo_path).suffix or '.png'}"
        local_path = images_dir / local_name
        shutil.copy2(image_cache_path, local_path)

        example_id = f"{task}-uid{uid}"
        records.append(
            {
                "id": example_id,
                "question": row["question"],
                "answer": extract_answer(row),
                "image": f"images/{local_name}",
            }
        )
        manifest.append(
            {
                "id": example_id,
                "task": task,
                "uid": uid,
                "question": row["question"],
                "answer": extract_answer(row),
                "image_repo_path": image_repo_path,
                "local_image": f"images/{local_name}",
            }
        )

    input_jsonl = output_dir / "mira_eval.jsonl"
    with input_jsonl.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                f"Dataset: {DATASET_REPO}",
                f"Num examples: {len(records)}",
                "Strategy: round_robin",
                "Input file: mira_eval.jsonl",
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    token = require_hf_token()
    files = repo_files(token)
    tasks = task_names(files)
    selection = build_round_robin_selection(tasks, args.count, token, files)
    stage_selection(args.output_dir, selection, token, files)

    summary = {
        "dataset": DATASET_REPO,
        "num_tasks": len(tasks),
        "num_examples": len(selection),
        "output_dir": str(args.output_dir),
        "input_jsonl": str(args.output_dir / "mira_eval.jsonl"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
