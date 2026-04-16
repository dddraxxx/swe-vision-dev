#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swe_vision import VLMToolCallAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-Vision over a JSONL eval set.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id/question/answer/image or images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_image_paths(row: dict, base_dir: Path) -> list[str]:
    images = []
    if isinstance(row.get("image"), str):
        images.append(row["image"])
    if isinstance(row.get("images"), list):
        images.extend(p for p in row["images"] if isinstance(p, str))

    resolved = []
    for image in images:
        path = Path(image)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved.append(str(path))
    return resolved


def row_id(row: dict, index: int) -> str:
    return str(row.get("id", f"item-{index:05d}"))


async def run_row(args: argparse.Namespace, row: dict, base_dir: Path) -> dict:
    agent_kwargs = {
        "max_iterations": args.max_iterations,
        "verbose": False,
        "reasoning": args.reasoning,
    }
    if args.model is not None:
        agent_kwargs["model"] = args.model

    agent = VLMToolCallAgent(**agent_kwargs)
    ground_truth = str(row.get("answer", "")).strip()
    try:
        answer = await agent.run(
            row["question"],
            normalize_image_paths(row, base_dir),
            trajectory_metadata={
                "eval_id": row["id"],
                "ground_truth": ground_truth,
            },
        )
        prediction = str(answer).strip()
        error = None
    except Exception as exc:
        prediction = ""
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await agent.cleanup()

    record = {
        "id": row["id"],
        "question": row["question"],
        "ground_truth": ground_truth,
        "prediction": prediction,
        "exact_match": prediction == ground_truth,
    }
    if error is not None:
        record["error"] = error
    return record


async def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.input.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_items is not None:
        rows = rows[: args.max_items]
    rows = [
        {
            **row,
            "id": row_id(row, index),
        }
        for index, row in enumerate(rows, start=1)
    ]

    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    existing_records = []
    if predictions_path.exists():
        with predictions_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))

    completed_ids = {str(record.get("id")) for record in existing_records}
    pending_rows = [row for row in rows if row["id"] not in completed_ids]
    completed = len(existing_records)
    correct = sum(1 for record in existing_records if record.get("exact_match"))
    total_rows = len(rows)
    write_lock = asyncio.Lock()

    def write_summary() -> None:
        summary = {
            "num_items": completed,
            "target_items": total_rows,
            "exact_match": (correct / completed) if completed else 0.0,
            "predictions_path": str(predictions_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2))

    async def persist_record(out_handle, record: dict) -> None:
        nonlocal completed, correct
        async with write_lock:
            completed += 1
            correct += int(record["exact_match"])
            out_handle.write(json.dumps(record) + "\n")
            out_handle.flush()
            write_summary()
            status = " exact_match=" + str(record["exact_match"])
            if "error" in record:
                status += f" error={record['error']}"
            print(f"[{completed}/{total_rows}] {record['id']}{status}", flush=True)

    async def bounded_run(row: dict, sem: asyncio.Semaphore) -> dict:
        async with sem:
            return await run_row(args, row, args.input.parent)

    mode = "a" if existing_records else "w"
    with predictions_path.open(mode) as out:
        write_summary()
        if pending_rows:
            sem = asyncio.Semaphore(max(1, args.concurrency))
            tasks = [asyncio.create_task(bounded_run(row, sem)) for row in pending_rows]
            for task in asyncio.as_completed(tasks):
                record = await task
                await persist_record(out, record)

    summary = {
        "num_items": completed,
        "target_items": total_rows,
        "exact_match": (correct / completed) if completed else 0.0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
