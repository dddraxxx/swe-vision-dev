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

    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    correct = 0
    total = 0

    with predictions_path.open("w") as out:
        for row in rows:
            agent_kwargs = {
                "max_iterations": args.max_iterations,
                "verbose": False,
                "reasoning": args.reasoning,
            }
            if args.model is not None:
                agent_kwargs["model"] = args.model

            agent = VLMToolCallAgent(**agent_kwargs)
            try:
                answer = await agent.run(
                    row["question"],
                    normalize_image_paths(row, args.input.parent),
                )
            finally:
                await agent.cleanup()

            ground_truth = str(row.get("answer", "")).strip()
            prediction = str(answer).strip()
            is_exact = prediction == ground_truth
            correct += int(is_exact)
            total += 1

            record = {
                "id": row.get("id", f"item-{total:05d}"),
                "question": row["question"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": is_exact,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{total}/{len(rows)}] {record['id']} exact_match={is_exact}", flush=True)

    summary = {
        "num_items": total,
        "exact_match": (correct / total) if total else 0.0,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
