#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REASONING_URL = "https://raw.githubusercontent.com/princeton-nlp/CharXiv/main/data/reasoning_val.json"
METADATA_URL = "https://raw.githubusercontent.com/princeton-nlp/CharXiv/main/data/image_metadata_val.json"
IMAGES_ZIP_URL = "https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one CharXiv reasoning example for local SWE-Vision runs.")
    parser.add_argument("--figure-id", type=int, default=0, help="Validation figure id to extract.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "workspaces" / "charxiv_reasoning",
        help="Workspace that will hold the chart image and sample prompt.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "charxiv",
        help="Directory for downloaded CharXiv metadata and zip archives.",
    )
    return parser.parse_args()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    reasoning_path = download(REASONING_URL, cache_dir / "reasoning_val.json")
    metadata_path = download(METADATA_URL, cache_dir / "image_metadata_val.json")
    images_zip_path = download(IMAGES_ZIP_URL, cache_dir / "images.zip")

    reasoning_data = json.loads(reasoning_path.read_text())
    metadata_data = json.loads(metadata_path.read_text())

    figure_key = str(args.figure_id)
    if figure_key not in reasoning_data:
        raise SystemExit(f"Figure id {args.figure_id} not found in reasoning_val.json")
    if figure_key not in metadata_data:
        raise SystemExit(f"Figure id {args.figure_id} not found in image_metadata_val.json")

    qa = reasoning_data[figure_key]
    meta = metadata_data[figure_key]
    archive_member = meta["figure_path"]
    archive_name = Path(archive_member).name
    archive_suffix = Path(archive_member).suffix or ".jpg"
    zip_member = f"{args.figure_id}{archive_suffix}"
    image_name = f"figure_{args.figure_id:04d}{archive_suffix}"
    image_out = output_dir / image_name

    with zipfile.ZipFile(images_zip_path) as zf:
        if zip_member not in zf.namelist():
            if archive_name in zf.namelist():
                zip_member = archive_name
            elif archive_member in zf.namelist():
                zip_member = archive_member
            else:
                raise SystemExit(
                    f"Could not find figure {args.figure_id} in images.zip using {zip_member} or {archive_name}"
                )
        with zf.open(zip_member) as src, image_out.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    record = {
        "id": f"charxiv-val-{args.figure_id}",
        "source": "princeton-nlp/CharXiv",
        "split": "val",
        "mode": "reasoning",
        "figure_id": args.figure_id,
        "title": meta["title"],
        "caption": meta["caption"],
        "question": qa["query"],
        "answer": qa["answer"],
        "image": image_name,
    }

    (output_dir / "sample.json").write_text(json.dumps(record, indent=2) + "\n")
    (output_dir / "sample.jsonl").write_text(json.dumps(record) + "\n")
    (output_dir / "prompt.txt").write_text(qa["query"] + "\n")
    (output_dir / "answer.txt").write_text(str(qa["answer"]) + "\n")

    print(json.dumps({"workspace": str(output_dir), "image": str(image_out), "question": qa["query"], "answer": qa["answer"]}, indent=2))


if __name__ == "__main__":
    main()
