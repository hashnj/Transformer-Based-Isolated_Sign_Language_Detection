"""Download WLASL samples from the Voxel51 Hugging Face dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from sign_language.data.manifest import PROJECT_ROOT

DATASET_ID = "Voxel51/WLASL"
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "huggingface_wlasl" / "videos"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "huggingface_wlasl_manifest.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WLASL videos and metadata from Hugging Face."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of samples to load for inspection (default: 100).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load the full Hub dataset instead of a limited sample.",
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset ID.")
    return parser.parse_args()


def _metadata_json(sample: Any) -> str:
    metadata = sample.to_dict()
    metadata.pop("_id", None)
    metadata.pop("filepath", None)
    return json.dumps(metadata, ensure_ascii=False, default=str)


def main() -> None:
    args = _parse_args()
    if not args.all and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero.")

    try:
        import fiftyone.utils.huggingface as fouh
    except ImportError as exc:
        raise SystemExit(
            "FiftyOne is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    max_samples = None if args.all else args.max_samples
    print(
        f"Loading {args.dataset}: "
        f"{'all samples' if max_samples is None else f'{max_samples} samples'}"
    )
    dataset = fouh.load_from_hub(args.dataset, max_samples=max_samples)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as manifest_file:
        fields = ["sample_id", "file_name", "local_path", "metadata_json"]
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()

        count = 0
        for sample in dataset.iter_samples(progress=True):
            source_path = Path(sample.filepath)
            if not source_path.is_file():
                raise FileNotFoundError(f"Downloaded media is missing: {source_path}")

            destination = VIDEO_DIR / source_path.name
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            writer.writerow(
                {
                    "sample_id": str(sample.id),
                    "file_name": destination.name,
                    "local_path": str(destination.relative_to(PROJECT_ROOT)),
                    "metadata_json": _metadata_json(sample),
                }
            )
            count += 1

    print(f"Saved {count} video files to {VIDEO_DIR}")
    print(f"Saved sample metadata to {MANIFEST_PATH}")
    print("Check labels and splits before using this Hub copy as a benchmark.")
