"""Read the sample manifest created by the Hugging Face downloader."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "huggingface_wlasl_manifest.csv"
)


@dataclass(frozen=True)
class VideoSample:
    sample_id: str
    file_name: str
    video_path: Path
    gloss: str


def _metadata_label(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field)
    if isinstance(value, dict):
        value = value.get("label")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest metadata has no usable {field!r} label")
    return value.strip()


def read_video_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> list[VideoSample]:
    """Load video paths and gloss labels from the downloader's CSV manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    samples: list[VideoSample] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_fields = {"sample_id", "file_name", "local_path", "metadata_json"}
        missing_fields = required_fields.difference(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Manifest is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                metadata = json.loads(row["metadata_json"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid metadata JSON at {manifest_path}:{row_number}"
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Expected metadata object at {manifest_path}:{row_number}"
                )

            local_path = Path(row["local_path"].replace("\\", "/"))
            video_path = (
                local_path if local_path.is_absolute() else project_root / local_path
            )
            samples.append(
                VideoSample(
                    sample_id=row["sample_id"],
                    file_name=row["file_name"],
                    video_path=video_path,
                    gloss=_metadata_label(metadata, "gloss"),
                )
            )

    if not samples:
        raise ValueError(f"Dataset manifest contains no samples: {manifest_path}")
    return samples
