import csv
import json

import pytest

from sign_language.data.manifest import read_video_manifest


def test_read_video_manifest_extracts_gloss_and_resolves_path(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    row = {
        "sample_id": "sample-1",
        "file_name": "clip.mp4",
        "local_path": r"data\raw\clip.mp4",
        "metadata_json": json.dumps({"gloss": {"label": "hello"}}),
    }
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    samples = read_video_manifest(manifest_path, tmp_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample-1"
    assert samples[0].gloss == "hello"
    assert samples[0].video_path == tmp_path / "data" / "raw" / "clip.mp4"


def test_read_video_manifest_rejects_missing_gloss(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    row = {
        "sample_id": "sample-1",
        "file_name": "clip.mp4",
        "local_path": "clip.mp4",
        "metadata_json": json.dumps({"tags": []}),
    }
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="gloss"):
        read_video_manifest(manifest_path, tmp_path)
