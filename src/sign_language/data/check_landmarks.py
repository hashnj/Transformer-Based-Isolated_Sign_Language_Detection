"""Validate extracted landmark arrays and summarize their contents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sign_language.data.landmarks import (
    COORDINATE_COUNT,
    DEFAULT_OUTPUT_MANIFEST,
    LANDMARK_COUNT,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check landmark extraction outputs listed in the output manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Expected sequence length; if omitted, infer it from the first valid file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.manifest.is_file():
        raise SystemExit(
            f"Landmark manifest not found: {args.manifest}. "
            "Run the extractor first."
        )
    if args.sequence_length is not None and args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be greater than zero.")

    expected_length = args.sequence_length
    counts = {"processed": 0, "already_exists": 0, "failed": 0}
    total_frames = 0
    total_detected_landmarks = 0
    problems: list[str] = []

    with args.manifest.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_fields = {"file_name", "feature_path", "status", "error"}
        missing_fields = required_fields.difference(reader.fieldnames or ())
        if missing_fields:
            raise SystemExit(
                "Landmark manifest is missing columns: "
                + ", ".join(sorted(missing_fields))
            )

        rows = list(reader)
    if not rows:
        raise SystemExit(f"Landmark manifest is empty: {args.manifest}")

    for row in rows:
        status = row["status"]
        if status not in counts:
            problems.append(f"{row['file_name']}: unknown status {status!r}")
            continue
        counts[status] += 1
        if status == "failed":
            problems.append(f"{row['file_name']}: {row['error']}")
            continue

        feature_path = Path(row["feature_path"])
        if not feature_path.is_file():
            problems.append(f"{row['file_name']}: feature file not found: {feature_path}")
            continue

        try:
            with np.load(feature_path, allow_pickle=False) as feature:
                keypoints = feature["keypoints"]
                landmark_mask = feature["landmark_mask"]
                frame_mask = feature["frame_mask"]
                gloss = str(feature["gloss"].item())
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{row['file_name']}: could not read feature file: {exc}")
            continue

        if keypoints.ndim != 3 or keypoints.shape[1:] != (
            LANDMARK_COUNT,
            COORDINATE_COUNT,
        ):
            problems.append(
                f"{row['file_name']}: unexpected keypoints shape {keypoints.shape}"
            )
            continue
        if expected_length is None:
            expected_length = int(keypoints.shape[0])
        if keypoints.shape[0] != expected_length:
            problems.append(
                f"{row['file_name']}: expected {expected_length} frames, "
                f"got {keypoints.shape[0]}"
            )
            continue
        if landmark_mask.shape != (expected_length, LANDMARK_COUNT):
            problems.append(
                f"{row['file_name']}: unexpected landmark mask shape "
                f"{landmark_mask.shape}"
            )
            continue
        if frame_mask.shape != (expected_length,):
            problems.append(
                f"{row['file_name']}: unexpected frame mask shape {frame_mask.shape}"
            )
            continue
        if not np.isfinite(keypoints).all():
            problems.append(f"{row['file_name']}: keypoints contain NaN or infinity")
            continue
        if not np.isin(landmark_mask, [False, True]).all():
            problems.append(f"{row['file_name']}: landmark mask is not boolean-valued")
            continue
        if not np.isin(frame_mask, [False, True]).all():
            problems.append(f"{row['file_name']}: frame mask is not boolean-valued")
            continue
        if frame_mask.sum() == 0:
            problems.append(f"{row['file_name']}: no decoded frames were marked valid")
            continue

        valid_landmarks = int(landmark_mask.sum())
        total_frames += int(frame_mask.sum())
        total_detected_landmarks += valid_landmarks
        print(
            f"OK  {row['file_name']}: gloss={gloss!r}, "
            f"keypoints={keypoints.shape}, detected_landmarks={valid_landmarks}"
        )

    print(
        f"\nRows: {len(rows)} | processed: {counts['processed']} | "
        f"already present: {counts['already_exists']} | failed: {counts['failed']}"
    )
    print(
        f"Valid frames: {total_frames} | detected landmarks: "
        f"{total_detected_landmarks}"
    )
    if total_detected_landmarks == 0:
        problems.append(
            "no landmarks were detected in any feature file; check that extraction "
            "ran with MediaPipe and that the videos contain visible people"
        )
    if problems:
        print(f"\nValidation issues ({len(problems)}):")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("All listed feature files passed validation.")


if __name__ == "__main__":
    main()
