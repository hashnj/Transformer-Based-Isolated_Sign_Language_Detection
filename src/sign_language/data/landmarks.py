"""Extract fixed-length MediaPipe Holistic landmarks from local video clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sign_language.data.manifest import DEFAULT_MANIFEST, PROJECT_ROOT, VideoSample, read_video_manifest

LANDMARK_COUNT = 33 + 21 + 21
COORDINATE_COUNT = 3
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "manifests" / "landmark_manifest.csv"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "landmarks.json"


class VideoProcessingError(RuntimeError):
    """An expected video decoding or frame-processing failure."""


@dataclass(frozen=True)
class LandmarkSequence:
    keypoints: np.ndarray
    landmark_mask: np.ndarray
    frame_mask: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose and hand landmarks from the local video manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N manifest rows (useful for a smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute feature files that already exist.",
    )
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2))
    parser.add_argument("--min-detection-confidence", type=float)
    parser.add_argument("--min-tracking-confidence", type=float)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read landmark config: {args.config}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in landmark config: {args.config}") from exc
    if not isinstance(config, dict):
        raise SystemExit("Landmark config must contain a JSON object.")

    args.sequence_length = (
        args.sequence_length
        if args.sequence_length is not None
        else config.get("sequence_length", 32)
    )
    args.model_complexity = (
        args.model_complexity
        if args.model_complexity is not None
        else config.get("model_complexity", 1)
    )
    args.min_detection_confidence = (
        args.min_detection_confidence
        if args.min_detection_confidence is not None
        else config.get("min_detection_confidence", 0.5)
    )
    args.min_tracking_confidence = (
        args.min_tracking_confidence
        if args.min_tracking_confidence is not None
        else config.get("min_tracking_confidence", 0.5)
    )
    return args


def _frame_indices(frame_count: int, sequence_length: int) -> np.ndarray:
    if frame_count <= 0:
        raise VideoProcessingError("Video reports no frames")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    sample_count = min(frame_count, sequence_length)
    return np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int64)


def _store_landmarks(
    target: np.ndarray,
    mask: np.ndarray,
    offset: int,
    landmarks: object | None,
    count: int,
) -> None:
    if landmarks is None:
        return
    points = getattr(landmarks, "landmark")
    for index, point in enumerate(points[:count]):
        target[offset + index] = (point.x, point.y, point.z)
        mask[offset + index] = True


def _read_sampled_frames(
    capture: object,
    frame_indices: np.ndarray,
    frame_count: int,
    video_path: Path,
) -> list[np.ndarray]:
    """Read selected frames sequentially, tolerating one missing terminal frame."""
    frames: list[np.ndarray] = []
    next_frame_index = 0
    selected_index = 0
    last_frame: np.ndarray | None = None

    while selected_index < len(frame_indices):
        target_index = int(frame_indices[selected_index])
        while next_frame_index <= target_index:
            success, frame = capture.read()
            if not success:
                is_single_missing_terminal_frame = (
                    selected_index == len(frame_indices) - 1
                    and target_index == frame_count - 1
                    and next_frame_index == frame_count - 1
                    and last_frame is not None
                )
                if is_single_missing_terminal_frame:
                    frames.append(last_frame.copy())
                    selected_index += 1
                    break
                raise VideoProcessingError(
                    f"Could not decode frame {next_frame_index} from {video_path}"
                )

            last_frame = frame
            if next_frame_index == target_index:
                frames.append(frame)
                selected_index += 1
            next_frame_index += 1

    return frames


def extract_video(
    video_path: Path,
    holistic: object,
    sequence_length: int,
) -> LandmarkSequence:
    """Decode a video at uniform temporal intervals and extract Holistic points."""
    import cv2

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"OpenCV could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = _frame_indices(frame_count, sequence_length)
        keypoints = np.zeros(
            (sequence_length, LANDMARK_COUNT, COORDINATE_COUNT), dtype=np.float32
        )
        landmark_mask = np.zeros((sequence_length, LANDMARK_COUNT), dtype=np.bool_)
        frame_mask = np.zeros(sequence_length, dtype=np.bool_)
        sampled_frames = _read_sampled_frames(
            capture, frame_indices, frame_count, video_path
        )

        for output_index, bgr_frame in enumerate(sampled_frames):
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb_frame)
            frame_mask[output_index] = True
            _store_landmarks(
                keypoints[output_index],
                landmark_mask[output_index],
                0,
                result.pose_landmarks,
                33,
            )
            _store_landmarks(
                keypoints[output_index],
                landmark_mask[output_index],
                33,
                result.left_hand_landmarks,
                21,
            )
            _store_landmarks(
                keypoints[output_index],
                landmark_mask[output_index],
                54,
                result.right_hand_landmarks,
                21,
            )

        return LandmarkSequence(keypoints, landmark_mask, frame_mask)
    finally:
        capture.release()


def _feature_path(output_dir: Path, sample: VideoSample) -> Path:
    return output_dir / f"{Path(sample.file_name).stem}.npz"


def _save_feature(
    feature_path: Path,
    sample: VideoSample,
    sequence: LandmarkSequence,
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        keypoints=sequence.keypoints,
        landmark_mask=sequence.landmark_mask,
        frame_mask=sequence.frame_mask,
        sample_id=np.asarray(sample.sample_id),
        gloss=np.asarray(sample.gloss),
        file_name=np.asarray(sample.file_name),
    )


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "file_name",
        "gloss",
        "source_path",
        "feature_path",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_extraction(args: argparse.Namespace) -> int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
            version = ".".join(map(str, sys.version_info[:3]))
            raise SystemExit(
                f"This project requires Python 3.11 or 3.12 for its pinned "
                f"MediaPipe dependency; the active interpreter is Python {version}. "
                "Create/select a compatible project environment, then install "
                "requirements-dev.txt in that environment."
            ) from exc
        raise SystemExit(
            "OpenCV or MediaPipe is missing from the active Python environment. "
            "Install the project dependencies in this same environment with "
            "`python -m pip install -r requirements-dev.txt`."
        ) from exc
    print(f"Using OpenCV {cv2.__version__} and MediaPipe {mp.__version__}")

    if not isinstance(args.sequence_length, int) or args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be greater than zero.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")
    if not isinstance(args.min_detection_confidence, (int, float)) or not (
        0.0 <= args.min_detection_confidence <= 1.0
    ):
        raise SystemExit("--min-detection-confidence must be between 0 and 1.")
    if not isinstance(args.min_tracking_confidence, (int, float)) or not (
        0.0 <= args.min_tracking_confidence <= 1.0
    ):
        raise SystemExit("--min-tracking-confidence must be between 0 and 1.")
    if args.model_complexity not in (0, 1, 2):
        raise SystemExit("--model-complexity must be 0, 1, or 2.")

    samples = read_video_manifest(args.manifest, PROJECT_ROOT)
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures = 0
    for index, sample in enumerate(samples, start=1):
        feature_path = _feature_path(args.output_dir, sample)
        row = {
            "sample_id": sample.sample_id,
            "file_name": sample.file_name,
            "gloss": sample.gloss,
            "source_path": str(sample.video_path),
            "feature_path": str(feature_path),
            "status": "",
            "error": "",
        }
        if feature_path.exists() and not args.overwrite:
            row["status"] = "already_exists"
        else:
            try:
                with mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=args.model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=args.min_detection_confidence,
                    min_tracking_confidence=args.min_tracking_confidence,
                ) as holistic:
                    sequence = extract_video(
                        sample.video_path, holistic, args.sequence_length
                    )
                _save_feature(feature_path, sample, sequence)
                row["status"] = "processed"
            except (FileNotFoundError, VideoProcessingError) as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                failures += 1
        rows.append(row)
        print(f"[{index}/{len(samples)}] {sample.file_name}: {row['status']}")

    _write_manifest(args.output_manifest, rows)
    print(f"Feature manifest: {args.output_manifest}")
    print(f"Feature arrays: {args.output_dir}")
    if failures:
        print(f"Landmark extraction failed for {failures} sample(s); see the manifest.")
        return 1
    return 0


def main() -> None:
    raise SystemExit(run_extraction(_parse_args()))
