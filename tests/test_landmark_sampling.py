import numpy as np
import pytest

from sign_language.data.landmarks import (
    VideoProcessingError,
    _frame_indices,
    _read_sampled_frames,
)


class FakeCapture:
    def __init__(self, frames, fail_at=None):
        self.frames = frames
        self.fail_at = fail_at
        self.position = 0

    def read(self):
        if self.position == self.fail_at or self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame


def test_frame_indices_sample_uniformly_and_pad_short_sequences():
    assert _frame_indices(100, 4).tolist() == [0, 33, 66, 99]
    assert _frame_indices(2, 4).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("frame_count", "sequence_length"),
    [(0, 4), (4, 0), (4, -1)],
)
def test_frame_indices_reject_invalid_dimensions(frame_count, sequence_length):
    with pytest.raises((ValueError, RuntimeError)):
        _frame_indices(frame_count, sequence_length)


def test_read_sampled_frames_reuses_last_frame_for_one_missing_terminal_frame():
    frames = [np.asarray(index) for index in range(71)]
    selected_indices = _frame_indices(72, 4)

    result = _read_sampled_frames(
        FakeCapture(frames),
        selected_indices,
        frame_count=72,
        video_path="video.mp4",
    )

    assert result == [0, 23, 47, 70]
    assert result[-1] == frames[-1]


@pytest.mark.parametrize(
    ("fail_at", "frame_count", "selected_indices"),
    [
        (2, 5, [0, 2, 4]),
        (3, 6, [0, 3, 5]),
        (4, 6, [0, 2, 5]),
    ],
)
def test_read_sampled_frames_reports_nonterminal_or_multiframe_failure(
    fail_at, frame_count, selected_indices
):
    with pytest.raises(VideoProcessingError, match=f"frame {fail_at}"):
        _read_sampled_frames(
            FakeCapture(list(range(fail_at)), fail_at=fail_at),
            selected_indices,
            frame_count,
            "video.mp4",
        )
