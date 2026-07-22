from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pytest

from rlab.dataset_contract import STORAGE_FORMAT_LOSSLESS_VIDEO
from rlab.dataset_media import LosslessVideoWriter, iter_episode_frames


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_lossless_video_round_trip_streams_exact_rgb(tmp_path):
    output = tmp_path / "videos" / "episode.rgb.mkv.bin"
    frames = [
        np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3),
        np.full((4, 5, 3), 193, dtype=np.uint8),
    ]
    writer = LosslessVideoWriter(output, fps=30)
    try:
        for frame in frames:
            writer.write(frame)
        writer.close()
    except Exception:
        writer.abort()
        raise

    rows = [
        {
            "storage_format": STORAGE_FORMAT_LOSSLESS_VIDEO,
            "video_path": "videos/episode.rgb.mkv.bin",
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "frame_width": 5,
            "frame_height": 4,
            "step_index": index,
        }
        for index, frame in enumerate(frames)
    ]
    decoded = list(iter_episode_frames(rows, root=tmp_path))
    assert len(decoded) == len(frames)
    for actual, expected in zip(decoded, frames, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_video_media_rejects_parent_traversal(tmp_path):
    rows = [
        {
            "storage_format": STORAGE_FORMAT_LOSSLESS_VIDEO,
            "video_path": "../outside.rgb.mkv.bin",
            "frame_sha256": "0" * 64,
            "frame_width": 5,
            "frame_height": 4,
            "step_index": 0,
        }
    ]
    with pytest.raises(ValueError, match="unsafe dataset media path"):
        list(iter_episode_frames(rows, root=tmp_path))
