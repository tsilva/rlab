from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rlab.video import (
    PolicyObservationPreview,
    policy_observation_mosaic,
    write_preview_video,
)


class PolicyObservationPreviewTests(unittest.TestCase):
    def test_channel_first_stack_uses_newest_frame_and_two_lane_layout(self) -> None:
        observation = np.zeros((2, 4, 3, 4), dtype=np.uint8)
        observation[0, -1] = 30
        observation[1, -1] = 80

        converted = policy_observation_mosaic(observation)

        self.assertIsNotNone(converted)
        frame, lane_count = converted or (None, 0)
        self.assertEqual(lane_count, 2)
        self.assertEqual(frame.shape, (3, 8))
        np.testing.assert_array_equal(frame[:, :4], 30)
        np.testing.assert_array_equal(frame[:, 4:], 80)

    def test_channel_last_dict_stack_and_same_step_resets_are_captured(self) -> None:
        capture = PolicyObservationPreview(max_frames=2, max_lanes=2)
        first = {"image": np.zeros((2, 5, 6, 4), dtype=np.uint8)}
        reset = {"image": np.zeros((2, 5, 6, 4), dtype=np.uint8)}
        first["image"][..., -1] = 25
        reset["image"][0, ..., -1] = 0
        reset["image"][1, ..., -1] = 90

        self.assertTrue(capture.capture(first))
        self.assertTrue(capture.capture(reset))
        self.assertFalse(capture.capture(first))

        self.assertEqual(len(capture.frames), 2)
        np.testing.assert_array_equal(capture.frames[1][:, :6], 0)
        np.testing.assert_array_equal(capture.frames[1][:, 6:], 90)

    def test_non_image_observation_disables_capture_without_raising(self) -> None:
        capture = PolicyObservationPreview()

        self.assertFalse(capture.capture(np.zeros((2, 8), dtype=np.float32)))
        self.assertEqual(capture.frames, [])
        self.assertIn("not a supported image", capture.error or "")

    def test_encoder_contract_is_bounded_and_reports_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preview.mp4"

            def fake_run(command, **kwargs):
                Path(command[-1]).write_bytes(b"mp4")
                self.assertEqual(kwargs["timeout"], 2.0)
                self.assertIn("250k", command)
                self.assertIn("yuv420p", command)
                return subprocess.CompletedProcess(command, 0, b"", b"")

            with (
                mock.patch("rlab.video.shutil.which", return_value="/usr/bin/ffmpeg"),
                mock.patch("rlab.video.subprocess.run", side_effect=fake_run),
            ):
                metadata = write_preview_video(
                    [np.zeros((84, 168), dtype=np.uint8) for _ in range(15)],
                    output,
                    fps=15,
                    scale=2,
                    timeout_seconds=2,
                    max_bytes=2 * 1024 * 1024,
                )

        self.assertEqual(metadata["width"], 336)
        self.assertEqual(metadata["height"], 168)
        self.assertEqual(metadata["duration_seconds"], 1.0)
        self.assertEqual(metadata["size_bytes"], 3)


if __name__ == "__main__":
    unittest.main()
