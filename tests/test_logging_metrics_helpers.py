from __future__ import annotations

import io
import re
import unittest
from pathlib import Path


import rlab.metric_names as metric_names
from rlab.train import (
    Sb3HumanOutputFormatCallback,
    disable_sb3_human_output_truncation,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)

class Sb3LoggerTests(unittest.TestCase):
    def test_human_output_truncation_is_disabled_for_long_level_complete_metrics(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        key_values = {
            "train/info/level_complete/from/Level1-2_bonus_room_checkpoint/count": 1,
            "train/info/level_complete/from/Level1-2_bonus_room_checkpoint/rate": 0.0,
        }
        key_excluded = {key: () for key in key_values}

        with self.assertRaisesRegex(ValueError, "truncated"):
            HumanOutputFormat(io.StringIO()).write(key_values, key_excluded)

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            logger = FakeLogger()

        disable_sb3_human_output_truncation(FakeModel())

        output_format.write(key_values, key_excluded)
        self.assertEqual(output_format.max_length, 512)

    def test_uninitialized_sb3_logger_is_ignored(self) -> None:
        class FakeSb3Model:
            @property
            def logger(self):
                raise AttributeError("'FakeSb3Model' object has no attribute '_logger'")

        disable_sb3_human_output_truncation(FakeSb3Model())

    def test_callback_updates_logger_after_training_starts(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            _logger = FakeLogger()

        callback = Sb3HumanOutputFormatCallback(max_length=256)
        callback.model = FakeModel()
        callback._on_training_start()

        self.assertEqual(output_format.max_length, 256)

class MetricsDocumentationTests(unittest.TestCase):
    def test_metrics_reference_mentions_metric_name_constants_and_core_templates(self) -> None:
        metrics_doc = Path(__file__).resolve().parents[1] / "METRICS.md"
        content = metrics_doc.read_text(encoding="utf-8")

        constant_values = sorted(
            value
            for name, value in vars(metric_names).items()
            if name.isupper() and isinstance(value, str)
        )
        missing_constants = [value for value in constant_values if value not in content]
        self.assertEqual(missing_constants, [])

        required_templates = [
            "train/done/<reason>/from/<prev>",
            "train/done/<reason>/from/<prev>/ep_window/rate",
            "train/done/<reason>/from_rate/min",
            "train/done/<reason>/from_rate/mean",
            "train/info/level_complete/from/<prev>/count",
            "train/info/level_complete/from/<prev>/attempts",
            "train/info/level_complete/from/<prev>/rate/current",
            "train/info/level_complete/from/<prev>/rate",
            "train/info/level_complete/rate/min/current",
            "train/info/level_complete/rate/mean/current",
            "train/info/level_complete/rate/min/last",
            "train/info/level_complete/rate/mean/last",
            "train/reward/<component>/<stat>",
            "train/reward_share/<component>",
            "eval/done/<reason>/from/<start>",
            "eval/info/level_complete/rate/min",
            "eval/info/level_complete/rate/mean",
            "eval/info/level_complete/rate/min/last",
            "eval/info/level_complete/rate/mean/last",
        ]
        missing_templates = [template for template in required_templates if template not in content]
        self.assertEqual(missing_templates, [])
