from __future__ import annotations

import io
import itertools
import re
import unittest
from pathlib import Path


import rlab.metric_names as metric_names
from rlab.training.sb3_helpers import (
    Sb3HumanOutputFormatHelper,
    disable_sb3_human_output_truncation,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class Sb3LoggerTests(unittest.TestCase):
    def test_human_output_truncation_is_disabled_for_long_level_complete_metrics(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        key_values = {
            "train/outcome/success/from/Level1-2_bonus_room_checkpoint/count": 1,
            "train/outcome/success/from/Level1-2_bonus_room_checkpoint/rate/current": 0.0,
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

        callback = Sb3HumanOutputFormatHelper(max_length=256)
        callback.model = FakeModel()
        callback._on_training_start()

        self.assertEqual(output_format.max_length, 256)


class MetricsDocumentationTests(unittest.TestCase):
    def test_metrics_reference_registry_section_is_generated_from_code(self) -> None:
        metrics_doc = Path(__file__).resolve().parents[1] / "METRICS.md"
        content = metrics_doc.read_text(encoding="utf-8")
        generated = content.split("<!-- METRIC_REGISTRY_START -->\n", 1)[1].split(
            "\n<!-- METRIC_REGISTRY_END -->", 1
        )[0]
        self.assertEqual(generated, metric_names.render_metric_registry_markdown())

    def test_registry_rejects_unknown_metrics_and_unsafe_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("train/mystery/value")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("eval/full/episode/return/typo")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("leader/checkpoint/typo")
        with self.assertRaisesRegex(ValueError, "metric dimension"):
            metric_names.train_success_count_metric("unsafe start")

    def test_logger_boundary_rejects_misspelled_rlab_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "logger boundary"):
            metric_names.canonical_training_scalars({"train/outcome/succes/current/rate/min": 0.5})

    def test_cardinality_has_no_start_by_reason_scalar_product(self) -> None:
        starts = [f"Start-{index}" for index in range(32)]
        reasons = [f"reason-{index}" for index in range(5)]
        names = {metric_names.train_success_count_metric(start) for start in starts} | {
            metric_names.train_outcome_reason_count_metric(reason) for reason in reasons
        }
        self.assertEqual(len(names), len(starts) + len(reasons))
        self.assertFalse(any("/reason/" in name and "/from/" in name for name in names))

    def test_eval_outcome_cardinality_stays_bounded_across_all_protocols(self) -> None:
        starts = [f"Start-{index}" for index in range(32)]
        reasons = [f"reason-{index}" for index in range(5)]
        names = set()
        for protocol in metric_names.EVAL_PROTOCOLS:
            names.update(
                metric_names.eval_success_from_rate_metric(protocol, start) for start in starts
            )
            names.update(
                metric_names.eval_reason_count_metric(protocol, reason) for reason in reasons
            )
            names.update(
                metric_names.eval_reason_rate_metric(protocol, reason) for reason in reasons
            )
            names.update(
                {
                    metric_names.eval_success_rate_metric(protocol, "min"),
                    metric_names.eval_success_rate_metric(protocol, "mean"),
                }
            )

        self.assertEqual(len(names), 132)
        self.assertLessEqual(len(names), 150)
        self.assertFalse(any("/reason/" in name and "/from/" in name for name in names))

    def test_schema_v2_cardinality_margins_and_single_start_lifecycle(self) -> None:
        protocols = list(metric_names.EVAL_PROTOCOLS)
        starts = ["Start"]
        reasons = [f"reason-{index}" for index in range(5)]
        values = {
            "protocol": protocols,
            "reason": reasons,
            "start": starts,
            "component": ["progress"],
            "signal": ["progress"],
            "progress": ["x"],
        }
        scalar_names: set[str] = set()
        for definition in metric_names.METRIC_DEFINITIONS:
            if definition.unit in {"histogram", "table"} or definition.storage == "summary":
                continue
            placeholders = re.findall(r"\{([^}]+)\}", definition.name)
            for replacements in itertools.product(*(values[name] for name in placeholders)):
                name = definition.name
                for placeholder, replacement in zip(placeholders, replacements, strict=True):
                    name = name.replace(f"{{{placeholder}}}", replacement, 1)
                scalar_names.add(name)

        self.assertLessEqual(len(scalar_names), 150)
        self.assertEqual(
            len(
                {
                    metric_names.train_success_count_metric("A"),
                    metric_names.train_success_attempts_metric("A"),
                    metric_names.train_success_current_rate_metric("A"),
                    metric_names.train_success_window_rate_metric("A"),
                }
            ),
            4,
        )
        self.assertEqual(
            len(
                {
                    metric_names.train_reward_component_metric("active", stat)
                    for stat in ("mean", "nonzero_rate", "share")
                }
            ),
            3,
        )
