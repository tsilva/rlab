from __future__ import annotations

import argparse
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rlab.callbacks import (
    CallbackHelper,
    LedgerCheckpointHelper,
    MetricStoreLoggerHelper,
    MetricStoreOutputFormat,
    MetricThresholdStopHelper,
    RlabCallback,
    RolloutDiagnosticsHelper,
    RuntimeMetricsHelper,
    ThroughputHelper,
    task_metric_source,
)
from rlab.env import EnvConfig
from rlab.metric_store import MetricStore
from rlab.metric_names import EVAL_FULL_SUCCESS_RATE_MIN, TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TaskMetricSourceTests(unittest.TestCase):
    def test_start_names_are_shared_across_training_and_evaluation(self) -> None:
        self.assertEqual(task_metric_source("Level1-1"), "Level1-1")
        self.assertEqual(task_metric_source("Level2-3"), "Level2-3")
        self.assertEqual(task_metric_source("custom-start"), "custom-start")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class RuntimeMetricsHelperTests(unittest.TestCase):
    def test_consumes_episode_records_without_info_payloads(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(event_names=("life_loss", "stalled"))
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]
        callback.num_timesteps = 20

        self.assertTrue(
            callback._on_records(
                [
                    SimpleNamespace(
                        episode_return=1.0,
                        events=("life_loss",),
                        start_id="Level1-1",
                        truncated=False,
                    ),
                    SimpleNamespace(
                        episode_return=2.0,
                        events=("stalled",),
                        start_id="Level1-1",
                        truncated=True,
                    ),
                ]
            )
        )
        self.assertEqual(logger.records, {})
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/outcome/terminal/count"], 2)
        self.assertEqual(logger.records["train/outcome/reason/life_loss/count"], 1)
        self.assertEqual(logger.records["train/outcome/reason/stalled/count"], 1)
        self.assertEqual(logger.records["train/outcome/reason/max_steps/count"], 1)
        self.assertNotIn("train/outcome/reason/unclassified/count", logger.records)
        self.assertFalse(any("/success/" in key for key in logger.records))

    def test_done_reason_rate_is_pooled_without_start_cross_product(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(event_names=("life_loss", "stalled"))
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]

        records = [
            SimpleNamespace(
                episode_return=0.0,
                events=(("life_loss",) if index < 50 else ("stalled",)),
                start_id="Level1-1",
                truncated=False,
            )
            for index in range(100)
        ]
        callback._on_records(records)
        callback._on_rollout_end()

        self.assertEqual(
            logger.records["train/outcome/reason/life_loss/rate/window_100"],
            0.5,
        )
        self.assertEqual(
            logger.records["train/outcome/reason/stalled/rate/window_100"],
            0.5,
        )


class RlabCallbackTests(unittest.TestCase):
    def test_drains_episode_records_once_for_done_and_completion_metrics(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class RecordEnv:
            def __init__(self) -> None:
                self.drain_calls = 0

            def drain_records(self):
                self.drain_calls += 1
                return [
                    SimpleNamespace(
                        lane=0,
                        events=("level_change",),
                        start_id="Level1-1",
                        transitions={"level_change": ((0, 0), (0, 1))},
                        metrics={},
                    ),
                    SimpleNamespace(
                        lane=0,
                        events=("level_change",),
                        start_id="Level1-1",
                        truncated=False,
                        outcome=SimpleNamespace(name="SUCCESS"),
                        episode_return=1.0,
                        metrics={},
                    ),
                ]

        class Wrapper:
            def __init__(self, env) -> None:
                self.venv = env

        class Model:
            def __init__(self, env) -> None:
                self.env = Wrapper(env)
                self.logger = Logger()

        env = RecordEnv()
        model = Model(env)
        callback = RlabCallback(
            [
                RuntimeMetricsHelper(
                    event_names=("level_change",),
                    configured_starts=("Level1-1",),
                    track_success=True,
                ),
            ]
        )
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 32
        callback.locals = {}

        self.assertTrue(callback._on_step())
        callback._on_rollout_end()

        self.assertEqual(env.drain_calls, 1)
        self.assertEqual(model.logger.records["train/outcome/terminal/count"], 1)
        self.assertEqual(model.logger.records["train/outcome/reason/level_change/count"], 0)
        self.assertNotIn("train/outcome/reason/life_loss/count", model.logger.records)
        self.assertEqual(
            model.logger.records["train/outcome/success/from/Level1-1/count"],
            1,
        )

    def test_precomputed_step_dispatch_preserves_component_order(self) -> None:
        calls: list[str] = []

        class StepHelper(CallbackHelper):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def _on_step(self) -> bool:
                calls.append(self.name)
                return True

        class RecordHelper(CallbackHelper):
            def _on_records(self, records) -> bool:
                calls.append(f"records:{len(records)}")
                return True

        env = SimpleNamespace(drain_records=lambda: [object()])
        model = SimpleNamespace(env=env, logger=SimpleNamespace())
        callback = RlabCallback([StepHelper("first"), RecordHelper(), StepHelper("last")])
        callback.model = model  # type: ignore[assignment]
        callback.locals = {}

        self.assertTrue(callback._on_step())
        self.assertEqual(calls, ["first", "records:1", "last"])


class RuntimeMetricsCompletionTests(unittest.TestCase):
    def test_tracks_generic_episode_outcomes_by_readable_start(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(
            event_names=("goal_reached",),
            configured_starts=("StartA", "StartB"),
            track_success=True,
        )
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]

        callback._on_records(
            [
                SimpleNamespace(
                    start_id="StartA",
                    events=("goal_reached",),
                    episode_return=1.0,
                    outcome="success",
                    truncated=False,
                ),
                SimpleNamespace(
                    start_id="StartB",
                    events=("failure",),
                    episode_return=0.0,
                    outcome="failure",
                    truncated=False,
                ),
            ]
        )
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/outcome/success/from/StartA/count"], 1)
        self.assertEqual(logger.records["train/outcome/success/from/StartB/count"], 0)
        self.assertEqual(logger.records["train/outcome/success/start_coverage/rate"], 1.0)

    def test_success_window_requires_every_configured_start_and_attempts_are_cumulative(
        self,
    ) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(
            event_names=("goal_reached",),
            configured_starts=("StartA", "StartB"),
            track_success=True,
        )
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]
        callback._on_records(
            [
                SimpleNamespace(
                    start_id="StartA",
                    events=("goal_reached",),
                    episode_return=1.0,
                    outcome="success",
                    truncated=False,
                )
                for _ in range(150)
            ]
        )
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/outcome/success/from/StartA/attempts"], 150)
        self.assertNotIn("train/outcome/success/window_100/rate/min", logger.records)

        callback._on_records(
            [
                SimpleNamespace(
                    start_id="StartB",
                    events=(),
                    episode_return=0.0,
                    outcome="failure",
                    truncated=False,
                )
                for _ in range(100)
            ]
        )
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/outcome/success/window_100/rate/min"], 0.0)
        self.assertEqual(logger.records["train/outcome/success/window_100/rate/mean"], 0.5)


class MetricThresholdStopHelperTests(unittest.TestCase):
    class FakeLogger:
        def __init__(self) -> None:
            self.records: dict[str, int | float] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.logger = MetricThresholdStopHelperTests.FakeLogger()

    def make_callback(self, marker_path: Path) -> tuple[MetricThresholdStopHelper, FakeModel]:
        model = self.FakeModel()
        callback = MetricThresholdStopHelper(
            detector=[
                {
                    "metric": TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
                    "operator": ">",
                    "threshold": 0.99,
                }
            ],
            marker_path=marker_path,
        )
        callback.model = model  # type: ignore[assignment]
        return callback, model

    def test_waits_until_metric_crosses_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker_path = Path(tmp) / "run" / "early_stop.txt"
            callback, model = self.make_callback(marker_path)
            callback.num_timesteps = 100

            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN] = 0.99
            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN] = 1.0
            callback.num_timesteps = 200
            self.assertFalse(callback._on_step())

            marker = marker_path.read_text(encoding="utf-8")
            self.assertIn("early_stop=metric_threshold", marker)
            self.assertIn(f"early_stop_metric={TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN}", marker)
            self.assertIn("early_stop_operator=>", marker)
            self.assertIn("early_stop_threshold=0.99", marker)
            self.assertIn("early_stop_value=1", marker)
            self.assertIn("timesteps=200", marker)

    def test_structured_detector_requires_all_metric_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker_path = Path(tmp) / "run" / "early_stop.txt"
            model = self.FakeModel()
            callback = MetricThresholdStopHelper(
                detector=[
                    {
                        "metric": TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
                        "operator": ">",
                        "threshold": 0.99,
                    },
                    {
                        "metric": "train/episode/return/shaped/mean",
                        "operator": ">=",
                        "threshold": 1000,
                    },
                ],
                marker_path=marker_path,
            )
            callback.model = model  # type: ignore[assignment]
            callback.num_timesteps = 100

            model.logger.records[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN] = 1.0
            model.logger.records["train/episode/return/shaped/mean"] = 999
            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records["train/episode/return/shaped/mean"] = 1000
            callback.num_timesteps = 200
            self.assertFalse(callback._on_step())

            marker = marker_path.read_text(encoding="utf-8")
            self.assertIn("early_stop_detector_json=", marker)
            self.assertIn("early_stop_value/train/episode/return/shaped/mean=1000", marker)

    def test_polls_metric_store_at_rollout_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "run" / "rlab.sqlite"
            marker_path = Path(tmp) / "run" / "early_stop.txt"
            store = MetricStore(store_path)
            store.init()
            store.append_metrics(
                {EVAL_FULL_SUCCESS_RATE_MIN: 1.0},
                step=120000,
                source="eval",
                checkpoint_step=120000,
            )
            model = self.FakeModel()
            clock_value = 100.0
            callback = MetricThresholdStopHelper(
                detector=[
                    {
                        "metric": EVAL_FULL_SUCCESS_RATE_MIN,
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
                marker_path=marker_path,
                metric_store_path=store_path,
                poll_seconds=30.0,
                clock=lambda: clock_value,
            )
            callback.model = model  # type: ignore[assignment]
            callback.num_timesteps = 130000

            self.assertTrue(callback._on_step())
            callback._on_rollout_end()
            self.assertFalse(callback._on_step())

            marker = marker_path.read_text(encoding="utf-8")
            self.assertIn(f"early_stop_metric={EVAL_FULL_SUCCESS_RATE_MIN}", marker)
            self.assertIn("early_stop_value=1", marker)
            self.assertIn("timesteps=130000", marker)

    def test_metric_store_output_format_writes_complete_numeric_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "run" / "rlab.sqlite"
            output_format = MetricStoreOutputFormat(store_path)
            output_format.write(
                {
                    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN: 0.5,
                    "rollout/ep_rew_mean": np.float32(4.25),
                    "train/entropy_loss": -0.7,
                    "train/clip_range": 0.2,
                    "time/iterations": 3,
                    "ignored/text": "nope",
                    "ignored/bool": True,
                    "ignored/infinite": float("inf"),
                    "ignored/nan": float("nan"),
                },
                {},
                step=10,
            )

            store = MetricStore(store_path)
            self.assertEqual(store.latest_metric(TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN), 0.5)
            self.assertEqual(store.latest_metric("train/episode/return/shaped/mean"), 4.25)
            self.assertEqual(store.latest_metric("train/algorithm/ppo/policy/entropy"), 0.7)
            self.assertIsNone(store.latest_metric("train/clip_range"))
            self.assertIsNone(store.latest_metric("time/iterations"))
            self.assertIsNone(store.latest_metric("ignored/text"))
            self.assertIsNone(store.latest_metric("ignored/bool"))
            self.assertIsNone(store.latest_metric("ignored/infinite"))
            self.assertIsNone(store.latest_metric("ignored/nan"))
            with store.connection() as conn:
                count = conn.execute("SELECT count(*) FROM metric_frames").fetchone()[0]
            self.assertEqual(count, 1)
            with store.connection() as conn:
                row = conn.execute(
                    "SELECT step FROM metric_frames ORDER BY id DESC LIMIT 1"
                ).fetchone()
            self.assertEqual(row[0], 10)

    def test_metric_store_logger_helper_flushes_final_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "run" / "rlab.sqlite"
            logger = Logger(folder=None, output_formats=[])
            model = SimpleNamespace(logger=logger)
            callback = MetricStoreLoggerHelper(store_path)
            callback.model = model  # type: ignore[assignment]
            callback.num_timesteps = 10
            callback._on_training_start()
            callback._on_training_start()

            logger.record("train/value_loss", 1.25)
            callback._on_training_end()

            store = MetricStore(store_path)
            self.assertEqual(store.latest_metric("train/algorithm/ppo/update/value_loss"), 1.25)
            with store.connection() as conn:
                row = conn.execute(
                    "SELECT step, source FROM metric_frames ORDER BY id DESC LIMIT 1"
                ).fetchone()
            self.assertEqual(tuple(row), (10, "train"))
            self.assertEqual(logger.name_to_value, {})
            self.assertEqual(logger.output_formats.count(callback.output_format), 1)

    def test_sb3_lifecycle_publishes_rollout_metrics_after_dump_logs(self) -> None:
        class OneStepEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            action_space = gym.spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                del action
                return np.zeros(1, dtype=np.float32), 2.0, True, False, {}

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "run" / "rlab.sqlite"
            env = DummyVecEnv([lambda: Monitor(OneStepEnv())])
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                seed=1,
                verbose=0,
            )
            callback = RlabCallback([MetricStoreLoggerHelper(store_path)])
            try:
                model.learn(total_timesteps=4, callback=callback)
            finally:
                env.close()

            store = MetricStore(store_path)
            self.assertEqual(store.latest_metric("train/episode/return/shaped/mean"), 2.0)
            self.assertEqual(store.latest_metric("train/episode/length/mean"), 1.0)
            self.assertIsNone(store.latest_metric("time/fps"))
            self.assertIsNone(store.latest_metric("train/loss"))


class LedgerCheckpointHelperTests(unittest.TestCase):
    def test_checkpoint_save_uses_exact_sb3_zip_path_for_hidden_uuid_base(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.saved_bases: list[str] = []

            def save(self, path: str) -> None:
                self.saved_bases.append(path)
                target = Path(path if path.endswith(".zip") else path + ".zip")
                target.write_bytes(b"checkpoint")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            store_path = run_dir / "rlab.sqlite"
            callback = LedgerCheckpointHelper(
                args=argparse.Namespace(run_name="run", run_description=""),
                config=EnvConfig(game="SuperMarioBros-Nes-v0", state="Level1-1"),
                save_freq=1,
                save_path=run_dir / "checkpoints",
                name_prefix="ppo_supermariobros-nes-v0",
                metric_store_path=store_path,
            )
            callback.model = FakeModel()  # type: ignore[assignment]
            callback._init_callback()

            final_path = callback.save_checkpoint(500000, kind="checkpoint")

            self.assertTrue(final_path.is_file())
            self.assertEqual(final_path.name, "ppo_supermariobros-nes-v0_500000_steps.zip")
            self.assertTrue(callback.model.saved_bases[0].endswith(".zip"))  # type: ignore[attr-defined]
            self.assertFalse(any(final_path.parent.glob(".*.zip")))
            store = MetricStore(store_path)
            rows = store.pending_evals()
            self.assertEqual(rows[0]["path"], str(final_path))
            self.assertEqual(rows[0]["step"], 500000)


class ThroughputHelperTests(unittest.TestCase):
    def test_logs_one_complete_aligned_frame_at_the_next_rollout_start(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        times = iter([0.0, 2.0, 5.0, 7.0])
        callback = ThroughputHelper(clock=lambda: next(times))
        model = Model()
        callback.model = model  # type: ignore[assignment]

        callback.num_timesteps = 0
        callback._on_rollout_start()
        callback.num_timesteps = 100
        callback._on_rollout_end()

        callback.num_timesteps = 100
        callback._on_rollout_start()
        callback.num_timesteps = 220
        callback._on_rollout_end()

        self.assertEqual(
            dict(model.logger.records),
            {
                "train/throughput/loop_fps": 20.0,
                "train/throughput/rollout_fps": 50.0,
                "train/throughput/loop_seconds": 5.0,
                "train/throughput/rollout_seconds": 2.0,
                "train/throughput/between_rollouts_seconds": 3.0,
            },
        )

    def test_logs_native_env_step_throughput_when_available(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class NativeStatsEnv:
            def __init__(self) -> None:
                self.stats = [
                    {"seconds_total": 1.0, "calls_total": 10, "num_envs": 4},
                    {"seconds_total": 3.0, "calls_total": 35, "num_envs": 4},
                    {"seconds_total": 3.0, "calls_total": 35, "num_envs": 4},
                ]

            def native_step_stats(self):
                return self.stats.pop(0)

        class Wrapper:
            def __init__(self, env) -> None:
                self.venv = env

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()
                self.env = Wrapper(NativeStatsEnv())

        times = iter([0.0, 5.0, 7.0])
        callback = ThroughputHelper(clock=lambda: next(times))
        model = Model()
        callback.model = model  # type: ignore[assignment]

        callback.num_timesteps = 40
        callback._on_rollout_start()
        callback.num_timesteps = 140
        callback._on_rollout_end()
        callback._on_rollout_start()

        frame = dict(model.logger.records)
        self.assertEqual(frame["train/throughput/loop_seconds"], 7.0)
        self.assertEqual(frame["train/throughput/rollout_seconds"], 5.0)
        self.assertEqual(frame["train/throughput/between_rollouts_seconds"], 2.0)
        self.assertEqual(frame["train/throughput/env_step_seconds"], 2.0)
        self.assertEqual(frame["train/throughput/rollout_overhead_seconds"], 3.0)
        self.assertEqual(frame["train/throughput/env_step_fps"], 50.0)
        self.assertEqual(
            frame["train/throughput/loop_seconds"],
            frame["train/throughput/rollout_seconds"]
            + frame["train/throughput/between_rollouts_seconds"],
        )
        self.assertEqual(
            frame["train/throughput/rollout_seconds"],
            frame["train/throughput/env_step_seconds"]
            + frame["train/throughput/rollout_overhead_seconds"],
        )

    def test_complete_frame_uses_the_completed_rollout_step(self) -> None:
        class Run:
            def __init__(self) -> None:
                self.payloads = []

            def log(self, payload) -> None:
                self.payloads.append(payload)

        run = Run()
        times = iter([0.0, 2.0, 5.0])
        with tempfile.TemporaryDirectory() as tmp:
            callback = ThroughputHelper(
                clock=lambda: next(times),
                metric_store_path=Path(tmp) / "rlab.sqlite",
                wandb_run=run,
            )
            callback.model = SimpleNamespace(
                logger=SimpleNamespace(record=lambda *_args: None),
                env=None,
            )
            callback.num_timesteps = 0
            callback._on_rollout_start()
            callback.num_timesteps = 100
            callback._on_rollout_end()
            callback._on_rollout_start()

        self.assertEqual(len(run.payloads), 1)
        self.assertEqual(run.payloads[0]["global_step"], 100)


class RolloutDiagnosticsHelperTests(unittest.TestCase):
    def test_logs_value_prediction_and_advantage_stats(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class RolloutBuffer:
            values = np.array([[1.0, 2.0], [3.0, 4.0]])
            advantages = np.array([[-1.0, 0.0], [1.0, 2.0]])

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()
                self.rollout_buffer = RolloutBuffer()

        model = Model()
        callback = RolloutDiagnosticsHelper(log_histograms=False)
        callback.model = model  # type: ignore[assignment]

        callback._on_rollout_end()

        records = dict(model.logger.records)
        self.assertEqual(records["train/algorithm/ppo/rollout/value_prediction/mean"], 2.5)
        self.assertAlmostEqual(
            records["train/algorithm/ppo/rollout/value_prediction/std"],
            float(np.std([1.0, 2.0, 3.0, 4.0])),
        )
        self.assertEqual(records["train/algorithm/ppo/rollout/value_prediction/min"], 1.0)
        self.assertEqual(records["train/algorithm/ppo/rollout/value_prediction/max"], 4.0)
        self.assertEqual(records["train/algorithm/ppo/rollout/advantage/mean"], 0.5)
        self.assertAlmostEqual(
            records["train/algorithm/ppo/rollout/advantage/std"],
            float(np.std([-1.0, 0.0, 1.0, 2.0])),
        )
        self.assertEqual(records["train/algorithm/ppo/rollout/advantage/min"], -1.0)
        self.assertEqual(records["train/algorithm/ppo/rollout/advantage/max"], 2.0)
        self.assertFalse(any(name.endswith("abs_mean") for name in records))

    def test_logs_discrete_policy_dominant_action_rate(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        rollout_buffer = SimpleNamespace(
            values=np.asarray([0.0]),
            advantages=np.asarray([0.0]),
            actions=np.asarray([[1], [1], [1], [0]]),
        )
        model = SimpleNamespace(
            logger=logger,
            rollout_buffer=rollout_buffer,
            action_space=gym.spaces.Discrete(2),
        )
        callback = RolloutDiagnosticsHelper(log_histograms=False)
        callback.model = model  # type: ignore[assignment]

        callback._on_rollout_end()

        self.assertEqual(logger.records["train/algorithm/ppo/policy/dominant_action_rate"], 0.75)

    def test_logs_action_histogram_on_the_configured_rollout_cadence(self) -> None:
        run = SimpleNamespace(payload=None)
        run.log = lambda payload: setattr(run, "payload", payload)
        logger = SimpleNamespace(record=lambda *_args: None)
        model = SimpleNamespace(
            logger=logger,
            rollout_buffer=SimpleNamespace(
                values=np.asarray([1.0]),
                advantages=np.asarray([0.0]),
                actions=np.asarray([[0], [1], [1]]),
            ),
            action_space=gym.spaces.Discrete(2),
        )
        callback = RolloutDiagnosticsHelper(
            wandb_run=run,
            histogram_interval=1,
        )
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 64

        with mock.patch("wandb.Histogram", side_effect=lambda values: tuple(values)):
            callback._on_rollout_end()

        self.assertEqual(run.payload["global_step"], 64)
        self.assertEqual(
            run.payload["train/algorithm/ppo/policy/action_hist"],
            (0.0, 1.0, 1.0),
        )


class RuntimeMetricsRewardTests(unittest.TestCase):
    def test_consumes_batched_metric_records(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(active_reward_components=("progress", "death"))
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]

        self.assertTrue(
            callback._on_records(
                [
                    SimpleNamespace(
                        num_envs=2,
                        metrics={
                            "shaped_reward": np.asarray([1.0, 3.0]),
                            "progress_reward_component": np.asarray([2.0, 4.0]),
                            "death_penalty_component": np.asarray([0.0, -2.0]),
                        },
                    )
                ]
            )
        )
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/reward/shaped/mean"], 2.0)
        self.assertEqual(logger.records["train/reward/component/death/mean"], -1.0)

    def test_streams_reward_stats_filters_nonfinite_values_and_resets(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(active_reward_components=("progress", "death", "score"))
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]

        values = np.asarray([1.0, -2.0, np.nan, np.inf])
        callback._on_records([SimpleNamespace(num_envs=4, metrics={"shaped_reward": values})])
        values[:] = 100.0
        callback._on_records(
            [
                SimpleNamespace(
                    num_envs=2,
                    metrics={"shaped_reward": np.asarray([0.0, 5.0])},
                )
            ]
        )
        callback._on_rollout_end()

        expected = np.asarray([1.0, -2.0, 0.0, 5.0])
        self.assertAlmostEqual(logger.records["train/reward/shaped/mean"], np.mean(expected))
        self.assertAlmostEqual(logger.records["train/reward/shaped/std"], np.std(expected))
        self.assertEqual(logger.records["train/reward/shaped/min"], -2.0)
        self.assertEqual(logger.records["train/reward/shaped/max"], 5.0)
        self.assertAlmostEqual(
            logger.records["train/reward/shaped/nonzero_rate"],
            0.75,
        )

        logger.records.clear()
        callback._on_records(
            [
                SimpleNamespace(
                    num_envs=2,
                    metrics={"shaped_reward": np.asarray([7.0, 7.0])},
                )
            ]
        )
        callback._on_rollout_end()
        self.assertEqual(logger.records["train/reward/shaped/mean"], 7.0)
        self.assertEqual(logger.records["train/reward/shaped/std"], 0.0)

    def test_reward_shares_use_absolute_rollout_contribution(self) -> None:
        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        callback = RuntimeMetricsHelper(
            active_reward_components=("progress", "death"),
        )
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]

        callback._on_records(
            [
                SimpleNamespace(
                    num_envs=2,
                    metrics={
                        "progress_reward_component": np.asarray([1.0, 1.0]),
                        "death_penalty_component": np.asarray([0.0, -2.0]),
                    },
                )
            ]
        )
        callback._on_rollout_end()

        self.assertEqual(logger.records["train/reward/component/progress/share"], 0.5)
        self.assertEqual(logger.records["train/reward/component/death/share"], 0.5)
        self.assertNotIn("train/reward/component/score/share", logger.records)

    def test_flushes_runtime_metrics_without_a_direct_wandb_write(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.payloads: list[tuple[dict[str, object], int]] = []

            def log(self, payload: dict[str, object], *, step: int) -> None:
                self.payloads.append((payload, step))

        logger = SimpleNamespace(records={})
        logger.record = lambda key, value: logger.records.__setitem__(key, value)
        run = FakeRun()
        callback = RuntimeMetricsHelper(wandb_run=run)
        callback.model = SimpleNamespace(logger=logger)  # type: ignore[assignment]
        callback.num_timesteps = 64

        for _ in range(3):
            callback._on_records(
                [
                    SimpleNamespace(
                        lane=0,
                        start_id="Level1-1",
                        events=("life_loss",),
                        truncated=False,
                        episode_return=0.0,
                        outcome="failure",
                        metrics={"level_hi": 0, "level_lo": 0},
                    )
                ]
            )
        self.assertEqual(run.payloads, [])

        callback._on_rollout_end()

        self.assertEqual(run.payloads, [])
        self.assertEqual(logger.records["train/outcome/terminal/count"], 3)

    def test_reward_accumulator_reuses_preallocated_buffers(self) -> None:
        callback = RuntimeMetricsHelper()
        callback.model = SimpleNamespace(n_steps=100)  # type: ignore[assignment]
        for _ in range(100):
            callback._on_records(
                [
                    SimpleNamespace(
                        num_envs=16,
                        metrics={"shaped_reward": np.ones(16, dtype=np.float32)},
                    )
                ]
            )

        accumulator = callback.reward_stats.shaped
        buffer_id = id(accumulator.buffer)
        self.assertEqual(accumulator.size, 1600)
        self.assertEqual(accumulator.buffer.size, 1600)

        callback.model = SimpleNamespace(
            logger=SimpleNamespace(record=lambda *_: None), n_steps=100
        )
        callback._on_rollout_end()
        callback._on_records(
            [
                SimpleNamespace(
                    num_envs=16,
                    metrics={"shaped_reward": np.ones(16, dtype=np.float32)},
                )
            ]
        )
        self.assertEqual(id(accumulator.buffer), buffer_id)
        self.assertEqual(accumulator.size, 16)
