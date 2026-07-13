from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rlab.batch_runtime import EpisodeRecord
from rlab.env import EnvConfig
from rlab.eval import ScriptedPolicy
from rlab.eval_metrics import (
    episode_rank,
    episode_result_from_record,
    is_level_complete,
    run_eval_episode,
    summarize_episode_results,
)
from rlab.eval_runner import _eval_runtime_config, evaluate_model_episodes
from rlab.metric_names import EVAL_DURATION_SECONDS, metric_path_segment
from rlab.targets import target_for_game
from rlab.task_kernels import Outcome
from rlab.checkpoint_eval_worker import (
    checkpoint_eval_config_from_args,
    eval_score as eval_checkpoint_score,
    log_checkpoint_eval_metrics,
)
from rlab.task_kernels import default_task_document


class FakeWandbRun:
    def __init__(self, summary: object | None = None) -> None:
        self.payload: dict[str, object] | None = None
        self.kwargs: dict[str, object] | None = None
        self.summary = {} if summary is None else summary

    def log(self, payload: dict[str, object], **kwargs: object) -> None:
        self.payload = payload
        self.kwargs = kwargs


class WandbLikeSummary:
    def __init__(self) -> None:
        self.values: dict[str, object] = {"leader/checkpoint/steps_to_completion_goal": 500000}

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.values[key] = value

    def __delitem__(self, key: str) -> None:
        del self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __getattr__(self, key: str) -> object:
        raise KeyError(key)


def checkpoint_metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "checkpoint_step": 120000,
        "checkpoint_artifact": "entity/project/run-checkpoint:step-120000",
        "reward_mean": 10.0,
        "reward_std": 1.0,
        "reward_max": 12.0,
        "max_x_mean": 300.0,
        "max_x_max": 400.0,
        "max_level_x_mean": 300.0,
        "max_level_x_max": 400.0,
        "death_count": 1,
        "death_rate": 0.1,
        "best_episode": {"reward": 12.0, "max_x_pos": 400.0},
        "eval/done/all": 10,
        "eval/done/level_change": 9,
        "eval/done/level_change/rate": 0.9,
        "eval/done/level_change/from_rate/min": 0.8,
        "eval/done/level_change/from_rate/mean": 0.9,
        "eval/reward/mean": 10.0,
        EVAL_DURATION_SECONDS: 12.5,
    }
    metrics.update(overrides)
    return metrics


def log_checkpoint_eval(
    run: FakeWandbRun,
    metrics: dict[str, object] | None = None,
    *,
    step: int = 120000,
    artifact_ref: str = "entity/project/run-checkpoint:step-120000",
) -> None:
    log_checkpoint_eval_metrics(
        wandb_run=run,
        args=argparse.Namespace(hud_crop_top=32),
        metrics=checkpoint_metrics() if metrics is None else metrics,
        checkpoint_path=Path(f"/tmp/model_{step}_steps.zip"),
        checkpoint_step_value=step,
        artifact_ref=artifact_ref,
        config=EnvConfig(game="SuperMarioBros-Nes-v0", hud_crop_top=32),
    )


class EvalMetricTests(unittest.TestCase):
    def test_model_eval_rejects_deterministic_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic policy evaluation is unsupported"):
            evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=1,
                seed=10_000,
                max_steps=10,
                deterministic=True,
            )

    def test_scripted_policy_resets_per_episode_and_uses_bound_action_space(self) -> None:
        class ActionSpace:
            def sample(self) -> int:
                return 7

        right = ScriptedPolicy("right", ("noop", "right_b", "right_a_b"))
        first_action, _ = right.predict(None, deterministic=False)
        right.predict(None, deterministic=False)
        right.reset_episode()
        reset_action, _ = right.predict(None, deterministic=False)
        self.assertEqual(first_action.tolist(), reset_action.tolist())

        random = ScriptedPolicy("random", ())
        random.bind_action_space(ActionSpace())
        action, _ = random.predict(None, deterministic=False)
        self.assertEqual(action.tolist(), [7])

    def test_episode_record_clean_completion_precedence(self) -> None:
        success = EpisodeRecord(
            lane=0,
            episode_index=2,
            start_id="Level1-1",
            episode_return=12.5,
            episode_length=40,
            terminated=True,
            truncated=False,
            outcome=Outcome.SUCCESS,
            events=("level_change",),
            metrics={"max_x_pos": 3200, "completion_event": True, "died": False},
        )
        simultaneous_failure = EpisodeRecord(
            lane=1,
            episode_index=3,
            start_id="Level1-2",
            episode_return=-2.0,
            episode_length=12,
            terminated=True,
            truncated=False,
            outcome=Outcome.FAILURE,
            events=("life_loss", "level_change"),
            metrics={"max_x_pos": 900, "completion_event": False, "died": True},
        )

        success_result = episode_result_from_record(success)
        failure_result = episode_result_from_record(simultaneous_failure)

        self.assertTrue(success_result["level_complete"])
        self.assertEqual(success_result["outcome"], "success")
        self.assertEqual(success_result["start_state"], "Level1-1")
        self.assertFalse(failure_result["level_complete"])
        self.assertTrue(failure_result["died"])
        self.assertGreater(episode_rank(success_result), episode_rank(failure_result))

    def test_checkpoint_score_prefers_min_completion_rate_when_available(self) -> None:
        metrics = {
            "completion_rate": 0.95,
            "eval/done/level_change/from_rate/min": 0.80,
            "eval/done/level_change/from_rate/mean": 0.90,
            "checkpoint_step": 5000000,
            "max_x_max": 3200,
            "reward_mean": 1200.0,
        }

        self.assertEqual(eval_checkpoint_score(metrics), (0.8, 0.9, float("-inf"), 1200.0))

    def test_checkpoint_score_uses_reward_when_completion_is_absent(self) -> None:
        metrics = {
            "reward_mean": 34.0,
            "reward_max": 55.0,
            "checkpoint_step": 5000000,
        }

        self.assertEqual(eval_checkpoint_score(metrics), (34.0, 55.0, -5000000.0))

    def test_checkpoint_score_executes_explicit_goal_rank(self) -> None:
        metrics = {
            "reward_mean": 34.0,
            "reward_max": 55.0,
            "checkpoint_step": 5000000,
        }

        self.assertEqual(
            eval_checkpoint_score(
                metrics,
                [
                    "min(leader/checkpoint/steps_to_completion_goal)",
                    "max(eval/reward/mean)",
                ],
            ),
            (-5000000.0, 34.0),
        )

    def test_generic_eval_summary_does_not_emit_mario_completion_metrics(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "default",
                    "reward": 10.0,
                    "steps": 100,
                    "terminated": True,
                    "truncated": False,
                    "final_info": {"ale.lives": 4},
                },
                {
                    "start_state": "default",
                    "reward": 4.0,
                    "steps": 200,
                    "terminated": False,
                    "truncated": True,
                    "final_info": {"ale.lives": 3},
                },
            ],
            deterministic=False,
            semantics=target_for_game("breakout").eval_semantics,
        )

        self.assertEqual(summary["reward_mean"], 7.0)
        self.assertEqual(summary["eval/done/all"], 2)
        self.assertEqual(summary["eval/done/terminated"], 1)
        self.assertEqual(summary["eval/done/terminated/rate"], 0.5)
        self.assertEqual(summary["eval/done/max_steps"], 1)
        self.assertNotIn("completion_count", summary)
        self.assertNotIn("eval/done/level_change", summary)
        self.assertNotIn("max_x_mean", summary)
        self.assertNotIn("death_count", summary)

    def test_generic_eval_summary_counts_configured_terminal_events(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "Start",
                    "reward": 10.0,
                    "steps": 856,
                    "terminated": True,
                    "truncated": False,
                    "events": ["serve_stall"],
                    "final_info": {"ball_y": 0},
                },
                {
                    "start_state": "Start",
                    "reward": 4.0,
                    "steps": 54000,
                    "terminated": False,
                    "truncated": True,
                    "events": [],
                    "final_info": {"ball_y": 32},
                },
            ],
            deterministic=False,
            semantics=target_for_game("breakout").eval_semantics,
        )

        self.assertEqual(summary["eval/done/serve_stall"], 1)
        self.assertEqual(summary["eval/done/serve_stall/rate"], 0.5)
        self.assertEqual(summary["eval/done/serve_stall/from/Start"], 1)
        self.assertEqual(summary["eval/done/serve_stall/from/Start/rate"], 0.5)
        self.assertEqual(summary["eval/done/unclassified"], 0)

    def test_eval_runtime_preserves_non_life_loss_failures(self) -> None:
        config = EnvConfig(
            game="Breakout-Atari2600-v0",
            task={
                "termination": {
                    "failure": ["life_loss", "serve_stall"],
                    "max_episode_steps": 54000,
                }
            },
        )

        runtime_config = _eval_runtime_config(
            config,
            max_steps=54000,
            semantics=target_for_game("breakout").eval_semantics,
        )

        self.assertEqual(runtime_config.task["termination"]["failure"], ["serve_stall"])

    def test_checkpoint_score_prefers_fewer_timesteps_after_completion_goal(self) -> None:
        slower_higher_reward = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 1.0,
            "eval/done/level_change/from_rate/mean": 1.0,
            "checkpoint_step": 5000000,
            "reward_mean": 1200.0,
        }
        faster_lower_reward = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 1.0,
            "eval/done/level_change/from_rate/mean": 1.0,
            "checkpoint_step": 3500000,
            "reward_mean": 900.0,
        }

        self.assertGreater(
            eval_checkpoint_score(faster_lower_reward),
            eval_checkpoint_score(slower_higher_reward),
        )

    def test_async_checkpoint_eval_logs_checkpoint_step_as_metric(self) -> None:
        run = FakeWandbRun()

        log_checkpoint_eval(run)

        assert run.payload is not None
        self.assertEqual(run.kwargs, {})
        self.assertEqual(run.payload["global_step"], 120000)
        self.assertEqual(run.payload["eval/checkpoint/step"], 120000)
        self.assertEqual(run.payload[EVAL_DURATION_SECONDS], 12.5)
        self.assertEqual(run.summary["leader/checkpoint/eval_source"], "async_worker")
        self.assertEqual(run.summary["leader/checkpoint/completion_rate"], 0.8)
        self.assertEqual(run.summary["leader/checkpoint/completion_rate_mean"], 0.9)
        self.assertEqual(run.summary["leader/checkpoint/reward_mean"], 10.0)
        self.assertNotIn("leader/checkpoint/steps_to_completion_goal", run.summary)

    def test_async_checkpoint_summary_handles_wandb_summary_without_pop(self) -> None:
        run = FakeWandbRun(summary=WandbLikeSummary())

        log_checkpoint_eval(run)

        self.assertNotIn("leader/checkpoint/steps_to_completion_goal", run.summary)
        self.assertEqual(run.summary["leader/checkpoint/eval_source"], "async_worker")

    def test_async_checkpoint_summary_tracks_steps_to_completion_goal(self) -> None:
        run = FakeWandbRun()
        metrics = checkpoint_metrics(
            **{
                "checkpoint_step": 3500000,
                "checkpoint_artifact": "entity/project/run-checkpoint:step-3500000",
                "eval/done/level_change": 10,
                "eval/done/level_change/rate": 1.0,
                "eval/done/level_change/from_rate/min": 1.0,
                "eval/done/level_change/from_rate/mean": 1.0,
            }
        )

        log_checkpoint_eval(
            run,
            metrics,
            step=3500000,
            artifact_ref="entity/project/run-checkpoint:step-3500000",
        )

        self.assertEqual(run.summary["leader/checkpoint/steps_to_completion_goal"], 3500000)

    def test_checkpoint_eval_config_uses_goal_termination(self) -> None:
        task = default_task_document("mario")
        task["termination"] = {
            **task["termination"],
            "failure": [],
            "success": ["level_change"],
        }
        eval_config = checkpoint_eval_config_from_args(
            argparse.Namespace(
                checkpoint_eval_environment={
                    "env_provider": "supermariobrosnes-turbo",
                    "game": "SuperMarioBros-Nes-v0",
                    "task": task,
                }
            )
        )

        self.assertEqual(eval_config.task["termination"]["failure"], [])
        self.assertEqual(eval_config.task["termination"]["success"], ["level_change"])

    def test_checkpoint_eval_config_does_not_inherit_training_termination(self) -> None:
        task = default_task_document("mario")
        task["termination"] = {
            **task["termination"],
            "failure": [],
            "success": [],
        }
        eval_config = checkpoint_eval_config_from_args(
            argparse.Namespace(
                checkpoint_eval_environment={
                    "env_provider": "supermariobrosnes-turbo",
                    "game": "SuperMarioBros-Nes-v0",
                    "task": task,
                }
            )
        )

        self.assertEqual(eval_config.task["termination"]["failure"], [])
        self.assertEqual(eval_config.task["termination"]["success"], [])

    def test_metric_path_segment_preserves_retro_state_names(self) -> None:
        self.assertEqual(metric_path_segment("Level1-2"), "Level1-2")
        self.assertEqual(metric_path_segment("Level 1/2"), "Level_1_2")

    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "reward": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "reward": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "reward": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))

    def test_level_complete_uses_explicit_completion_flag(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": False, "level_max_x_pos": 5000},
            )
        )
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": True},
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_complete": True, "level_changed": True},
            )
        )

    def test_run_eval_episode_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.array([0], dtype=np.int64), None

        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def seed(self, seed: int) -> None:
                self.seed_value = seed

            def reset(self):
                self.step_count = 0
                self.records = []
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0], dtype=np.float32),
                        np.array([False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                                "score": 10,
                                "lives": 3,
                                "time": 300,
                            }
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=3.0,
                        episode_length=2,
                        terminated=False,
                        truncated=True,
                        outcome=Outcome.TIMEOUT,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([2.0], dtype=np.float32),
                    np.array([True]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "score": 20,
                            "lives": 3,
                            "time": 299,
                        }
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

        result = run_eval_episode(
            FakeEnv(),
            FakeModel(),
            max_steps=2,
            deterministic=False,
            seed=7,
            default_start_state="Level1-1",
        )

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["reward"], 3.0)
        self.assertEqual(result["max_x_pos"], 250)
        self.assertEqual(result["start_state"], "Level1-1")
        self.assertTrue(result["level_complete"])
        self.assertFalse(result["terminated"])
        self.assertTrue(result["truncated"])

    def test_vector_eval_accumulates_completed_slots_independently(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def reset(self):
                self.records = []
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    self.records = [
                        EpisodeRecord(
                            lane=1,
                            episode_index=0,
                            start_id="Level1-2",
                            episode_return=2.0,
                            episode_length=1,
                            terminated=False,
                            truncated=True,
                            outcome=Outcome.TIMEOUT,
                            events=(),
                            metrics={
                                "max_x_pos": 20,
                                "level_max_x_pos": 20,
                                "died": True,
                            },
                        )
                    ]
                    return (
                        obs,
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([False, True]),
                        [
                            {"max_x_pos": 10, "level_max_x_pos": 10},
                            {
                                "start_state": "Level1-2",
                                "max_x_pos": 20,
                                "level_max_x_pos": 20,
                                "died": True,
                                "death_x_pos": 20,
                                "TimeLimit.truncated": True,
                                "score": 100,
                                "lives": 2,
                            },
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=4.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.SUCCESS,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([3.0, 4.0], dtype=np.float32),
                    np.array([True, False]),
                    [
                        {
                            "start_state": "Level1-1",
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "level_changed": True,
                            "score": 200,
                            "lives": 3,
                        },
                        {"max_x_pos": 40, "level_max_x_pos": 40},
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        config = EnvConfig(game="SuperMarioBros-Nes-v0")
        with (
            patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeVecEnv()),
            patch("rlab.eval_runner.time.perf_counter", side_effect=[10.0, 12.5]),
        ):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=2,
                seed=7,
                max_steps=10,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["eval_n_envs"], 2)
        self.assertEqual(metrics[EVAL_DURATION_SECONDS], 2.5)
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["reward_mean"], 3.0)
        self.assertEqual(metrics["completion_count"], 1)
        self.assertEqual(metrics["death_count"], 1)
        self.assertEqual(metrics["terminated_count"], 1)
        self.assertEqual(metrics["terminated_rate"], 0.5)
        self.assertEqual(metrics["truncated_count"], 1)
        self.assertEqual(metrics["truncated_rate"], 0.5)
        self.assertEqual(metrics["eval/done/all"], 2)
        self.assertEqual(metrics["eval/done/level_change"], 1)
        self.assertEqual(metrics["eval/done/level_change/rate"], 0.5)
        self.assertEqual(metrics["eval/done/max_steps"], 1)
        self.assertEqual(metrics["eval/done/max_steps/rate"], 0.5)
        self.assertEqual(metrics["eval/done/unclassified"], 0)
        self.assertEqual(metrics["eval/done/unclassified/rate"], 0.0)
        self.assertEqual(metrics["eval/done/all/from/Level1-1"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-1"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-1/rate"], 1.0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-1"], 0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-1/rate"], 0.0)
        self.assertEqual(metrics["eval/done/all/from/Level1-2"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-2"], 0)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-2/rate"], 0.0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-2"], 1)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-2/rate"], 1.0)
        self.assertEqual(metrics["eval/done/level_change/from_rate/min"], 0.0)
        self.assertEqual(metrics["eval/done/level_change/from_rate/mean"], 0.5)
        self.assertEqual(metrics["eval/done/level_change/from_rate/min"], 0.0)
        self.assertEqual(metrics["eval/done/level_change/from_rate/mean"], 0.5)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["seed"], 7)
        self.assertEqual(metrics["episode_results"][0]["seed_protocol"], "vector-lane-v1")
        self.assertEqual(metrics["episode_results"][0]["seed_lane"], 1)
        self.assertEqual(metrics["episode_results"][0]["seed_episode_ordinal"], 0)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-2")
        self.assertEqual(metrics["episode_results"][0]["reward"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["seed_lane"], 0)
        self.assertEqual(metrics["episode_results"][1]["seed_episode_ordinal"], 0)
        self.assertEqual(metrics["episode_results"][1]["start_state"], "Level1-1")
        self.assertEqual(metrics["episode_results"][1]["reward"], 4.0)

    def test_vector_eval_uses_canonical_episode_records(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeRecordVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.records = []

            def reset(self):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=4.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.SUCCESS,
                        events=("level_change",),
                        metrics={"max_x_pos": 300, "completion_event": True},
                    ),
                    EpisodeRecord(
                        lane=1,
                        episode_index=0,
                        start_id="Level1-2",
                        episode_return=-1.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.FAILURE,
                        events=("life_loss", "level_change"),
                        metrics={"max_x_pos": 100, "completion_event": False, "died": True},
                    ),
                ]
                return (
                    np.zeros((2, 4, 84, 84), dtype=np.uint8),
                    np.zeros(2, dtype=np.float32),
                    np.ones(2, dtype=bool),
                    [{"score": 10, "lives": 3}, {"score": 5, "lives": 2}],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        with patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeRecordVecEnv()):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=2,
                seed=7,
                max_steps=10,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["completion_count"], 1)
        self.assertEqual(metrics["death_count"], 1)
        self.assertEqual(metrics["best_episode"]["outcome"], "success")
        self.assertTrue(metrics["episode_results"][0]["level_complete"])
        self.assertFalse(metrics["episode_results"][1]["level_complete"])

    def test_vector_eval_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def reset(self):
                self.step_count = 0
                self.records = []
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0, 10.0], dtype=np.float32),
                        np.array([False, False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                            },
                            {
                                "start_state": "Level1-2",
                                "state": "Level1-2",
                                "max_x_pos": 110,
                                "level_max_x_pos": 110,
                                "level_changed": True,
                            },
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=3.0,
                        episode_length=2,
                        terminated=False,
                        truncated=True,
                        outcome=Outcome.TIMEOUT,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([2.0, 20.0], dtype=np.float32),
                    np.array([True, False]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                        },
                        {
                            "state": "Level1-3",
                            "max_x_pos": 260,
                            "level_max_x_pos": 160,
                        },
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        fake_env = FakeVecEnv()
        config = EnvConfig(game="SuperMarioBros-Nes-v0")
        with patch("rlab.eval_runner.make_eval_vec_env", return_value=fake_env):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=1,
                seed=7,
                max_steps=2,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(fake_env.step_count, 2)
        self.assertEqual(metrics["episodes"], 1)
        self.assertEqual(metrics["completion_count"], 1)
        self.assertEqual(metrics["terminated_count"], 0)
        self.assertEqual(metrics["truncated_count"], 1)
        self.assertEqual(metrics["eval/done/level_change"], 1)
        self.assertEqual(metrics["eval/done/max_steps"], 1)
        self.assertEqual(metrics["episode_results"][0]["steps"], 2)
        self.assertEqual(metrics["episode_results"][0]["reward"], 3.0)
        self.assertEqual(metrics["episode_results"][0]["max_x_pos"], 250)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-1")
        self.assertTrue(metrics["episode_results"][0]["level_complete"])
        self.assertFalse(metrics["episode_results"][0]["terminated"])
        self.assertTrue(metrics["episode_results"][0]["truncated"])

    def test_evaluate_model_episodes_updates_progress_bar(self) -> None:
        class FakeEnv:
            def close(self) -> None:
                pass

        class FakeProgressBar:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def update(self, count: int) -> None:
                self.updates.append(count)

        progress_bars: list[FakeProgressBar] = []

        def fake_tqdm(**kwargs) -> FakeProgressBar:
            progress_bar = FakeProgressBar(**kwargs)
            progress_bars.append(progress_bar)
            return progress_bar

        def fake_run_eval_episode(*args, **kwargs) -> dict:
            return {
                "actions": [],
                "start_state": "Level1-1",
                "reward": 1.0,
                "max_x_pos": 10,
                "max_level_x_pos": 10,
                "score": 0,
                "lives": 3,
                "time": 399,
                "steps": 1,
                "terminated": True,
                "truncated": False,
                "level_complete": True,
                "died": False,
                "death_x_pos": None,
                "final_info": {"start_state": "Level1-1"},
            }

        with (
            patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeEnv()),
            patch("rlab.eval_runner.run_eval_episode", side_effect=fake_run_eval_episode),
            patch("rlab.eval_runner.tqdm", side_effect=fake_tqdm),
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=3,
                seed=7,
                max_steps=10,
                deterministic=False,
                progress=True,
                progress_description="eval checkpoint 4100000",
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["episodes"], 3)
        self.assertEqual(len(progress_bars), 1)
        self.assertEqual(progress_bars[0].kwargs["total"], 3)
        self.assertEqual(progress_bars[0].kwargs["desc"], "eval checkpoint 4100000")
        self.assertEqual(progress_bars[0].kwargs["disable"], False)
        self.assertEqual(progress_bars[0].updates, [1, 1, 1])

    def test_best_episode_video_replays_through_eval_vec_env(self) -> None:
        class FakePolicyEnv:
            def close(self) -> None:
                pass

        class FakeVideoEnv:
            def __init__(self) -> None:
                self.actions = []
                self.frame = 0

            def seed(self, seed: int) -> None:
                self.seed_value = seed

            def reset(self):
                self.frame = 0
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                self.frame += 1
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=bool),
                    [{}],
                )

            def get_images(self):
                return [np.full((4, 4, 3), self.frame, dtype=np.uint8)]

            def close(self) -> None:
                pass

        result = {
            "actions": [1, 2],
            "start_state": "Level1-1",
            "reward": 3.0,
            "max_x_pos": 20,
            "max_level_x_pos": 20,
            "score": 0,
            "lives": 3,
            "time": 399,
            "steps": 2,
            "terminated": True,
            "truncated": False,
            "level_complete": True,
            "died": False,
            "death_x_pos": None,
            "final_info": {},
        }
        video_env = FakeVideoEnv()
        output = Path("/tmp/rlab-eval-video.mp4")
        with (
            patch(
                "rlab.eval_runner.make_eval_vec_env",
                side_effect=[FakePolicyEnv(), video_env],
            ) as make_env,
            patch("rlab.eval_runner.run_eval_episode", return_value=result),
            patch("rlab.eval_runner.write_video") as write_video,
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=1,
                seed=10_007,
                max_steps=10,
                deterministic=False,
                capture_best_video=True,
                video_path=output,
            )

        self.assertEqual(video_path, output)
        self.assertEqual(metrics["best_episode_video"], str(output))
        self.assertEqual(make_env.call_count, 2)
        self.assertEqual(
            make_env.call_args_list[1].kwargs["config"].task["termination"]["success"],
            [],
        )
        self.assertEqual(len(video_env.actions), 2)
        written_frames = write_video.call_args.args[0]
        self.assertEqual(len(written_frames), 3)
