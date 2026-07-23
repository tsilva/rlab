from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from rlab.batch_runtime import BatchRuntime, EpisodeRecord, ProviderDescriptor, SignalSpec
from rlab.snapshot_curriculum import (
    SnapshotCurriculum,
    SnapshotCurriculumConfig,
    normalize_snapshot_curriculum_config,
    validate_snapshot_curriculum_runtime_contract,
)
from rlab.task_kernels import IdentityTaskDefinition
from rlab.training.sb3_vec_env import RlabVecEnv


def curriculum_config(
    *,
    n_envs: int = 3,
    restore_snapshots: bool = True,
) -> dict[str, Any]:
    value = normalize_snapshot_curriculum_config(
        {
            "cell": {"signal": "score", "bucket_size": 50},
            "snapshot_share": 0.2,
            "priority_metric": "value_error",
            "restore_snapshots": restore_snapshots,
        },
        n_envs=n_envs,
    )
    assert value is not None
    return value


class SnapshotProvider:
    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(0, 10_000, shape=(1,), dtype=np.int64)
        self.single_action_space = gym.spaces.Discrete(2)
        self.observations = np.zeros((num_envs, 1), dtype=np.int64)
        self.score = np.zeros(num_envs, dtype=np.int64)
        self.queued: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []
        self.capture_masks: list[np.ndarray] = []

    def queue_step(self, **values: Any) -> None:
        self.queued.append(values)

    def _infos(self, *, mask: np.ndarray | None = None, sources=None):
        infos: dict[str, Any] = {"score": self.score.copy()}
        if mask is not None:
            infos.update(
                {
                    "_score": mask.copy(),
                    "start_id": np.full(self.num_envs, "Start", dtype=object),
                    "_start_id": mask.copy(),
                    "start_source": np.asarray(sources, dtype=object),
                    "_start_source": mask.copy(),
                }
            )
        return infos

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        options = dict(options or {})
        mask = np.asarray(
            options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
            dtype=np.bool_,
        )
        seeds = [None] * self.num_envs if seed is None else list(seed)
        snapshots = tuple(options.get("snapshots", (None,) * self.num_envs))
        sources = np.full(self.num_envs, None, dtype=object)
        for lane in np.flatnonzero(mask):
            lane_index = int(lane)
            snapshot = snapshots[lane_index]
            if snapshot is None:
                self.score[lane_index] = 0
                self.observations[lane_index, 0] = 0
                sources[lane_index] = "environment"
            else:
                if seeds[lane_index] is not None:
                    raise ValueError("snapshot reset seed must be None")
                self.score[lane_index] = int(snapshot["score"])
                self.observations[lane_index, 0] = int(snapshot["observation"])
                sources[lane_index] = "snapshot"
        self.reset_calls.append(
            {"mask": mask.copy(), "seeds": tuple(seeds), "snapshots": snapshots}
        )
        return self.observations, self._infos(mask=mask, sources=sources)

    def capture_snapshots(self, mask: np.ndarray):
        self.capture_masks.append(np.asarray(mask, dtype=np.bool_).copy())
        return tuple(
            {
                "score": int(self.score[lane]),
                "observation": int(self.observations[lane, 0]),
            }
            if bool(mask[lane])
            else None
            for lane in range(self.num_envs)
        )

    def step(self, actions):
        del actions
        values = self.queued.pop(0) if self.queued else {}
        self.observations[:, 0] += 1
        if "score" in values:
            self.score[:] = np.asarray(values["score"], dtype=np.int64)
        return (
            self.observations,
            np.asarray(values.get("rewards", [0.0] * self.num_envs), dtype=np.float32),
            np.asarray(values.get("terminated", [False] * self.num_envs), dtype=np.bool_),
            np.asarray(values.get("truncated", [False] * self.num_envs), dtype=np.bool_),
            self._infos(),
        )

    def close(self):
        return None


class SnapshotCurriculumConfigTests(unittest.TestCase):
    def test_snapshot_restoring_is_disabled_by_default(self) -> None:
        normalized = normalize_snapshot_curriculum_config(
            {
                "cell": {"signal": "score", "bucket_size": 50},
                "snapshot_share": 0.2,
                "priority_metric": "value_error",
            },
            n_envs=128,
        )

        assert normalized is not None
        self.assertIs(normalized["restore_snapshots"], False)

    def test_breakout_defaults_resolve_twenty_percent_of_128_lanes(self) -> None:
        normalized = curriculum_config(n_envs=128)

        self.assertEqual(normalized["resolved_snapshot_lanes"], 26)
        self.assertEqual(normalized["representatives_per_cell"], 4)
        self.assertEqual(normalized["max_snapshots"], 1024)
        self.assertEqual(normalized["feedback_ema_alpha"], 0.1)

    def test_scalar_floor_cells_have_versioned_edge_semantics(self) -> None:
        config = SnapshotCurriculumConfig.from_mapping(curriculum_config(), n_envs=3)
        archive = SnapshotCurriculum(config, n_envs=3, run_seed=7, global_lane_ids=(0, 1, 2))

        self.assertEqual(archive.cell_id(49.999), "score:0")
        self.assertEqual(archive.cell_id(50), "score:1")
        self.assertEqual(archive.cell_id(-0.001), "score:-1")
        with self.assertRaisesRegex(ValueError, "finite"):
            archive.cell_id(float("nan"))

    def test_priority_name_is_backend_capability_not_archive_domain_knowledge(self) -> None:
        config = curriculum_config()
        config["priority_metric"] = "temporal_difference_error"
        normalized = normalize_snapshot_curriculum_config(config, n_envs=3)
        self.assertEqual(normalized["priority_metric"], "temporal_difference_error")

        common = {
            "n_envs": 3,
            "snapshot_curriculum": normalized,
            "env_args": {},
            "task": {"signals": {"score": "score"}},
        }
        validate_snapshot_curriculum_runtime_contract(
            common,
            backend_id="custom.actor_critic",
            supported_priority_metrics=("temporal_difference_error",),
        )
        with self.assertRaisesRegex(ValueError, "does not provide snapshot priority"):
            validate_snapshot_curriculum_runtime_contract(
                common,
                backend_id="custom.policy_gradient",
                supported_priority_metrics=(),
            )

    def test_value_error_and_staleness_sampler_is_bounded(self) -> None:
        config = SnapshotCurriculumConfig.from_mapping(curriculum_config(), n_envs=3)
        archive = SnapshotCurriculum(config, n_envs=3, run_seed=11, global_lane_ids=(0, 1, 2))
        for index in range(5):
            cell_id = f"score:{index}"
            self.assertTrue(archive.admit(cell_id, {"cell": index}))
            selection = archive.sample(lane=0, episode_index=index)
            archive.close_episode(selection.cell_id)
            archive.submit_feedback(selection.cell_id, float(index + 1))
        metrics = archive.complete_rollout()

        self.assertLessEqual(metrics["sampling_probability_max"], 0.25 + 1e-12)
        self.assertGreaterEqual(metrics["sampling_effective_cell_count"], 4.0)


class SnapshotCurriculumRuntimeTests(unittest.TestCase):
    def make_env(self) -> tuple[SnapshotProvider, BatchRuntime, RlabVecEnv]:
        provider = SnapshotProvider()
        descriptor = ProviderDescriptor(
            provider_id="breakout-turbo-env",
            native_observation_space=provider.single_observation_space,
            native_action_space=provider.single_action_space,
            signal_schema={"score": SignalSpec("score", np.int64)},
            start_catalog=("Start",),
            supports_live_snapshots=True,
            live_snapshots_deterministic=True,
        )
        kernel = IdentityTaskDefinition(signals={"score": "score"}).bind(
            descriptor, provider.num_envs
        )
        runtime = BatchRuntime(
            provider,
            descriptor,
            kernel,
            run_seed=17,
            snapshot_curriculum=curriculum_config(),
        )
        return provider, runtime, RlabVecEnv(runtime)

    def test_provider_preflight_proves_exact_masked_round_trip(self) -> None:
        provider, runtime, env = self.make_env()

        receipt = runtime.preflight_snapshot_round_trip(seed=17)

        self.assertEqual(receipt["provider_id"], "breakout-turbo-env")
        self.assertEqual(receipt["cell_id"], "score:0")
        self.assertTrue(receipt["observation_exact"])
        self.assertTrue(receipt["one_step_continuation_exact"])
        self.assertTrue(receipt["masked_capture"])
        self.assertEqual(provider.capture_masks[-1].tolist(), [True, False, False])
        self.assertEqual(provider.reset_calls[-1]["mask"].tolist(), [True, True, True])
        env.close()

    def test_capture_only_archive_never_schedules_snapshot_restores(self) -> None:
        provider = SnapshotProvider()
        descriptor = ProviderDescriptor(
            provider_id="breakout-turbo-env",
            native_observation_space=provider.single_observation_space,
            native_action_space=provider.single_action_space,
            signal_schema={"score": SignalSpec("score", np.int64)},
            start_catalog=("Start",),
            supports_live_snapshots=True,
            live_snapshots_deterministic=True,
        )
        kernel = IdentityTaskDefinition(signals={"score": "score"}).bind(
            descriptor, provider.num_envs
        )
        runtime = BatchRuntime(
            provider,
            descriptor,
            kernel,
            run_seed=17,
            snapshot_curriculum=curriculum_config(restore_snapshots=False),
        )
        env = RlabVecEnv(runtime)
        receipt = runtime.preflight_snapshot_capture(seed=17)
        self.assertIs(receipt["restore_snapshots"], False)
        self.assertTrue(receipt["masked_capture"])
        self.assertTrue(all(value is None for value in provider.reset_calls[-1]["snapshots"]))
        env.seed(17)
        env.reset()
        runtime.curriculum_begin_rollout()
        provider.queue_step(score=[50, 0, 0])
        env.step(np.zeros(3, dtype=np.int64))

        metrics = runtime.curriculum_complete_rollout()

        self.assertEqual(metrics["archive_cell_count"], 1.0)
        self.assertFalse(runtime.snapshot_curriculum.activation_scheduled)
        self.assertFalse(np.any(runtime.snapshot_curriculum.snapshot_lane_mask))
        self.assertFalse(runtime._has_pending_resets)
        env.close()

    def test_score_crossing_activates_snapshot_lane_without_fake_episode(self) -> None:
        provider, runtime, env = self.make_env()
        env.seed(17)
        env.reset()
        runtime.curriculum_begin_rollout()
        provider.queue_step(score=[50, 0, 0])
        _observations, _rewards, dones, _infos = env.step(np.zeros(3, dtype=np.int64))
        self.assertFalse(np.any(dones))

        metrics = runtime.curriculum_complete_rollout()
        self.assertEqual(metrics["archive_cell_count"], 1.0)
        provider.queue_step(score=[50, 0, 0])
        _observations, _rewards, dones, infos = env.step(np.zeros(3, dtype=np.int64))

        np.testing.assert_array_equal(dones, [True, False, False])
        self.assertNotIn("episode", infos[0])
        activation_sidecar = env.take_curriculum_step()
        self.assertTrue(activation_sidecar.control_boundaries[0])
        self.assertEqual(runtime.drain_records(), [])
        self.assertEqual(runtime._start_origins.tolist(), ["curriculum", "target", "target"])
        self.assertEqual(runtime._curriculum_cell_ids[0], "score:1")
        self.assertIsNone(provider.reset_calls[-1]["seeds"][0])

        provider.queue_step(score=[50, 0, 0], terminated=[True, False, False])
        _observations, _rewards, dones, infos = env.step(np.zeros(3, dtype=np.int64))
        self.assertTrue(dones[0])
        self.assertIn("episode", infos[0])
        sidecar = env.take_curriculum_step()
        self.assertTrue(sidecar.curriculum_feedback_dones[0])
        episode = next(
            record for record in runtime.drain_records() if isinstance(record, EpisodeRecord)
        )
        self.assertEqual(episode.start_origin, "curriculum")
        self.assertEqual(episode.curriculum_cell_id, "score:1")

        # BatchRuntime reuses two BatchStep objects. The callback-facing sidecar must
        # retain its owned data after that same backing slot is reused.
        provider.queue_step(score=[50, 0, 0])
        env.step(np.zeros(3, dtype=np.int64))
        self.assertTrue(activation_sidecar.control_boundaries[0])


if __name__ == "__main__":
    unittest.main()
