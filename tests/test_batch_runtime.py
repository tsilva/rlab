from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from rlab.batch_runtime import (
    BatchMetricRecord,
    BatchRuntime,
    EpisodeRecord,
    ProviderDescriptor,
    RlabVecEnv,
    SignalSpec,
    TaskEventRecord,
)
from rlab.env import EnvConfig
from rlab.task_kernels import (
    IdentityTaskDefinition,
    MarioTaskConfig,
    MarioTaskDefinition,
    Outcome,
)


class DeterministicNativeVectorProvider:
    """Manual-reset provider whose returned arrays are intentionally reused."""

    def __init__(self, num_envs: int = 2):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, shape=(2,), dtype=np.uint8),
                "aux": gym.spaces.Box(-1000, 1000, shape=(1,), dtype=np.int32),
            }
        )
        self.single_action_space = gym.spaces.MultiBinary(3)
        self.render_mode = "rgb_array"
        self._observations = {
            "image": np.zeros((num_envs, 2), dtype=np.uint8),
            "aux": np.zeros((num_envs, 1), dtype=np.int32),
        }
        self._x = np.zeros(num_envs, dtype=np.int64)
        self._score = np.zeros(num_envs, dtype=np.int64)
        self._lives = np.full(num_envs, 3, dtype=np.int64)
        self._level_hi = np.ones(num_envs, dtype=np.int64)
        self._level_lo = np.ones(num_envs, dtype=np.int64)
        self._queued_steps: list[dict[str, Any]] = []
        self.reset_calls: list[dict[str, Any]] = []
        self.step_actions: list[Any] = []
        self.closed = False

    def queue_step(self, **values: Any) -> None:
        self._queued_steps.append(values)

    def _infos(self, start_ids: Sequence[str | None] | None = None) -> dict[str, Any]:
        infos: dict[str, Any] = {
            "x": self._x,
            "score": self._score,
            "lives": self._lives,
            "level_hi": self._level_hi,
            "level_lo": self._level_lo,
        }
        if start_ids is not None:
            infos["start_id"] = np.asarray(start_ids, dtype=object)
        return infos

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        options = dict(options or {})
        mask = np.asarray(options.get("reset_mask", np.ones(self.num_envs, dtype=bool)), dtype=bool)
        catalog = ("Level1-1", "Level1-2")
        starts = tuple(options.get("start_ids", (None,) * self.num_envs))
        start_indices = np.asarray(
            [catalog.index(start) if start in catalog else -1 for start in starts],
            dtype=np.int32,
        )
        self.reset_calls.append(
            {
                "mask": mask.copy(),
                "seed": None if seed is None else list(seed) if not isinstance(seed, int) else seed,
                "start_ids": starts,
                "start_indices": start_indices.copy(),
            }
        )
        self._observations["image"][mask] = 0
        self._observations["aux"][mask] = 0
        self._x[mask] = 0
        self._score[mask] = 0
        self._lives[mask] = 3
        self._level_hi[mask] = 1
        self._level_lo[mask] = 1
        return self._observations, self._infos(starts)

    def step(self, actions: Any):
        self.step_actions.append(np.asarray(actions).copy())
        values = self._queued_steps.pop(0) if self._queued_steps else {}
        image = values.get(
            "image",
            self._observations["image"].astype(np.int64) + np.asarray([[1, 1]]),
        )
        self._observations["image"][:] = np.asarray(image, dtype=np.uint8)
        self._observations["aux"][:, 0] += 1
        for name, target in (
            ("x", self._x),
            ("score", self._score),
            ("lives", self._lives),
            ("level_hi", self._level_hi),
            ("level_lo", self._level_lo),
        ):
            if name in values:
                target[:] = np.asarray(values[name], dtype=target.dtype)
        rewards = np.asarray(values.get("rewards", [1.0] * self.num_envs), dtype=np.float32)
        terminated = np.asarray(values.get("terminated", [False] * self.num_envs), dtype=bool)
        truncated = np.asarray(values.get("truncated", [False] * self.num_envs), dtype=bool)
        return self._observations, rewards, terminated, truncated, self._infos()

    def render(self):
        return np.zeros((self.num_envs, 4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def descriptor_for(provider: DeterministicNativeVectorProvider) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="fake-native",
        native_observation_space=provider.single_observation_space,
        native_action_space=provider.single_action_space,
        signal_schema={
            name: SignalSpec(name, np.int64)
            for name in ("x", "score", "lives", "level_hi", "level_lo")
        },
        start_catalog=("Level1-1", "Level1-2"),
        render_support=("rgb_array",),
    )


class ProviderContractTests(unittest.TestCase):
    def test_descriptor_rejects_autoreset_and_missing_mario_signals(self):
        provider = DeterministicNativeVectorProvider()
        with self.assertRaisesRegex(ValueError, "disabled provider autoreset"):
            ProviderDescriptor(
                provider_id="bad",
                native_observation_space=provider.single_observation_space,
                native_action_space=provider.single_action_space,
                autoreset_mode="same_step",
            )

        descriptor = ProviderDescriptor(
            provider_id="missing",
            native_observation_space=provider.single_observation_space,
            native_action_space=provider.single_action_space,
            signal_schema={"x": SignalSpec("x")},
        )
        with self.assertRaisesRegex(ValueError, "does not expose task signals"):
            MarioTaskDefinition(
                MarioTaskConfig(x="x", score="score", lives="lives", level=("hi", "lo"))
            ).bind(descriptor, provider.num_envs)

    def test_descriptor_validates_start_selection_contract(self):
        provider = DeterministicNativeVectorProvider()
        with self.assertRaisesRegex(ValueError, "probabilities must match"):
            ProviderDescriptor(
                provider_id="bad-starts",
                native_observation_space=provider.single_observation_space,
                native_action_space=provider.single_action_space,
                start_catalog=("Level1-1", "Level1-2"),
                start_probabilities=(1.0,),
            )

    def test_identity_kernel_normalizes_channel_last_images(self):
        native_observation_space = gym.spaces.Box(0, 255, shape=(12, 16, 3), dtype=np.uint8)
        descriptor = ProviderDescriptor(
            provider_id="image-native",
            native_observation_space=native_observation_space,
            native_action_space=gym.spaces.Discrete(2),
        )
        kernel = IdentityTaskDefinition().bind(descriptor, 2)
        observations = np.zeros((2, 12, 16, 3), dtype=np.uint8)

        self.assertEqual(kernel.observation_space.shape, (3, 12, 16))
        self.assertEqual(kernel.encode_observations(observations).shape, (2, 3, 12, 16))

        masked = IdentityTaskDefinition(
            observation_mask=(6, 0, 0, 0),
            observation_mask_fill=7,
            observation_source_shape=(12, 16),
        ).bind(descriptor, 2)
        encoded = masked.encode_observations(observations)
        np.testing.assert_array_equal(encoded[:, :, :6, :], 7)
        np.testing.assert_array_equal(encoded[:, :, 6:, :], 0)

    def test_identity_kernel_requires_action_codec_for_structured_actions(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = ProviderDescriptor(
            provider_id="structured-actions",
            native_observation_space=provider.single_observation_space,
            native_action_space=gym.spaces.Dict({"turn": gym.spaces.Discrete(3)}),
        )
        with self.assertRaisesRegex(ValueError, "configure a task action codec"):
            IdentityTaskDefinition().bind(descriptor, provider.num_envs)
        with self.assertRaisesRegex(ValueError, "absent from the start catalog"):
            ProviderDescriptor(
                provider_id="bad-lanes",
                native_observation_space=provider.single_observation_space,
                native_action_space=provider.single_action_space,
                start_catalog=("Level1-1",),
                lane_start_ids=("Level1-2",),
            )

    def test_identity_kernel_maps_discrete_actions_into_a_native_action_tree(self):
        provider = DeterministicNativeVectorProvider()
        native_action_space = gym.spaces.Dict(
            {
                "attack": gym.spaces.MultiBinary(2),
                "turn": gym.spaces.Discrete(3),
            }
        )
        descriptor = ProviderDescriptor(
            provider_id="structured-actions",
            native_observation_space=provider.single_observation_space,
            native_action_space=native_action_space,
        )
        kernel = IdentityTaskDefinition(
            action_values=(
                {"attack": np.asarray([0, 0], dtype=np.int8), "turn": 0},
                {"attack": np.asarray([1, 0], dtype=np.int8), "turn": 2},
            )
        ).bind(descriptor, provider.num_envs)

        self.assertEqual(kernel.action_space, gym.spaces.Discrete(2))
        first = kernel.map_actions(np.asarray([1, 0]))
        np.testing.assert_array_equal(first["attack"], [[1, 0], [0, 0]])
        np.testing.assert_array_equal(first["turn"], [2, 0])
        attack_buffer = first["attack"]
        second = kernel.map_actions(np.asarray([0, 1]))
        self.assertIs(second["attack"], attack_buffer)
        np.testing.assert_array_equal(second["attack"], [[0, 0], [1, 0]])

        with self.assertRaisesRegex(ValueError, "outside native action space"):
            IdentityTaskDefinition(
                action_values=({"attack": [0, 0], "turn": 7},)
            ).bind(descriptor, provider.num_envs)

    def test_identity_auto_action_runs_after_reset_and_signal_decrease(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = descriptor_for(provider)
        kernel = IdentityTaskDefinition(
            action_values=([0, 0, 0], [1, 0, 0]),
            auto_action_id=1,
            auto_action_repeat_steps=2,
            auto_action_signal="lives",
        ).bind(descriptor, provider.num_envs)
        runtime = BatchRuntime(provider, descriptor, kernel, run_seed=17)
        policy_noop = np.zeros(provider.num_envs, dtype=np.int64)

        runtime.reset()
        runtime.step(policy_noop)
        runtime.step(policy_noop)
        runtime.step(policy_noop)
        np.testing.assert_array_equal(provider.step_actions[0], [[1, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(provider.step_actions[1], [[1, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(provider.step_actions[2], [[0, 0, 0], [0, 0, 0]])

        provider.queue_step(lives=[2, 3])
        life_loss_step = runtime.step(policy_noop)
        next_step = runtime.step(policy_noop)

        self.assertFalse(np.any(life_loss_step.dones))
        self.assertFalse(np.any(next_step.dones))
        np.testing.assert_array_equal(provider.step_actions[-1], [[1, 0, 0], [0, 0, 0]])


class BatchRuntimeTests(unittest.TestCase):
    def make_identity_runtime(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = descriptor_for(provider)
        kernel = IdentityTaskDefinition().bind(descriptor, provider.num_envs)
        return provider, BatchRuntime(provider, descriptor, kernel, run_seed=17)

    def test_provider_done_snapshots_terminal_observation_and_masked_resets_once(self):
        provider, runtime = self.make_identity_runtime()
        initial = runtime.reset()
        initial_snapshot = {key: value.copy() for key, value in initial.items()}
        provider.queue_step(
            image=[[9, 9], [4, 4]],
            rewards=[2.5, 1.0],
            terminated=[True, False],
            truncated=[True, False],
        )

        step = runtime.step(np.zeros((2, 3), dtype=np.int8))

        self.assertTrue(all(isinstance(seed, int) for seed in provider.reset_calls[0]["seed"]))
        np.testing.assert_array_equal(initial["image"], initial_snapshot["image"])
        np.testing.assert_array_equal(step.infos[0]["terminal_observation"]["image"], [9, 9])
        np.testing.assert_array_equal(step.observations["image"][0], [0, 0])
        np.testing.assert_array_equal(step.observations["image"][1], [4, 4])
        np.testing.assert_array_equal(provider.reset_calls[-1]["mask"], [True, False])
        self.assertTrue(step.terminated[0])
        self.assertFalse(step.truncated[0], "termination must win over truncation")
        self.assertFalse(step.infos[0]["TimeLimit.truncated"])
        self.assertEqual(len(provider.reset_calls), 2)

        records = runtime.drain_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].lane, 0)
        self.assertEqual(records[0].episode_return, 2.5)
        self.assertEqual(records[0].episode_length, 1)
        self.assertEqual(records[0].outcome, Outcome.NEUTRAL)
        self.assertEqual(runtime.drain_records(), [])

    def test_double_buffers_protect_the_observation_sb3_is_still_using(self):
        provider, runtime = self.make_identity_runtime()
        first = runtime.reset()
        provider.queue_step(image=[[1, 2], [3, 4]])
        second = runtime.step(np.zeros((2, 3), dtype=np.int8)).observations
        second_snapshot = {key: value.copy() for key, value in second.items()}

        np.testing.assert_array_equal(first["image"], np.zeros((2, 2), dtype=np.uint8))
        provider.queue_step(image=[[5, 6], [7, 8]])
        third = runtime.step(np.zeros((2, 3), dtype=np.int8)).observations

        np.testing.assert_array_equal(second["image"], second_snapshot["image"])
        np.testing.assert_array_equal(third["image"], [[5, 6], [7, 8]])
        self.assertIsNot(first["image"], second["image"])

    def test_masked_reset_preserves_unselected_lane_state_and_rng_request(self):
        provider, runtime = self.make_identity_runtime()
        runtime.reset()
        provider.queue_step(
            image=[[8, 8], [6, 6]],
            terminated=[True, False],
            x=[20, 30],
        )

        step = runtime.step(np.zeros((2, 3), dtype=np.int8))

        self.assertEqual(provider._x[1], 30)
        np.testing.assert_array_equal(step.observations["image"][1], [6, 6])
        seeds = provider.reset_calls[-1]["seed"]
        self.assertIsInstance(seeds[0], int)
        self.assertIsNone(seeds[1])
        self.assertEqual(runtime._episode_indices.tolist(), [1, 0])

    def test_fixed_lane_starts_remain_fixed_across_masked_resets(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = ProviderDescriptor(
            provider_id="fake-native",
            native_observation_space=provider.single_observation_space,
            native_action_space=provider.single_action_space,
            start_catalog=("Level1-1", "Level1-2"),
            lane_start_ids=("Level1-2", "Level1-1"),
        )
        runtime = BatchRuntime(
            provider,
            descriptor,
            IdentityTaskDefinition().bind(descriptor, provider.num_envs),
            run_seed=17,
        )
        runtime.reset()
        provider.queue_step(terminated=[True, False])
        runtime.step(np.zeros((2, 3), dtype=np.int8))

        self.assertEqual(provider.reset_calls[0]["start_ids"], ("Level1-2", "Level1-1"))
        self.assertEqual(provider.reset_calls[1]["start_ids"][0], "Level1-2")

    def test_identity_task_timeout_is_kernel_derived(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = descriptor_for(provider)
        runtime = BatchRuntime(
            provider,
            descriptor,
            IdentityTaskDefinition(max_episode_steps=2).bind(descriptor, provider.num_envs),
            run_seed=17,
        )
        runtime.reset()
        provider.queue_step()
        first = runtime.step(np.zeros((2, 3), dtype=np.int8))
        provider.queue_step()
        second = runtime.step(np.zeros((2, 3), dtype=np.int8))

        self.assertFalse(np.any(first.dones))
        self.assertTrue(np.all(second.truncated))
        records = [
            record for record in runtime.drain_records() if isinstance(record, EpisodeRecord)
        ]
        self.assertTrue(all(record.outcome == Outcome.TIMEOUT for record in records))


class MarioKernelTests(unittest.TestCase):
    @staticmethod
    def make_runtime(**config_values: Any):
        provider = DeterministicNativeVectorProvider()
        descriptor = descriptor_for(provider)
        action_masks = np.asarray([[0, 0, 0], [1, 0, 1]], dtype=np.int8)
        reward_mode = config_values.pop("reward_mode", "native")
        config = MarioTaskConfig(
            x="x",
            score="score",
            lives="lives",
            level=("level_hi", "level_lo"),
            action_masks=action_masks,
            reward_mode=reward_mode,
            **config_values,
        )
        kernel = MarioTaskDefinition(config).bind(descriptor, provider.num_envs)
        return provider, kernel, BatchRuntime(provider, descriptor, kernel, run_seed=5)

    def test_life_loss_is_task_done_and_uses_reset_data_for_next_baseline(self):
        provider, _kernel, runtime = self.make_runtime()
        runtime.reset()
        provider.queue_step(
            x=[12, 4],
            lives=[2, 3],
            image=[[12, 12], [4, 4]],
        )

        step = runtime.step(np.asarray([1, 0]))

        np.testing.assert_array_equal(provider.step_actions[-1], [[1, 0, 1], [0, 0, 0]])
        self.assertTrue(step.terminated[0])
        self.assertFalse(step.terminated[1])
        records = runtime.drain_records()
        self.assertEqual(
            next(record for record in records if isinstance(record, EpisodeRecord)).outcome,
            Outcome.FAILURE,
        )
        self.assertIn(
            "life_loss",
            next(record for record in records if isinstance(record, TaskEventRecord)).events,
        )
        np.testing.assert_array_equal(provider.reset_calls[-1]["mask"], [True, False])

        provider.queue_step(x=[2, 5], lives=[3, 3])
        runtime.step(np.asarray([0, 0]))
        self.assertFalse(
            any(
                isinstance(record, (EpisodeRecord, TaskEventRecord))
                for record in runtime.drain_records()
            ),
            "reset lives must become the new baseline",
        )

    def test_clean_level_change_succeeds_but_simultaneous_death_fails(self):
        provider, _kernel, runtime = self.make_runtime()
        runtime.reset()
        provider.queue_step(
            level_lo=[2, 2],
            lives=[3, 2],
            x=[30, 30],
        )

        step = runtime.step(np.asarray([0, 0]))
        records = sorted(
            (record for record in runtime.drain_records() if isinstance(record, EpisodeRecord)),
            key=lambda record: record.lane,
        )

        self.assertTrue(np.all(step.dones))
        self.assertEqual(records[0].outcome, Outcome.SUCCESS)
        self.assertEqual(records[1].outcome, Outcome.FAILURE)
        self.assertIn("level_change", records[0].events)
        self.assertIn("life_loss", records[1].events)
        self.assertFalse(records[1].metrics["completion_event"])

    def test_no_progress_timeout_is_a_task_truncation(self):
        provider, _kernel, runtime = self.make_runtime(no_progress_timeout_steps=2)
        runtime.reset()
        provider.queue_step(x=[0, 0])
        first = runtime.step(np.asarray([0, 0]))
        provider.queue_step(x=[0, 0])
        second = runtime.step(np.asarray([0, 0]))

        self.assertFalse(np.any(first.dones))
        self.assertTrue(np.all(second.truncated))
        self.assertFalse(np.any(second.terminated))
        records = [
            record for record in runtime.drain_records() if isinstance(record, EpisodeRecord)
        ]
        self.assertTrue(all(record.outcome == Outcome.TIMEOUT for record in records))
        self.assertTrue(all("stalled" in record.events for record in records))

    def test_progress_coordinate_continues_across_level_changes(self):
        provider, _kernel, runtime = self.make_runtime(
            terminate_on_level_change=False,
            max_episode_steps=3,
        )
        runtime.reset()
        provider.queue_step(x=[100, 0])
        runtime.step(np.asarray([0, 0]))
        provider.queue_step(x=[4, 0], level_lo=[2, 1])
        runtime.step(np.asarray([0, 0]))
        provider.queue_step(x=[10, 0], level_lo=[2, 1])
        runtime.step(np.asarray([0, 0]))

        drained = runtime.drain_records()
        event_records = [record for record in drained if isinstance(record, TaskEventRecord)]
        self.assertTrue(
            any("level_change" in record.events for record in event_records),
            "non-terminal clean clears must be emitted immediately",
        )
        first_lane = next(
            record for record in drained if isinstance(record, EpisodeRecord) and record.lane == 0
        )
        self.assertEqual(first_lane.metrics["completed_level_base"], 100)
        self.assertEqual(first_lane.metrics["global_x_pos"], 110)
        self.assertEqual(first_lane.metrics["global_max_x_pos"], 110)
        self.assertEqual(first_lane.metrics["progress_delta"], 10)

    def test_reward_component_batches_avoid_step_info_materialization(self):
        provider, _kernel, runtime = self.make_runtime(
            reward_mode="score",
            use_native_reward=False,
            progress_reward_scale=2.0,
            time_penalty=0.5,
        )
        runtime.reset()
        provider.queue_step(x=[5, 2], score=[100, 40], rewards=[9.0, 9.0])
        runtime.step(np.asarray([0, 0]))

        metric_record = next(
            record for record in runtime.drain_records() if isinstance(record, BatchMetricRecord)
        )
        np.testing.assert_array_equal(
            metric_record.metrics["progress_reward_component"], [10.0, 4.0]
        )
        np.testing.assert_allclose(metric_record.metrics["score_reward_component"], [1.0, 0.4])
        np.testing.assert_array_equal(metric_record.metrics["time_penalty_component"], [-0.5, -0.5])
        np.testing.assert_allclose(metric_record.metrics["shaped_reward"], [10.5, 3.9])

    def test_canonical_task_softcodes_signal_bindings_and_stall_outcome(self):
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            task={
                "id": "mario",
                "action": {"set": "right"},
                "signals": {
                    "x": "custom_x",
                    "score": "custom_score",
                    "lives": "custom_lives",
                    "level": ["world", "stage"],
                },
                "events": {
                    "stalled": {
                        "signal": "x",
                        "operation": "unchanged_for",
                        "steps": 17,
                    }
                },
                "termination": {"failure": ["stalled"], "success": []},
                "reward": {"reward_mode": "score", "progress_reward_scale": 3.0},
            },
        )

        compiled = MarioTaskConfig.from_env_config(config)

        self.assertEqual(compiled.x, "custom_x")
        self.assertEqual(compiled.level, ("world", "stage"))
        self.assertEqual(compiled.no_progress_timeout_steps, 17)
        self.assertTrue(compiled.stall_is_failure)
        self.assertFalse(compiled.terminate_on_life_loss)
        self.assertFalse(compiled.terminate_on_level_change)
        self.assertEqual(compiled.progress_reward_scale, 3.0)
        self.assertEqual(compiled.action_masks.shape[0], 4)


class RlabVecEnvTests(unittest.TestCase):
    def test_sb3_facade_returns_same_step_reset_observation_and_drains_records(self):
        provider = DeterministicNativeVectorProvider()
        descriptor = descriptor_for(provider)
        runtime = BatchRuntime(
            provider,
            descriptor,
            IdentityTaskDefinition().bind(descriptor, provider.num_envs),
            run_seed=11,
        )
        env = RlabVecEnv(runtime)
        env.seed(100)
        observations = env.reset()
        np.testing.assert_array_equal(observations["image"], np.zeros((2, 2)))
        provider.queue_step(image=[[7, 7], [8, 8]], terminated=[False, True])

        next_observations, rewards, dones, infos = env.step(np.zeros((2, 3), dtype=np.int8))

        np.testing.assert_array_equal(next_observations["image"], [[7, 7], [0, 0]])
        np.testing.assert_array_equal(rewards, [1.0, 1.0])
        np.testing.assert_array_equal(dones, [False, True])
        np.testing.assert_array_equal(infos[1]["terminal_observation"]["image"], [8, 8])
        self.assertEqual(provider.reset_calls[0]["seed"], [100, 101])
        self.assertEqual(len(env.drain_records()), 1)
        self.assertEqual(env.drain_records(), [])
        self.assertEqual(len(env.get_images()), 2)
        env.close()
        self.assertTrue(provider.closed)


if __name__ == "__main__":
    unittest.main()
