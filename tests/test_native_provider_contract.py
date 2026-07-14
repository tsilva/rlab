from __future__ import annotations

import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import stable_retro as retro

from rlab.env import EnvConfig, _bound_task_kernel
from rlab.env_providers import (
    _AleManualResetAdapter,
    _StartInfoAdapter,
    make_provider_vec_env,
    provider_descriptor,
    provider_native_vec_kwargs,
)
from rlab.task_kernels import MarioTaskConfig, MarioTaskDefinition
from packaging.version import Version


class RegisteredNativeVectorEnv(gym.vector.VectorEnv):
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

    def __init__(self, num_envs: int, autoreset_mode, **kwargs):
        del kwargs
        if autoreset_mode is not gym.vector.AutoresetMode.DISABLED:
            raise ValueError("manual autoreset is required")
        self.num_envs = int(num_envs)
        self.autoreset_mode = autoreset_mode
        self.single_observation_space = gym.spaces.Box(-100, 100, shape=(2,), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self._observations = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.reset_masks: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = np.asarray(
            options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
            dtype=np.bool_,
        )
        self.reset_masks.append(mask.copy())
        self._observations[mask] = 0
        return self._observations, {}

    def step(self, actions):
        del actions
        self._observations += 1
        return (
            self._observations,
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )

    def close(self):
        return None


class GenericNativeProviderTests(unittest.TestCase):
    env_id = "RlabRegisteredNativeVector-v0"
    scalar_env_id = "RlabScalarOnly-v0"

    @classmethod
    def setUpClass(cls) -> None:
        gym.register(
            cls.env_id,
            entry_point=lambda: None,
            vector_entry_point=RegisteredNativeVectorEnv,
        )
        gym.register(cls.scalar_env_id, entry_point=lambda: None)

    @classmethod
    def tearDownClass(cls) -> None:
        gym.registry.pop(cls.env_id, None)
        gym.registry.pop(cls.scalar_env_id, None)

    def test_uses_only_registered_native_vector_entry_points(self) -> None:
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=3,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(config, native_kwargs=kwargs)

        self.assertEqual(env.num_envs, 3)
        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        mask = np.asarray([True, False, True], dtype=np.bool_)
        env.reset(seed=[1, None, 3], options={"reset_mask": mask})
        np.testing.assert_array_equal(env.reset_masks[-1], mask)
        env.close()

    def test_rejects_synthesized_vectorization(self) -> None:
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.scalar_env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        with self.assertRaisesRegex(RuntimeError, "no native Gymnasium vector entry point"):
            make_provider_vec_env(config, native_kwargs={"num_envs": 2})

    def test_descriptor_discovers_step_only_configured_signal(self) -> None:
        class StepSignalEnv(RegisteredNativeVectorEnv):
            def step(self, actions):
                observations, rewards, terminated, truncated, _infos = super().step(actions)
                return observations, rewards, terminated, truncated, {
                    "ball_y": np.arange(self.num_envs, dtype=np.int64)
                }

        env = StepSignalEnv(2, gym.vector.AutoresetMode.DISABLED)
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {"ball_y": "ball_y"},
                "events": {
                    "serve_stall": {
                        "signal": "ball_y",
                        "operation": "equals_for",
                        "value": 0,
                        "steps": 3,
                    }
                },
                "termination": {"failure": ["serve_stall"]},
                "reward": {"reward_mode": "native"},
            },
        )

        descriptor = provider_descriptor(config, env, state_weight_mapping=lambda _config: {})

        self.assertIn("ball_y", descriptor.signal_schema)
        self.assertFalse(descriptor.signal_schema["ball_y"].available_on_reset)
        self.assertTrue(descriptor.signal_schema["ball_y"].available_on_step)

    def test_descriptor_does_not_trust_safe_view_from_generic_provider(self) -> None:
        env = RegisteredNativeVectorEnv(2, gym.vector.AutoresetMode.DISABLED)
        env.obs_copy = "safe_view"
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )

        descriptor = provider_descriptor(config, env, state_weight_mapping=lambda _config: {})

        self.assertEqual(descriptor.observation_buffer_depth, 1)

class MarioNativeProviderTests(unittest.TestCase):
    @staticmethod
    def config(**updates):
        values = {
            "env_provider": "supermariobrosnes-turbo",
            "game": "SuperMarioBros-Nes-v0",
            "state": "Level1-1",
            "task": {
                "id": "mario",
                "action": {"set": "simple"},
                "signals": {
                    "x": ["xscrollHi", "xscrollLo"],
                    "score": "score",
                    "lives": "lives",
                    "level": ["levelHi", "levelLo"],
                },
                "events": {
                    "life_loss": {"signal": "lives", "operation": "decrease"},
                    "level_change": {"signal": "level", "operation": "change"},
                },
                "termination": {"failure": ["life_loss"], "success": ["level_change"]},
                "reward": {"reward_mode": "native"},
            },
        }
        values.update(updates)
        return EnvConfig(**values)

    def test_runtime_minimum_contains_masked_reset_release(self) -> None:
        installed = Version(importlib.metadata.version("supermariobrosnes-turbo"))
        self.assertEqual(installed, Version("0.2.25"))
        self.assertEqual(Version(retro.__version__), Version("1.0.1.post29"))

    def test_constructs_with_disabled_autoreset_and_describes_starts_and_signals(self) -> None:
        class FakeMarioVectorEnv:
            metadata = {
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
                "render_modes": ("rgb_array",),
            }

            def __init__(self, game, *, num_envs, autoreset_mode, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = autoreset_mode
                self.kwargs = kwargs
                self.single_observation_space = gym.spaces.Box(
                    0, 255, shape=(4, 84, 84), dtype=np.uint8
                )
                self.single_action_space = gym.spaces.MultiBinary(9)
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, num_envs
                )
                self.action_space = gym.vector.utils.batch_space(
                    self.single_action_space, num_envs
                )
                self.initial_state_names = ("Level1-1",)
                self._states = ["Level1-1" for _ in range(num_envs)]

            def reset(self, *, seed=None, options=None):
                del seed
                options = dict(options or {})
                mask = np.asarray(
                    options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
                    dtype=np.bool_,
                )
                starts = np.asarray(
                    options.get("start_indices", np.full(self.num_envs, -1, dtype=np.int32))
                )
                for lane in np.flatnonzero(mask):
                    if starts[lane] >= 0:
                        self._states[int(lane)] = self.initial_state_names[int(starts[lane])]
                infos = {
                    "xscrollHi": np.zeros(self.num_envs, dtype=np.int64),
                    "xscrollLo": np.zeros(self.num_envs, dtype=np.int64),
                    "score": np.zeros(self.num_envs, dtype=np.int64),
                    "lives": np.full(self.num_envs, 3, dtype=np.int64),
                    "levelHi": np.zeros(self.num_envs, dtype=np.int64),
                    "levelLo": np.zeros(self.num_envs, dtype=np.int64),
                }
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8), infos

            def active_states(self):
                return tuple(self._states)

        config = self.config()
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            super_mario_vec_env_type=lambda: FakeMarioVectorEnv,
        )
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )

        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertNotIn("done_on", env.kwargs)
        self.assertEqual(env.kwargs["info_filter"]["mode"], "all")
        self.assertEqual(descriptor.start_catalog, ("Level1-1",))
        self.assertEqual(descriptor.lane_start_ids, ("Level1-1", "Level1-1"))
        self.assertEqual(descriptor.render_support, ("rgb_array",))
        self.assertEqual(descriptor.observation_buffer_depth, 2)
        self.assertEqual(descriptor.signal_schema["lives"].dtype, np.dtype(np.int64))
        _observations, reset_infos = env.reset(
            seed=[1, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )
        np.testing.assert_array_equal(reset_infos["_start_id"], [True, False])
        self.assertEqual(reset_infos["start_id"].tolist(), ["Level1-1", "Level1-1"])

    def test_mario_provider_prefers_the_imported_stable_retro_rom(self) -> None:
        class FakeMarioVectorEnv:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, *, num_envs, autoreset_mode, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = autoreset_mode
                self.kwargs = kwargs

        config = self.config()
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("stable_retro.data.get_romfile_path") as get_romfile_path,
        ):
            rom_path = Path(temporary) / "rom.nes"
            rom_path.write_bytes(b"rom")
            get_romfile_path.return_value = str(rom_path)
            env = make_provider_vec_env(
                config,
                native_kwargs=kwargs,
                super_mario_vec_env_type=lambda: FakeMarioVectorEnv,
            )

        self.assertEqual(env.kwargs["rom_path"], str(rom_path))

    def test_rejects_native_task_detectors(self) -> None:
        config = self.config(env_args={"done_on": {"life_loss": ("lives", "decrease")}})
        with self.assertRaisesRegex(ValueError, "provider task detectors are unsupported"):
            provider_native_vec_kwargs(
                config,
                n_envs=2,
                native_obs_crop=lambda _config: None,
                state_weight_mapping=lambda _config: {},
            )

    def test_descriptor_does_not_invent_requested_signals(self) -> None:
        class Native:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
            single_observation_space = gym.spaces.Box(
                0, 255, shape=(4, 84, 84), dtype=np.uint8
            )
            single_action_space = gym.spaces.MultiBinary(9)
            observation_space = gym.vector.utils.batch_space(single_observation_space, 2)
            action_space = gym.vector.utils.batch_space(single_action_space, 2)
            initial_state_names = ("Level1-1",)

            def reset(self, *, seed=None, options=None):
                del seed, options
                infos = {
                    name: np.zeros(2, dtype=np.int64)
                    for name in ("score", "lives", "levelHi", "levelLo")
                }
                return np.zeros((2, 4, 84, 84), dtype=np.uint8), infos

        config = self.config()
        config = EnvConfig(
            **{
                **config.__dict__,
                "task": {
                    **config.task,
                    "signals": {**config.task["signals"], "x": "missing_x"},
                },
            }
        )
        descriptor = provider_descriptor(
            config,
            Native(),
            state_weight_mapping=lambda _config: {},
        )

        self.assertNotIn("missing_x", descriptor.signal_schema)
        with self.assertRaisesRegex(ValueError, "does not expose task signals"):
            MarioTaskDefinition(MarioTaskConfig.from_env_config(config)).bind(descriptor, 2)

    def test_start_adapter_renders_every_native_lane(self) -> None:
        class NativeScreens:
            def get_screen(self, lane):
                return np.full((3, 4, 3), lane, dtype=np.uint8)

        env = type("Env", (), {"num_envs": 2, "native": NativeScreens()})()
        frames = _StartInfoAdapter(env).get_images()

        self.assertEqual([frame.shape for frame in frames], [(3, 4, 3), (3, 4, 3)])
        self.assertEqual(int(frames[1][0, 0, 0]), 1)

    def test_rejects_provider_without_disabled_autoreset(self) -> None:
        class OldMarioVectorEnv:
            metadata = {}

            def __init__(self, game, *, num_envs, **kwargs):
                del game, kwargs
                self.num_envs = num_envs

        with self.assertRaisesRegex(RuntimeError, "does not advertise disabled autoreset"):
            make_provider_vec_env(
                self.config(),
                native_kwargs={"num_envs": 2},
                super_mario_vec_env_type=lambda: OldMarioVectorEnv,
            )

    def test_stable_retro_release_without_manual_lifecycle_is_rejected(self) -> None:
        class OldRetroVectorEnv:
            metadata = {}

            def __init__(self, game, **kwargs):
                del game, kwargs

        config = self.config(env_provider="stable-retro-turbo")
        with self.assertRaisesRegex(RuntimeError, "does not advertise disabled autoreset"):
            make_provider_vec_env(
                config,
                native_kwargs={"num_envs": 2},
                retro_vec_env_type=OldRetroVectorEnv,
            )

    def test_stable_retro_constructs_with_disabled_autoreset(self) -> None:
        class ManualRetroVectorEnv:
            metadata = {
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
                "render_modes": ("rgb_array",),
            }

            def __init__(self, game, *, num_envs, autoreset_mode, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = autoreset_mode
                self.kwargs = kwargs

        config = self.config(env_provider="stable-retro-turbo")
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: (32, 0, 0, 0),
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            retro_vec_env_type=ManualRetroVectorEnv,
        )

        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertEqual(env.kwargs["obs_crop_mode"], "remove")
        self.assertEqual(env.kwargs["obs_crop_fill"], 0)
        self.assertNotIn("done_on", env.kwargs)

    def test_stable_retro_atari_uses_retro_vec_env_contract(self) -> None:
        class ManualRetroVectorEnv:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, *, num_envs, autoreset_mode, **kwargs):
                if autoreset_mode is not gym.vector.AutoresetMode.DISABLED:
                    raise ValueError("RetroVecEnv requires disabled autoreset")
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = autoreset_mode
                self.kwargs = kwargs
                self.reset_calls = 0
                self.reset_masks = []
                self.observations = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)
                self.single_observation_space = gym.spaces.Box(
                    0, 255, shape=(4, 84, 84), dtype=np.uint8
                )
                self.single_action_space = gym.spaces.MultiBinary(8)
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, num_envs
                )
                self.action_space = gym.vector.utils.batch_space(
                    self.single_action_space, num_envs
                )

            def reset(self, *, seed=None, options=None):
                del seed
                self.reset_calls += 1
                options = dict(options or {})
                mask = np.asarray(
                    options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
                    dtype=np.bool_,
                )
                self.reset_masks.append(mask.copy())
                self.observations[mask] = 0
                return self.observations.copy(), {}

            def step(self, actions):
                del actions
                self.observations.fill(2)
                self.observations[0] = 9
                return (
                    self.observations.copy(),
                    np.zeros(self.num_envs, dtype=np.float32),
                    np.asarray([True] + [False] * (self.num_envs - 1)),
                    np.zeros(self.num_envs, dtype=np.bool_),
                    {},
                )

            def close(self):
                return None

        config = EnvConfig(
            env_provider="stable-retro-turbo",
            game="Breakout-Atari2600-v0",
            state="Start",
            obs_crop=(17, 0, 0, 0),
            obs_crop_mode="mask",
            sticky_action_prob=0.25,
            env_args={"info_filter": "all", "num_threads": 8, "reward_clip": True},
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {"max_episode_steps": 54_000},
                "reward": {"reward_mode": "native"},
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            retro_vec_env_type=ManualRetroVectorEnv,
        )

        self.assertEqual(env.game, "Breakout-Atari2600-v0")
        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertEqual(env.kwargs["info_filter"], "all")
        self.assertEqual(env.kwargs["num_threads"], 8)
        self.assertEqual(env.kwargs["state"], "Start")
        self.assertEqual(env.kwargs["obs_resize"], (84, 84))
        self.assertEqual(env.kwargs["obs_crop"], (17, 0, 0, 0))
        self.assertEqual(env.kwargs["obs_crop_mode"], "mask")
        self.assertEqual(env.kwargs["obs_layout"], "chw")
        self.assertEqual(env.kwargs["obs_copy"], "safe_view")
        self.assertEqual(env.kwargs["sticky_action_prob"], 0.25)
        self.assertIs(env.kwargs["use_fire_reset"], False)
        self.assertNotIn("max_episode_steps", env.kwargs)
        env.reset(seed=123)
        observations, _rewards, terminated, _truncated, infos = env.step(
            np.zeros((16, 8), dtype=np.int8)
        )
        self.assertTrue(terminated[0])
        self.assertEqual(int(observations[0, 0, 0, 0]), 9)
        self.assertEqual(int(observations[1, 0, 0, 0]), 2)
        self.assertEqual(infos, {})
        reset_observations, _reset_infos = env.reset(
            seed=[124] + [None] * 15,
            options={"reset_mask": np.asarray([True] + [False] * 15)},
        )
        self.assertEqual(int(reset_observations[0, 0, 0, 0]), 0)
        self.assertEqual(int(reset_observations[1, 0, 0, 0]), 2)
        self.assertEqual(env.reset_calls, 2)
        np.testing.assert_array_equal(env.reset_masks[-1], [True] + [False] * 15)

        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )
        kernel = _bound_task_kernel(config, descriptor, 16)
        self.assertIsNone(kernel._observation_mask)
        self.assertEqual(descriptor.observation_buffer_depth, 2)
        self.assertTrue(kernel.observation_encoding_is_view)


class AleManualLifecycleTests(unittest.TestCase):
    def test_next_step_engine_cannot_autoreset_behind_runtime(self) -> None:
        class FakeAle:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

            def __init__(self):
                self.steps = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                return np.zeros((2, 1), dtype=np.uint8), {}

            def step(self, actions):
                del actions
                self.steps += 1
                return (
                    np.zeros((2, 1), dtype=np.uint8),
                    np.zeros(2, dtype=np.float32),
                    np.asarray([self.steps == 1, False]),
                    np.zeros(2, dtype=np.bool_),
                    {},
                )

            def close(self):
                return None

        env = _AleManualResetAdapter(FakeAle())
        env.reset(seed=[1, 2])
        env.step(np.zeros(2, dtype=np.int64))
        with self.assertRaisesRegex(RuntimeError, "explicitly reset"):
            env.step(np.zeros(2, dtype=np.int64))
        env.reset(
            seed=[3, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )
        env.step(np.zeros(2, dtype=np.int64))

    def test_cached_policy_frames_are_renderable_rgb(self) -> None:
        class FakeAle:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

            def reset(self, *, seed=None, options=None):
                del seed, options
                observations = np.arange(2 * 4 * 3 * 5, dtype=np.uint8).reshape(2, 4, 3, 5)
                return observations, {}

            def close(self):
                return None

        env = _AleManualResetAdapter(FakeAle())
        env.reset(options={"reset_mask": np.ones(2, dtype=np.bool_)})
        frames = env.get_images()

        self.assertEqual([frame.shape for frame in frames], [(3, 5, 3), (3, 5, 3)])
        np.testing.assert_array_equal(frames[0][..., 0], frames[0][..., 1])

if __name__ == "__main__":
    unittest.main()
