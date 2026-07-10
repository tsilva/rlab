from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium.vector import AutoresetMode


LEARNING_SCRIPT = Path(__file__).parents[1] / "examples/learning/train_level1_1_ppo.py"
LEARNING_SPEC = importlib.util.spec_from_file_location("train_level1_1_ppo", LEARNING_SCRIPT)
if LEARNING_SPEC is None or LEARNING_SPEC.loader is None:
    raise RuntimeError(f"could not load learning example from {LEARNING_SCRIPT}")
learning = importlib.util.module_from_spec(LEARNING_SPEC)
LEARNING_SPEC.loader.exec_module(learning)


def _columnar(values: dict[str, np.ndarray], present: np.ndarray) -> dict[str, np.ndarray]:
    infos: dict[str, np.ndarray] = {}
    for key, value in values.items():
        infos[key] = value
        infos[f"_{key}"] = present.copy()
    return infos


class FakeManualResetMarioEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.observations = np.zeros((2, *learning.OBSERVATION_SHAPE), dtype=np.uint8)
        self.reset_masks: list[np.ndarray] = []
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        mask = np.asarray(options["reset_mask"], dtype=np.bool_)
        self.reset_masks.append(mask.copy())
        self.reset_seeds.append(seed)
        self.observations[mask] = 0
        infos = _columnar(
            {
                "lives": np.asarray([3, 0], dtype=np.int64),
                "levelHi": np.zeros(2, dtype=np.int64),
                "levelLo": np.zeros(2, dtype=np.int64),
                "score": np.zeros(2, dtype=np.int64),
                "xscrollHi": np.zeros(2, dtype=np.int64),
                "xscrollLo": np.zeros(2, dtype=np.int64),
            },
            mask,
        )
        return self.observations.copy(), infos

    def step(self, actions):
        self.observations[0].fill(5)
        self.observations[1].fill(7)
        infos = _columnar(
            {
                "lives": np.asarray([2, 3], dtype=np.int64),
                "levelHi": np.zeros(2, dtype=np.int64),
                "levelLo": np.zeros(2, dtype=np.int64),
                "score": np.zeros(2, dtype=np.int64),
                "xscrollHi": np.zeros(2, dtype=np.int64),
                "xscrollLo": np.asarray([10, 20], dtype=np.int64),
            },
            np.ones(2, dtype=np.bool_),
        )
        return (
            self.observations.copy(),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
            infos,
        )


class LearningExampleTests(unittest.TestCase):
    def test_make_env_requests_disabled_autoreset_and_raw_info(self) -> None:
        constructor_call = {}

        class ConstructorSpy:
            def __init__(self, game, **kwargs):
                constructor_call["game"] = game
                constructor_call["kwargs"] = kwargs

        with patch.object(learning, "SuperMarioBrosNesTurboVecEnv", ConstructorSpy):
            env = learning.make_env()

        self.assertIsInstance(env, ConstructorSpy)
        self.assertEqual(constructor_call["game"], learning.GAME)
        kwargs = constructor_call["kwargs"]
        self.assertIs(kwargs["autoreset_mode"], AutoresetMode.DISABLED)
        self.assertEqual(kwargs["info_filter"], "all")
        self.assertEqual(kwargs["obs_layout"], "chw")
        self.assertNotIn("done_on", kwargs)

    def test_step_uses_five_value_contract_and_masked_manual_reset(self) -> None:
        env = FakeManualResetMarioEnv()
        trackers = [learning.MarioTracker(), learning.MarioTracker()]
        observations = learning.reset_lanes(
            env,
            trackers,
            np.ones(2, dtype=np.bool_),
            seed=7,
        )

        self.assertEqual(observations.shape, (2, 4, 84, 84))
        self.assertEqual(observations.dtype, np.uint8)
        next_observations, rewards, dones, completed = learning.step_and_reset(
            env,
            np.asarray([0, 1]),
            trackers,
        )

        np.testing.assert_array_equal(env.reset_masks[0], [True, True])
        np.testing.assert_array_equal(env.reset_masks[1], [True, False])
        self.assertEqual(env.reset_seeds, [7, None])
        np.testing.assert_array_equal(dones, [True, False])
        np.testing.assert_array_equal(completed, [False, False])
        np.testing.assert_allclose(rewards, [-15.0, 20.0])
        self.assertTrue(np.all(next_observations[0] == 0))
        self.assertTrue(np.all(next_observations[1] == 7))
        self.assertEqual(trackers[0].prev_lives, 3)
        self.assertEqual(trackers[1].prev_lives, 3)

    def test_lane_info_honors_presence_masks_and_nested_dicts(self) -> None:
        infos = {
            "score": np.asarray([100, 200]),
            "_score": np.asarray([True, False]),
            "done_on_info": {
                "life_loss": {
                    "op": np.asarray(["decrease", None], dtype=object),
                    "_op": np.asarray([True, False]),
                },
                "_life_loss": np.asarray([True, False]),
            },
            "_done_on_info": np.asarray([True, False]),
        }

        self.assertEqual(
            learning.lane_info(infos, 0),
            {"score": 100, "done_on_info": {"life_loss": {"op": "decrease"}}},
        )
        self.assertEqual(learning.lane_info(infos, 1), {})

    def test_tracker_accepts_only_clean_level_changes_as_completion(self) -> None:
        tracker = learning.MarioTracker()
        tracker.reset({"lives": 3, "levelHi": 0, "levelLo": 0})

        clean_transition = {"lives": 3, "levelHi": 0, "levelLo": 1}
        tracker.shape(clean_transition)
        self.assertTrue(clean_transition["level_complete"])
        self.assertFalse(clean_transition["died"])

        tracker.reset({"lives": 3, "levelHi": 0, "levelLo": 0})
        death_transition = {"lives": 2, "levelHi": 0, "levelLo": 1}
        tracker.shape(death_transition)
        self.assertFalse(death_transition["level_complete"])
        self.assertTrue(death_transition["died"])


if __name__ == "__main__":
    unittest.main()
