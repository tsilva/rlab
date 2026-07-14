from __future__ import annotations

import unittest

from stable_baselines3.common.callbacks import BaseCallback

from rlab.callbacks import CallbackHelper, RlabCallback
from rlab.schedules import EntropyCoefficientScheduleHelper
from rlab.training.sb3_helpers import GracefulStopHelper
from rlab.training.sb3_ppo import checkpoint_save_frequency
from rlab.training_backend import GracefulStopFlag
from rlab.seeds import (
    DEFAULT_EVAL_SEED,
    validate_eval_seed,
)


class TrainTests(unittest.TestCase):
    def test_only_rlab_callback_implements_the_sb3_callback_protocol(self) -> None:
        self.assertTrue(issubclass(RlabCallback, BaseCallback))
        self.assertTrue(issubclass(GracefulStopHelper, CallbackHelper))
        self.assertFalse(issubclass(GracefulStopHelper, BaseCallback))

    def test_rlab_callback_drives_entropy_schedule_component(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: dict[str, float] = {}

            def record(self, key: str, value: float) -> None:
                self.records[key] = value

        class Model:
            def __init__(self) -> None:
                self.ent_coef = 0.0
                self.logger = Logger()

        model = Model()
        callback = RlabCallback(
            [EntropyCoefficientScheduleHelper(0.1, 0.0, schedule_timesteps=100)]
        )
        callback.model = model  # type: ignore[assignment]
        callback._on_training_start()
        self.assertEqual(model.ent_coef, 0.1)

        callback.num_timesteps = 50
        self.assertTrue(callback._on_step())
        self.assertAlmostEqual(model.ent_coef, 0.05)
        self.assertAlmostEqual(
            model.logger.records["train/algorithm/ppo/hyperparameter/entropy_coefficient"],
            0.05,
        )

    def test_checkpoint_save_frequency_disables_zero_or_negative(self) -> None:
        self.assertIsNone(checkpoint_save_frequency(0, 2))
        self.assertIsNone(checkpoint_save_frequency(-1, 2))

    def test_checkpoint_save_frequency_scales_by_vec_envs(self) -> None:
        self.assertEqual(checkpoint_save_frequency(500_000, 2), 250_000)
        self.assertEqual(checkpoint_save_frequency(1, 32), 1)

    def test_eval_seed_rejects_training_range(self) -> None:
        self.assertEqual(DEFAULT_EVAL_SEED, 10000)
        self.assertEqual(validate_eval_seed(10000), 10000)
        with self.assertRaisesRegex(ValueError, "reserved for training"):
            validate_eval_seed(9999)

    def test_graceful_stop_callback_stops_after_flag_request(self) -> None:
        stop_flag = GracefulStopFlag()
        callback = GracefulStopHelper(stop_flag)
        callback.num_timesteps = 123

        self.assertTrue(callback._on_step())

        stop_flag.request("SIGUSR1")

        self.assertFalse(callback._on_step())


if __name__ == "__main__":
    unittest.main()
