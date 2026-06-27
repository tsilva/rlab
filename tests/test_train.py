from __future__ import annotations

import unittest

from rlab.train import GracefulStopCallback, GracefulStopFlag, checkpoint_save_frequency
from rlab.seeds import eval_seed_for_training_seed


class TrainTests(unittest.TestCase):
    def test_checkpoint_save_frequency_disables_zero_or_negative(self) -> None:
        self.assertIsNone(checkpoint_save_frequency(0, 2))
        self.assertIsNone(checkpoint_save_frequency(-1, 2))

    def test_checkpoint_save_frequency_scales_by_vec_envs(self) -> None:
        self.assertEqual(checkpoint_save_frequency(500_000, 2), 250_000)
        self.assertEqual(checkpoint_save_frequency(1, 32), 1)

    def test_in_training_eval_seed_uses_reserved_eval_range(self) -> None:
        self.assertEqual(eval_seed_for_training_seed(123), 10123)

    def test_graceful_stop_callback_stops_after_flag_request(self) -> None:
        stop_flag = GracefulStopFlag()
        callback = GracefulStopCallback(stop_flag)
        callback.num_timesteps = 123

        self.assertTrue(callback._on_step())

        stop_flag.request("SIGUSR1")

        self.assertFalse(callback._on_step())


if __name__ == "__main__":
    unittest.main()
