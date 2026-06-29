from __future__ import annotations

import unittest

from rlab.train import GracefulStopCallback, GracefulStopFlag, checkpoint_save_frequency
from rlab.seeds import eval_seed_for_training_seed, eval_seeds, training_seeds, validate_eval_seed


class TrainTests(unittest.TestCase):
    def test_checkpoint_save_frequency_disables_zero_or_negative(self) -> None:
        self.assertIsNone(checkpoint_save_frequency(0, 2))
        self.assertIsNone(checkpoint_save_frequency(-1, 2))

    def test_checkpoint_save_frequency_scales_by_vec_envs(self) -> None:
        self.assertEqual(checkpoint_save_frequency(500_000, 2), 250_000)
        self.assertEqual(checkpoint_save_frequency(1, 32), 1)

    def test_in_training_eval_seed_uses_reserved_eval_range(self) -> None:
        self.assertEqual(eval_seed_for_training_seed(123), 10123)

    def test_eval_seed_rejects_training_range(self) -> None:
        self.assertEqual(validate_eval_seed(10000), 10000)
        with self.assertRaisesRegex(ValueError, "reserved for training"):
            validate_eval_seed(9999)

    def test_hardcoded_seed_sequences_do_not_overlap(self) -> None:
        self.assertEqual(training_seeds(5), [1, 2, 3, 4, 5])
        self.assertEqual(eval_seeds(5), [10001, 10002, 10003, 10004, 10005])

    def test_hardcoded_seed_sequence_counts_must_be_valid(self) -> None:
        with self.assertRaisesRegex(ValueError, "training seed count"):
            training_seeds(-1)
        with self.assertRaisesRegex(ValueError, "eval seed count"):
            eval_seeds("10")

    def test_graceful_stop_callback_stops_after_flag_request(self) -> None:
        stop_flag = GracefulStopFlag()
        callback = GracefulStopCallback(stop_flag)
        callback.num_timesteps = 123

        self.assertTrue(callback._on_step())

        stop_flag.request("SIGUSR1")

        self.assertFalse(callback._on_step())


if __name__ == "__main__":
    unittest.main()
