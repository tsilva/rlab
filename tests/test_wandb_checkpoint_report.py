from __future__ import annotations

import unittest

from rlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_BEST,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    LEADER_CHECKPOINT_BEST_RETURN,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
)
from scripts.create_wandb_checkpoint_leaderboard_report import (
    goal_rank,
    leader_order_spec,
)


class CheckpointReportOrderingTests(unittest.TestCase):
    def test_breakout_report_uses_the_goal_rank_and_minimizes_serve_stalls(self) -> None:
        criteria = goal_rank("alepy__breakout")

        self.assertEqual(criteria[0].metric, "eval/full/outcome/reason/serve_stall/rate")
        self.assertEqual(
            leader_order_spec(criteria),
            [
                (LEADER_CHECKPOINT_OBJECTIVE, True),
                (LEADER_CHECKPOINT_RETURN_MEAN, False),
                (LEADER_CHECKPOINT_BEST_RETURN, False),
                (LEADER_CHECKPOINT_STEP, True),
            ],
        )

    def test_mspacman_report_uses_return_best_return_then_step(self) -> None:
        criteria = goal_rank("alepy__mspacman")

        self.assertEqual(
            [criterion.metric for criterion in criteria],
            [
                EVAL_FULL_EPISODE_RETURN_MEAN,
                EVAL_FULL_EPISODE_RETURN_BEST,
                LEADER_CHECKPOINT_STEP,
            ],
        )
        self.assertEqual(
            leader_order_spec(criteria),
            [
                (LEADER_CHECKPOINT_OBJECTIVE, False),
                (LEADER_CHECKPOINT_BEST_RETURN, False),
                (LEADER_CHECKPOINT_STEP, True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
