from __future__ import annotations

import unittest

from rlab.wandb_leaders import (
    CheckpointLeader,
    RunScore,
    rank_checkpoint_leaders,
    rank_run_leaders,
)


def run_score(
    *,
    reward_shape: str,
    reward_hash: str,
    effective_goal_hash: str,
    objective: float,
    is_default: bool,
) -> RunScore:
    return RunScore(
        goal_slug="mario-level-1-1",
        recipe_slug="ppo",
        reward_shape=reward_shape,
        reward_shape_sha256=reward_hash,
        effective_goal_contract_sha256=effective_goal_hash,
        reward_shape_is_default=is_default,
        run_id=f"run-{reward_shape}-{objective}",
        run_name=f"run-{reward_shape}",
        url="https://example.invalid/run",
        seed=1,
        objective=objective,
    )


def checkpoint_leader(
    *,
    reward_shape: str,
    reward_hash: str,
    effective_goal_hash: str,
    rank_score: tuple[float, ...],
    is_default: bool,
) -> CheckpointLeader:
    return CheckpointLeader(
        goal_slug="mario-level-1-1",
        recipe_slug="ppo",
        reward_shape=reward_shape,
        reward_shape_sha256=reward_hash,
        effective_goal_contract_sha256=effective_goal_hash,
        reward_shape_is_default=is_default,
        run_id=f"run-{reward_shape}-{rank_score[0]}",
        run_name=f"run-{reward_shape}",
        url="https://example.invalid/run",
        objective=rank_score[0],
        objective_name="success",
        success_rate_min=rank_score[0],
        success_rate_mean=rank_score[0],
        progress_max=None,
        return_mean=1.0,
        steps_to_goal=None,
        checkpoint_step=1,
        artifact_ref="model:v1",
        eval_source="acceptance",
        rank_score=rank_score,
    )


class WandbLeaderRewardShapeTests(unittest.TestCase):
    def test_run_ranking_does_not_combine_different_reward_shapes(self):
        leaders = rank_run_leaders(
            [
                run_score(
                    reward_shape="score-v1",
                    reward_hash="sha256:score",
                    effective_goal_hash="sha256:goal-score",
                    objective=0.4,
                    is_default=True,
                ),
                run_score(
                    reward_shape="score-step-0p01-v1",
                    reward_hash="sha256:step",
                    effective_goal_hash="sha256:goal-step",
                    objective=0.9,
                    is_default=False,
                ),
            ]
        )

        self.assertEqual(len(leaders), 2)
        self.assertEqual(leaders[0].reward_shape, "score-v1")
        self.assertEqual({leader.seeds for leader in leaders}, {1})

    def test_checkpoint_ranking_is_scoped_to_each_reward_shape(self):
        leaders = rank_checkpoint_leaders(
            [
                checkpoint_leader(
                    reward_shape="score-v1",
                    reward_hash="sha256:score",
                    effective_goal_hash="sha256:goal-score",
                    rank_score=(0.4,),
                    is_default=True,
                ),
                checkpoint_leader(
                    reward_shape="score-step-0p01-v1",
                    reward_hash="sha256:step",
                    effective_goal_hash="sha256:goal-step",
                    rank_score=(0.9,),
                    is_default=False,
                ),
            ]
        )

        self.assertEqual(
            [leader.reward_shape for leader in leaders],
            ["score-v1", "score-step-0p01-v1"],
        )


if __name__ == "__main__":
    unittest.main()
