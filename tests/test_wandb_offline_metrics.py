from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import wandb

from rlab.checkpoint_eval_worker import log_checkpoint_eval_metrics
from rlab.env import EnvConfig
from rlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_BEST,
    LEADER_CHECKPOINT_BEST_RETURN,
)


class WandbOfflineMetricIntegrationTests(unittest.TestCase):
    def test_full_evaluation_scalars_summary_and_table_log_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = wandb.init(
                project="rlab-metrics-schema-test",
                dir=tmp,
                mode="offline",
                reinit="finish_previous",
                config={"metrics_schema_version": 4, "algorithm_id": "ppo"},
                settings=wandb.Settings(silent=True, disable_git=True),
            )
            assert run is not None
            metrics = {
                "return_mean": 5.0,
                "eval/full/episode/return/mean": 5.0,
                "eval/full/episode/return/std": 0.0,
                "eval/full/episode/return/median": 5.0,
                EVAL_FULL_EPISODE_RETURN_BEST: 5.0,
                "eval/full/episode/length/mean": 10.0,
                "eval/full/episode/count": 1,
                "episode_results": [
                    {
                        "start_state": "Start",
                        "return": 5.0,
                        "steps": 10,
                        "outcome": "success",
                        "events": ["goal_reached"],
                        "terminated": True,
                        "truncated": False,
                    }
                ],
            }
            log_checkpoint_eval_metrics(
                run,
                args=argparse.Namespace(
                    selection_rank=[
                        "max(eval/full/episode/return/mean)",
                        "max(eval/full/episode/return/best)",
                        "min(leader/checkpoint/step)",
                    ]
                ),
                metrics=metrics,
                checkpoint_path=Path(tmp) / "model.zip",
                checkpoint_step_value=100,
                artifact_ref="local/model:step-100",
                config=EnvConfig(game="Generic-v0"),
            )

            self.assertEqual(run.config["metrics_schema_version"], 4)
            self.assertEqual(run.summary[LEADER_CHECKPOINT_BEST_RETURN], 5.0)
            run.finish()


if __name__ == "__main__":
    unittest.main()
