from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from rlab import training_launch


ROOT = Path(__file__).resolve().parents[1]
GOAL = "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml"
RECIPE = "experiments/recipes/mario/single/ppo.yaml"


class TrainingLaunchTests(unittest.TestCase):
    def test_detached_launch_command_carries_caller_branch(self) -> None:
        args = training_launch.parser().parse_args(
            ["--goal", GOAL, "--recipe", RECIPE]
        )

        argv = training_launch.build_launch_command(args, ROOT, branch="main")

        branch_index = argv.index("--image-branch")
        self.assertEqual(argv[branch_index + 1], "main")
        self.assertIn("rlab", argv)
        self.assertIn("train", argv)

    def test_source_branch_reuses_runtime_resolution_contract(self) -> None:
        with mock.patch(
            "rlab.runtime_refs.current_git_branch", return_value="main"
        ) as current_branch:
            branch = training_launch.source_branch(ROOT)

        self.assertEqual(branch, "main")
        current_branch.assert_called_once_with(ROOT)

    def test_detached_caller_fails_before_worktree_creation(self) -> None:
        with (
            mock.patch.object(training_launch, "repository_root", return_value=ROOT),
            mock.patch.object(
                training_launch,
                "source_branch",
                side_effect=RuntimeError(
                    "automatic runtime builds require a named current Git branch"
                ),
            ),
            mock.patch.object(training_launch, "isolated_worktree") as worktree,
        ):
            with self.assertRaisesRegex(RuntimeError, "named current Git branch"):
                training_launch.main(
                    ["--goal", GOAL, "--recipe", RECIPE, "--validate-only"]
                )

        worktree.assert_not_called()

    def test_eval_retry_and_failed_attempt_share_one_stable_incident(self) -> None:
        incidents = training_launch.potential_bug_incidents(
            {"status": "running", "ready_at": None},
            {
                "failed_eval_jobs": 0,
                "eval_retries": 1,
                "failed_eval_attempts": 1,
            },
            now=10.0,
            progress_changed_at=10.0,
            backlog_started_at=None,
            stale_seconds=300,
        )

        self.assertEqual(list(incidents), ["eval_execution_failure"])
        self.assertIn("eval retries=1", incidents["eval_execution_failure"])
        self.assertIn("failed eval attempts=1", incidents["eval_execution_failure"])

    def test_wandb_retry_is_a_distinct_incident_category(self) -> None:
        incidents = training_launch.potential_bug_incidents(
            {
                "status": "running",
                "ready_at": datetime.now(UTC),
                "wandb_url": "https://wandb.ai/test/project/runs/id",
                "live_publication_attempts": 1,
            },
            {},
            now=10.0,
            progress_changed_at=10.0,
            backlog_started_at=None,
            stale_seconds=300,
        )

        self.assertEqual(
            incidents,
            {"wandb_publication_retry": "W&B publication retries=1"},
        )

    def test_worker_diagnostic_counts_only_active_statuses(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {}

        training_launch.diagnostics(conn, 53)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("status IN ('launching','running')", statement)
        self.assertIn("AS active_worker_attempts", statement)
        self.assertNotIn("AS nonterminal_worker_attempts", statement)

    def test_rejected_final_evidence_uses_canonical_natural_final_job(self) -> None:
        result = {
            "verdict": "rejected",
            "episode_results": [
                {"episode_id": "episode-1", "outcome": "success"},
                {"episode_id": "episode-2", "outcome": "failure"},
            ],
            "claimed_aggregates": {"episodes_completed": 2},
        }
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {
            "outcome": "not_accepted",
            "promoted_eval_job_id": None,
            "attempt_id": "attempt-1",
            "result_json": result,
            "contract_json": {
                "manifest": {
                    "episodes": [
                        {"episode_id": "episode-1"},
                        {"episode_id": "episode-2"},
                        {"episode_id": "episode-3"},
                    ]
                }
            },
        }

        with mock.patch(
            "rlab.modal_eval_protocol.validate_attempt_result",
            return_value=result,
        ):
            evidence = training_launch.evaluation_evidence(conn, 54)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("LEFT JOIN LATERAL", statement)
        self.assertIn("source_announcement_json->>'kind'='final'", statement)
        self.assertTrue(evidence["evidence_valid"])
        self.assertEqual(evidence["episodes_planned"], 3)
        self.assertEqual(evidence["episodes_completed"], 2)
        self.assertEqual(evidence["failures"], 1)

    def test_clean_fail_fast_rejection_is_not_operational_failure(self) -> None:
        classification = training_launch.classify_terminal(
            {
                "status": "succeeded",
                "error": None,
                "live_publication_status": "complete",
            },
            evidence={
                "evidence_valid": True,
                "outcome": "not_accepted",
                "verdict": "rejected",
                "episodes_planned": 100,
                "episodes_completed": 9,
                "failures": 1,
            },
            metrics={
                "eval/acceptance/pass": 0,
                "eval/acceptance/episodes/planned": 100,
                "eval/acceptance/episodes/completed": 9,
                "eval/acceptance/failure/count": 1,
            },
            durable_objects=True,
            best_artifact="entity/project/model:latest",
            operational_clean=True,
            wandb_state="finished",
            audit_errors=[],
        )

        self.assertEqual(classification, "goal_rejected")

    def test_missing_rejection_evidence_is_operational_failure(self) -> None:
        classification = training_launch.classify_terminal(
            {
                "status": "succeeded",
                "error": None,
                "live_publication_status": "complete",
            },
            evidence=None,
            metrics={"eval/acceptance/pass": 0},
            durable_objects=False,
            best_artifact=None,
            operational_clean=True,
            wandb_state="finished",
            audit_errors=[],
        )

        self.assertEqual(classification, "operational_failure")


if __name__ == "__main__":
    unittest.main()
