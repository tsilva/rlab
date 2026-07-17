from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from rlab import experiment_cli, run_observability
from rlab.cli_commands import (
    eval_modal_abandon_command,
    eval_modal_recover_command,
    eval_modal_retry_command,
    experiment_retry_finalization_command,
)
from rlab.modal_eval_cli import build_parser as build_modal_parser


ROOT = Path(__file__).resolve().parents[1]
GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = Path("experiments/recipes/mario/single/ppo.yaml")


class ExperimentCliTests(unittest.TestCase):
    def test_public_parser_uses_only_run_terminology(self) -> None:
        parser = experiment_cli.build_parser()
        args = parser.parse_args(["status", "--run", "42", "--json"])
        self.assertEqual(args.run_id, 42)
        with self.assertRaises(SystemExit):
            parser.parse_args(["status", "--job", "42"])

    def test_launch_requires_machine_and_supports_from_head(self) -> None:
        args = experiment_cli.build_parser().parse_args(
            [
                "launch",
                "--goal-file",
                str(GOAL),
                "--recipe-file",
                str(RECIPE),
                "--machine",
                "beast-3",
                "--from-head",
                "--json",
            ]
        )
        self.assertTrue(args.from_head)
        self.assertTrue(args.output_json)
        self.assertEqual(args.machine, "beast-3")

    def test_from_head_launch_is_in_process_and_never_reloads_controllers(self) -> None:
        args = experiment_cli.build_parser().parse_args(
            [
                "launch",
                "--goal-file",
                str(GOAL),
                "--recipe-file",
                str(RECIPE),
                "--machine",
                "beast-3",
                "--from-head",
                "--json",
            ]
        )

        @contextmanager
        def worktree(_root: Path, _sha: str):
            yield ROOT

        with (
            mock.patch.object(experiment_cli, "repository_root", return_value=ROOT),
            mock.patch.object(
                experiment_cli,
                "_tracked_committed_path",
                side_effect=(GOAL, RECIPE),
            ),
            mock.patch.object(
                experiment_cli,
                "_source",
                return_value=("a" * 40, "main", True),
            ),
            mock.patch.object(experiment_cli, "isolated_head_worktree", side_effect=worktree),
            mock.patch("rlab.fleet_service.require_compatible_controller_services") as preflight,
            mock.patch(
                "rlab.job_queue.cmd_enqueue_train",
                side_effect=lambda _args: (
                    print('{"batch_id":"bx1","job_ids":[9],"jobs":[],"machine":"beast-3"}') or 0
                ),
            ),
            mock.patch("subprocess.Popen") as popen,
        ):
            self.assertEqual(experiment_cli.cmd_launch(args), 0)

        preflight.assert_called_once()
        popen.assert_not_called()

    def test_selected_dirty_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "goal.yaml"
            path.write_text("goal: x\n", encoding="utf-8")
            results = iter(
                (
                    mock.Mock(returncode=0),
                    mock.Mock(returncode=1),
                )
            )
            with mock.patch.object(
                experiment_cli, "_git", side_effect=lambda *_a, **_k: next(results)
            ):
                with self.assertRaisesRegex(ValueError, "uncommitted changes"):
                    experiment_cli._tracked_committed_path(root, path, label="goal")

    def test_generated_remediation_commands_round_trip(self) -> None:
        experiment = experiment_cli.build_parser()
        args = experiment.parse_args(experiment_retry_finalization_command(9)[2:])
        self.assertEqual(args.run_id, 9)
        modal = build_modal_parser()
        for command, destination in (
            (eval_modal_retry_command(12), "eval_job_id"),
            (eval_modal_recover_command(13), "train_job_id"),
            (eval_modal_abandon_command(14), "train_job_id"),
        ):
            parsed = modal.parse_args(command[3:])
            self.assertGreater(int(getattr(parsed, destination)), 0)


class RunObservabilityTests(unittest.TestCase):
    def projection(self, *, status: str, outcome: str | None) -> dict:
        return {
            "schema_version": 1,
            "run_id": 7,
            "status": status,
            "evaluation": {
                "outcome": outcome,
                "acceptance_required": True,
                "learner_stop_observed_at": "2026-07-17T10:00:00Z",
            },
            "wandb": {"url": "https://wandb.ai/e/p/runs/id", "run_id": "id"},
            "artifacts": {"wandb_artifact": "e/p/id-checkpoint:step-10"},
            "incidents": {"current": [], "history": []},
            "timestamps": {},
            "terminal_classification": {
                "succeeded": "accepted" if outcome == "accepted" else "goal_rejected",
                "failed": "operational_failure",
                "canceled": "canceled",
            }[status],
        }

    def test_terminal_classification_uses_authoritative_state(self) -> None:
        accepted = self.projection(status="succeeded", outcome="accepted")
        rejected = self.projection(status="succeeded", outcome="not_accepted")
        failed = self.projection(status="failed", outcome="unknown")
        self.assertEqual(run_observability.terminal_classification(accepted), "accepted")
        self.assertEqual(run_observability.terminal_classification(rejected), "goal_rejected")
        self.assertEqual(run_observability.terminal_classification(failed), "operational_failure")

    def test_follow_emits_url_and_terminal_without_writes(self) -> None:
        events: list[dict] = []
        projection = self.projection(status="succeeded", outcome="accepted")
        conn = mock.MagicMock()
        with mock.patch.object(run_observability, "run_projection", return_value=projection):
            code = run_observability.follow_run(
                conn,
                7,
                emit=lambda event: events.append(dict(event)),
                sleep=lambda _seconds: None,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            [event["event"] for event in events], ["wandb_url", "progress", "terminal"]
        )
        self.assertEqual(
            events[-1]["play_command"], "uv run rlab play e/p/id-checkpoint:step-10 --episodes 1"
        )
        conn.commit.assert_not_called()
        conn.rollback.assert_not_called()

    def test_reporting_warning_does_not_change_classification(self) -> None:
        events: list[dict] = []
        projection = self.projection(status="succeeded", outcome="accepted")
        with mock.patch.object(run_observability, "run_projection", return_value=projection):
            code = run_observability.follow_run(
                mock.MagicMock(),
                7,
                emit=lambda event: events.append(dict(event)),
                sleep=lambda _seconds: None,
                wandb_reader=mock.Mock(side_effect=TimeoutError("offline")),
            )
        self.assertEqual(code, 0)
        self.assertIn("reporting_warning", [event["event"] for event in events])
        self.assertEqual(events[-1]["terminal_classification"], "accepted")

    def test_incident_fingerprint_is_stable(self) -> None:
        row = {"id": 3, "status": "running", "eval_status": "active"}
        diagnostics = {"eval_retries": 1}
        first = run_observability.current_incidents(row, diagnostics)
        second = run_observability.current_incidents(row, diagnostics)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["category"], "eval_execution_failure")

    def test_stalled_cursor_confirmation_is_a_potential_bug_once(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 63,
                "status": "finalizing",
                "eval_status": "finalizing",
                "live_publication_status": "pending",
            },
            {
                "unconfirmed_metric_batches": 8,
                "oldest_unconfirmed_submitted_at": datetime.now(UTC) - timedelta(seconds=121),
            },
        )
        stalled = next(
            incident
            for incident in incidents
            if incident["category"] == "wandb_cursor_confirmation_stalled"
        )
        active = {
            **self.projection(status="succeeded", outcome="accepted"),
            "status": "finalizing",
            "terminal_classification": None,
            "incidents": {"current": [stalled], "history": []},
        }
        terminal = self.projection(status="succeeded", outcome="accepted")
        events: list[dict] = []

        with mock.patch.object(
            run_observability,
            "run_projection",
            side_effect=[active, active, terminal],
        ):
            code = run_observability.follow_run(
                mock.MagicMock(),
                63,
                emit=lambda event: events.append(dict(event)),
                sleep=lambda _seconds: None,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
            )

        self.assertEqual(code, 0)
        potential_bugs = [event for event in events if event["event"] == "potential_bug"]
        self.assertEqual(len(potential_bugs), 1)
        self.assertEqual(
            potential_bugs[0]["incident"]["category"],
            "wandb_cursor_confirmation_stalled",
        )

    def test_completed_publication_attempts_are_history_not_a_current_incident(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 61,
                "status": "succeeded",
                "eval_status": "complete",
                "live_publication_status": "complete",
                "live_publication_attempts": 82,
            },
            {},
        )

        self.assertNotIn(
            "wandb_publication_retry",
            [incident["category"] for incident in incidents],
        )


if __name__ == "__main__":
    unittest.main()
