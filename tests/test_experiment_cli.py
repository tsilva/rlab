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
RECIPE = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml")


class ExperimentCliTests(unittest.TestCase):
    def test_public_parser_uses_only_run_terminology(self) -> None:
        parser = experiment_cli.build_parser()
        args = parser.parse_args(["status", "--run", "42", "--json"])
        self.assertEqual(args.run_id, 42)
        with self.assertRaises(SystemExit):
            parser.parse_args(["status", "--job", "42"])

    def test_cancel_uses_runtime_current_preflight_without_source_gate(self) -> None:
        args = experiment_cli.build_parser().parse_args(["cancel", "--run", "9"])
        with (
            mock.patch("rlab.fleet_service.require_compatible_controller_services") as preflight,
            mock.patch(
                "rlab.job_queue.cmd_cancel",
                side_effect=lambda _args: print('{"job_ids":[9]}') or 0,
            ),
        ):
            self.assertEqual(experiment_cli.cmd_cancel(args), 0)

        preflight.assert_called_once_with(require_source_current=False)

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

    def test_launch_accepts_existing_runtime_triplet_guards(self) -> None:
        image = "docker:ghcr.io/owner/image@sha256:" + "a" * 64
        args = experiment_cli.build_parser().parse_args(
            [
                "launch",
                "--goal-file",
                str(GOAL),
                "--recipe-file",
                str(RECIPE),
                "--machine",
                "beast-3",
                "--existing-runtime-only",
                "--expected-runtime-image-ref",
                image,
                "--expected-runtime-input-sha256",
                "b" * 64,
                "--expected-runtime-build-source-sha",
                "c" * 40,
            ]
        )

        self.assertTrue(args.existing_runtime_only)
        self.assertEqual(args.expected_runtime_image_ref, image)

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

        preflight.assert_called_once_with(require_source_current=False)
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

    def test_retry_finalization_scopes_legacy_publication_recovery_to_wandb(self) -> None:
        args = experiment_cli.build_parser().parse_args(
            ["retry-finalization", "--run", "184"]
        )
        conn = mock.MagicMock()
        with (
            mock.patch.object(experiment_cli, "repository_root", return_value=ROOT),
            mock.patch.object(experiment_cli, "_connect", return_value=conn),
            mock.patch(
                "rlab.job_queue.finalization_retry_controller_scope",
                return_value="wandb",
            ) as scope,
            mock.patch(
                "rlab.fleet_service.require_compatible_controller_services"
            ) as preflight,
            mock.patch(
                "rlab.job_queue.cmd_retry_finalization",
                side_effect=lambda _args: print('{"job":{"id":184}}') or 0,
            ),
        ):
            self.assertEqual(experiment_cli.cmd_retry_finalization(args), 0)

        scope.assert_called_once_with(conn, job_id=184)
        conn.close.assert_called_once_with()
        preflight.assert_called_once_with(
            require_source_current=False,
            controller_names=("wandb",),
        )
        self.assertEqual(args.preflight_controller_scope, "wandb")

    def test_follow_reconnect_factory_preserves_direct_database_selection(self) -> None:
        args = experiment_cli.build_parser().parse_args(
            ["follow", "--run", "7", "--jsonl", "--direct"]
        )
        root = Path("/repo")
        initial = mock.MagicMock()
        replacement = mock.MagicMock()
        connections = mock.Mock(side_effect=[initial, replacement])

        def follow(conn, run_id, **kwargs):
            self.assertIs(conn, initial)
            self.assertEqual(run_id, 7)
            self.assertIs(kwargs["connection_factory"](), replacement)
            return 0

        with (
            mock.patch.object(experiment_cli, "repository_root", return_value=root),
            mock.patch.object(experiment_cli, "_connect", connections),
            mock.patch.object(run_observability, "follow_run", side_effect=follow),
        ):
            self.assertEqual(experiment_cli.cmd_follow(args), 0)

        self.assertEqual(connections.call_args_list, [mock.call(args, root), mock.call(args, root)])


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

    def test_projection_exposes_recoverable_submission_and_runtime_identity(self) -> None:
        row = {
            "id": 9,
            "batch_id": "bx123",
            "status": "running",
            "machine": "beast-3",
            "repo_git_commit": "a" * 40,
            "runtime_image_ref": "docker:ghcr.io/owner/image@sha256:" + "b" * 64,
            "submission_key": "request-1",
            "submission_ordinal": 0,
            "request_hash": "c" * 64,
            "seed": 123,
            "goal_path": "goal.yaml",
            "goal_sha256": "d" * 64,
            "recipe_path": "recipe.yaml",
            "recipe_sha256": "e" * 64,
            "recipe_payload_json": {
                "recipe": {"recipe_overrides": ["train.backend.config.gamma=0.95"]}
            },
            "train_config": {
                "checkpoint_eval_backend": "modal",
                "runtime_input_sha256": "f" * 64,
                "runtime_build_source_sha": "1" * 40,
            },
        }
        diagnostics = {"blocked_budget_eval_jobs": 1}
        with (
            mock.patch("rlab.job_queue.queue_status", return_value={"runs": [row]}),
            mock.patch.object(run_observability, "_diagnostics", return_value=(diagnostics, [])),
        ):
            projection = run_observability.run_projection(mock.MagicMock(), 9)

        self.assertEqual(projection["submission"]["key"], "request-1")
        self.assertEqual(projection["submission"]["seed"], 123)
        self.assertEqual(
            projection["submission"]["recipe_overrides"],
            ["train.backend.config.gamma=0.95"],
        )
        self.assertEqual(projection["runtime"]["input_sha256"], "f" * 64)
        self.assertEqual(projection["operations"]["blocked_budget_eval_jobs"], 1)

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

    def test_follow_reconnects_without_reemitting_semantic_events(self) -> None:
        events: list[dict] = []
        active = {
            **self.projection(status="succeeded", outcome="accepted"),
            "status": "running",
            "terminal_classification": None,
            "operations": {},
            "incidents": {
                "current": [{"fingerprint": "incident-1", "category": "test"}],
                "history": [],
            },
        }
        terminal = self.projection(status="succeeded", outcome="accepted")
        initial = mock.MagicMock()
        replacement = mock.MagicMock()
        connection_factory = mock.Mock(return_value=replacement)

        with mock.patch.object(
            run_observability,
            "run_projection",
            side_effect=[
                active,
                run_observability.OperationalError("connection dropped"),
                active,
                terminal,
            ],
        ):
            code = run_observability.follow_run(
                initial,
                7,
                emit=lambda event: events.append(dict(event)),
                sleep=lambda _seconds: None,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
                connection_factory=connection_factory,
                reconnect_seconds=0,
            )

        self.assertEqual(code, 0)
        self.assertEqual([event["event"] for event in events].count("wandb_url"), 1)
        self.assertEqual([event["event"] for event in events].count("potential_bug"), 1)
        self.assertEqual([event["event"] for event in events].count("progress"), 2)
        self.assertEqual([event["event"] for event in events].count("terminal"), 1)
        connection_factory.assert_called_once_with()
        initial.close.assert_called_once_with()
        replacement.close.assert_called_once_with()

    def test_follow_retries_transient_reconnect_failures(self) -> None:
        terminal = self.projection(status="succeeded", outcome="accepted")
        initial = mock.MagicMock()
        replacement = mock.MagicMock()
        connection_factory = mock.Mock(
            side_effect=[run_observability.OperationalError("still offline"), replacement]
        )
        sleep = mock.Mock()

        with mock.patch.object(
            run_observability,
            "run_projection",
            side_effect=[run_observability.OperationalError("connection dropped"), terminal],
        ):
            code = run_observability.follow_run(
                initial,
                7,
                emit=lambda _event: None,
                sleep=sleep,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
                connection_factory=connection_factory,
                reconnect_seconds=0.25,
            )

        self.assertEqual(code, 0)
        self.assertEqual(connection_factory.call_count, 2)
        self.assertEqual(sleep.call_args_list, [mock.call(0.25), mock.call(0.25)])
        initial.close.assert_called_once_with()
        replacement.close.assert_called_once_with()

    def test_follow_reconnects_after_interface_error(self) -> None:
        terminal = self.projection(status="succeeded", outcome="accepted")
        initial = mock.MagicMock()
        replacement = mock.MagicMock()

        with mock.patch.object(
            run_observability,
            "run_projection",
            side_effect=[run_observability.InterfaceError("connection closed"), terminal],
        ):
            code = run_observability.follow_run(
                initial,
                7,
                emit=lambda _event: None,
                sleep=lambda _seconds: None,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
                connection_factory=lambda: replacement,
                reconnect_seconds=0,
            )

        self.assertEqual(code, 0)
        initial.close.assert_called_once_with()
        replacement.close.assert_called_once_with()

    def test_follow_does_not_mask_non_connectivity_errors(self) -> None:
        initial = mock.MagicMock()
        connection_factory = mock.Mock()
        with (
            mock.patch.object(
                run_observability,
                "run_projection",
                side_effect=RuntimeError("bad projection"),
            ),
            self.assertRaisesRegex(RuntimeError, "bad projection"),
        ):
            run_observability.follow_run(
                initial,
                7,
                emit=lambda _event: None,
                sleep=lambda _seconds: None,
                connection_factory=connection_factory,
            )

        connection_factory.assert_not_called()
        initial.close.assert_called_once_with()

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
        diagnostics = {"failed_eval_jobs": 1, "eval_outcome": None}
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
                "train_config": {"wandb": True},
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

    def test_active_publisher_lease_is_excluded_from_cursor_stall_query(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {}
        cursor.fetchall.return_value = []

        run_observability._diagnostics(conn, 63)

        statement = cursor.execute.call_args_list[0].args[0]
        self.assertIn("b.lease_owner IS NOT NULL", statement)
        self.assertIn("b.lease_expires_at > clock_timestamp()", statement)

    def test_actor_start_failure_has_distinct_incident(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 63,
                "status": "running",
                "eval_status": "active",
                "live_publication_status": "pending",
                "live_publication_error": ("publisher actor startup failed: missing PostgreSQL CA"),
                "train_config": {"wandb": True},
            },
            {},
        )

        self.assertEqual(incidents[0]["category"], "publisher_unavailable")

    def test_budget_block_is_attention_once_and_not_a_potential_bug(self) -> None:
        active = {
            **self.projection(status="succeeded", outcome="accepted"),
            "status": "finalizing",
            "terminal_classification": None,
            "operations": {"blocked_budget_eval_jobs": 2},
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
                7,
                emit=lambda event: events.append(dict(event)),
                sleep=lambda _seconds: None,
                wandb_reader=lambda _url: {"state": "finished", "metrics": {}},
            )

        self.assertEqual(code, 0)
        attention = [event for event in events if event["event"] == "attention_required"]
        self.assertEqual(len(attention), 1)
        self.assertEqual(attention[0]["attention"]["category"], "eval_budget_blocked")
        self.assertNotIn("potential_bug", [event["event"] for event in events])

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

    def test_visible_run_retry_is_not_mislabeled_as_startup_stall(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 62,
                "status": "running",
                "train_config": {"wandb": True},
                "live_publication_status": "pending",
                "live_publication_attempts": 1,
                "learner_ready_at": datetime.now(UTC) - timedelta(hours=1),
                "wandb_url": "https://wandb.ai/entity/project/runs/rlab-run",
            },
            {},
        )

        categories = [incident["category"] for incident in incidents]
        self.assertIn("wandb_publication_retry", categories)
        self.assertNotIn("wandb_publication_stalled", categories)

    def test_concrete_availability_artifact_precedes_promoted_fallback(self) -> None:
        self.assertEqual(
            run_observability._wandb_artifact_ref(
                {
                    "wandb_artifact_ref": "entity/project/rlab-run-checkpoint:v4",
                    "wandb_url": "https://wandb.ai/entity/project/runs/rlab-run",
                    "wandb_run_id": "rlab-run",
                    "promoted_step": None,
                }
            ),
            "entity/project/rlab-run-checkpoint:v4",
        )

    def test_expected_cancellation_and_historical_eval_failures_are_not_incidents(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 64,
                "status": "finalizing",
                "launch_state": "canceled",
                "launch_error": "cancel requested",
                "cancel_requested": True,
                "eval_status": "finalizing",
                "train_config": {"wandb": True},
                "live_publication_status": "pending",
            },
            {
                "eval_outcome": "canceled",
                "eval_retries": 1,
                "failed_eval_attempts": 1,
                "failed_eval_jobs": 0,
            },
        )

        self.assertEqual(incidents, [])

    def test_publication_failure_suppresses_duplicate_terminal_failure(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 64,
                "status": "finalization_failed",
                "error": "remote failed",
                "train_config": {"wandb": True},
                "live_publication_status": "failed",
                "live_publication_attempts": 3,
                "live_publication_error": "remote failed",
            },
            {},
        )

        self.assertEqual(
            [incident["category"] for incident in incidents],
            ["wandb_publication_failed"],
        )

    def test_terminal_launch_error_is_not_reported_twice(self) -> None:
        incidents = run_observability.current_incidents(
            {
                "id": 65,
                "status": "failed",
                "error": "checkpoint coordinator exited",
                "launch_state": "failed",
                "launch_error": "checkpoint coordinator exited",
                "eval_status": "failed",
            },
            {},
        )

        self.assertEqual(
            [incident["category"] for incident in incidents],
            ["terminal_operational_failure"],
        )


if __name__ == "__main__":
    unittest.main()
