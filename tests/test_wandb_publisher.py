from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab.fleet_wandb_publisher import (
    WandbArtifactVisibilityError,
    WandbCursorConfirmationError,
    WandbFinalizationVerificationError,
    WandbPublicationState,
    _canonical_goal_summary,
    _artifact_receipts_from_remote,
    _cursor_mapping,
    _drain_claim,
    _partition_batches,
    _publication_is_pristine,
    _record_committed_effects,
    _reconcile_non_modal_publication_finishing,
    _reconcile_terminal_artifact_publication,
    _repair_artifact_projection_identity,
    _raise_for_stalled_confirmations,
    _remote_publication_state,
    _summary_step_max,
    drain_cycle_parallel,
    finalize_finishing_run,
    publish_claimed_run,
    run_publisher_actor,
)
from rlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_STEP,
)
from rlab.metric_store import MetricStore
from rlab.wandb_publisher import project_payload_to_run, publish_pending_frames
from rlab import wandb_publisher
from rlab.wandb_utils import configure_wandb_metrics


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))


class FakeHtml:
    def __init__(self, data, **kwargs) -> None:
        self.data = data
        self.kwargs = kwargs


class WandbPublisherTests(unittest.TestCase):
    @staticmethod
    def _acceptance_payload(
        checkpoint_step: int,
        *,
        passed: bool,
        canonical: bool,
    ) -> dict[str, object]:
        metrics = {
            "global_step": checkpoint_step,
            EVAL_ACCEPTANCE_PASS: 1.0 if passed else 0.0,
        }
        raw_metrics = (
            {
                "checkpoint_step": checkpoint_step,
                "eval/full/outcome/success/rate/min": 1.0,
                "eval/full/outcome/success/rate/mean": 1.0,
                "eval/full/episode/return/mean": 3144.17,
                "eval/full/episode/return/best": 4000.0,
                "eval/full/progress/x/max": 3160,
            }
            if passed
            else {"checkpoint_step": checkpoint_step}
        )
        return {
            "train_config": {
                "wandb_run_id": "rlab-test",
                "selection_rank": [
                    "max(eval/full/outcome/success/rate/min)",
                    "max(eval/full/outcome/success/rate/mean)",
                    "min(leader/checkpoint/steps_to_goal)",
                    "max(eval/full/episode/return/mean)",
                ],
            },
            "purpose": "acceptance",
            "checkpoint_uri": f"s3://bucket/{checkpoint_step}/model.zip",
            "checkpoint_step": checkpoint_step,
            "canonical_promotion": canonical,
            "decision": {
                "passed": passed,
                "metrics": metrics,
                "raw_metrics": raw_metrics,
            },
        }

    def test_terminal_goal_summary_uses_the_promoted_acceptance_evidence(self) -> None:
        summary = _canonical_goal_summary(
            {
                "outcome": "accepted",
                "promotion_revision": 1,
                "train_config": {
                    "selection_rank": [
                        "max(eval/full/outcome/success/rate/min)",
                        "max(eval/full/outcome/success/rate/mean)",
                        "min(leader/checkpoint/steps_to_goal)",
                        "max(eval/full/episode/return/mean)",
                    ]
                },
                "promotion_json": {
                    "checkpoint_step": 8_000_000,
                    "checkpoint_uri": "s3://bucket/8000000/model.zip",
                    "raw_metrics": {
                        "episodes": 100,
                        "_acceptance_duration_seconds": 33.75,
                        "_acceptance_aggregates": {
                            "episodes_planned": 100,
                            "episodes_completed": 100,
                            "failure_count": 0,
                        },
                        "eval/full/outcome/success/rate/min": 1.0,
                        "eval/full/outcome/success/rate/mean": 1.0,
                        "eval/full/episode/return/mean": 3144.17,
                    },
                },
            }
        )

        self.assertEqual(summary["eval/acceptance/pass"], 1.0)
        self.assertEqual(summary["eval/acceptance/episodes/planned"], 100)
        self.assertEqual(summary["eval/acceptance/episodes/completed"], 100)
        self.assertEqual(summary["eval/acceptance/failure/count"], 0)
        self.assertEqual(summary["eval/full/checkpoint/step"], 8_000_000)
        self.assertEqual(summary["eval/full/outcome/success/rate/min"], 1.0)
        self.assertEqual(summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS], 1.0)
        self.assertEqual(summary[LEADER_CHECKPOINT_STEP], 8_000_000)
        self.assertEqual(
            summary[LEADER_CHECKPOINT_ARTIFACT_REF],
            "s3://bucket/8000000/model.zip",
        )

    def test_canonical_acceptance_survives_later_noncanonical_rejection(self) -> None:
        run = FakeRun()
        with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace()}):
            project_payload_to_run(
                run,
                self._acceptance_payload(8_000_000, passed=True, canonical=True),
            )
            project_payload_to_run(
                run,
                self._acceptance_payload(11_750_000, passed=False, canonical=False),
            )

        self.assertEqual(run.summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS], 1.0)
        self.assertEqual(run.summary[LEADER_CHECKPOINT_STEP], 8_000_000)
        self.assertEqual(run.summary["rlab/goal/outcome"], "accepted")
        self.assertEqual(
            [row[EVAL_ACCEPTANCE_PASS] for row in run.logged],
            [1.0, 0.0],
        )

    def test_reverse_order_batch_still_projects_the_canonical_acceptance(self) -> None:
        run = FakeRun()
        with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace()}):
            for payload in (
                self._acceptance_payload(11_750_000, passed=False, canonical=False),
                self._acceptance_payload(8_000_000, passed=True, canonical=True),
            ):
                project_payload_to_run(run, payload)

        self.assertEqual(run.summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS], 1.0)
        self.assertEqual(run.summary[LEADER_CHECKPOINT_STEP], 8_000_000)

    def test_acceptance_history_summary_uses_max(self) -> None:
        run = mock.MagicMock()

        configure_wandb_metrics(run)

        run.define_metric.assert_any_call(
            EVAL_ACCEPTANCE_PASS,
            step_metric="global_step",
            summary="max",
        )

    def test_publisher_actor_survives_idle_gaps_until_remote_completion(self) -> None:
        conn = mock.MagicMock()
        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.drain_once",
                side_effect=[2, 3],
            ) as drain,
            mock.patch(
                "rlab.fleet_wandb_publisher._publisher_actor_done",
                side_effect=[False, True],
            ),
            mock.patch("rlab.fleet_wandb_publisher.time.sleep") as sleep,
        ):
            published = run_publisher_actor(
                conn,
                41,
                limit=100,
                poll_seconds=2.0,
            )

        self.assertEqual(published, 5)
        self.assertEqual(drain.call_count, 2)
        self.assertEqual(
            drain.call_args_list[0].kwargs,
            {"limit": 100, "train_job_id": 41},
        )
        sleep.assert_called_once_with(2.0)
        statements = [
            call.args[0]
            for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
        ]
        self.assertTrue(any("pg_try_advisory_lock" in statement for statement in statements))
        self.assertTrue(any("pg_advisory_unlock" in statement for statement in statements))

    def test_parallel_cycle_uses_one_isolated_process_per_run(self) -> None:
        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.pending_metric_run_ids",
                return_value=[41, 42, 43],
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.subprocess.Popen",
            ) as popen,
        ):
            result = drain_cycle_parallel(
                mock.MagicMock(),
                repo_root=Path.cwd(),
                max_runs=3,
                limit=20,
            )

        self.assertEqual(
            result,
            {
                "runs_attempted": 3,
                "runs_started": 3,
                "batches_published": 0,
                "runs_failed": 0,
            },
        )
        self.assertEqual(popen.call_count, 3)
        train_ids = {
            call.args[0][call.args[0].index("--train-job-id") + 1] for call in popen.call_args_list
        }
        self.assertEqual(train_ids, {"41", "42", "43"})
        self.assertTrue(all(call.kwargs["start_new_session"] for call in popen.call_args_list))

    def test_resumed_projection_can_flush_without_finishing_active_run(self) -> None:
        captured: dict[str, object] = {}
        fake_run = SimpleNamespace(define_metric=lambda *_args, **_kwargs: None)

        def init(**kwargs):
            captured.update(kwargs)
            return fake_run

        fake_wandb = SimpleNamespace(
            init=init,
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch.object(wandb_publisher, "load_wandb_env"),
            mock.patch.object(
                wandb_publisher,
                "resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            mock.patch.object(
                wandb_publisher,
                "configure_wandb_metrics",
                side_effect=lambda run: run,
            ),
        ):
            projector = wandb_publisher.WandbProjector.resume(
                {"wandb_run_id": "rlab-test", "game": "game"},
                update_finish_state=False,
            )

        self.assertIs(projector.run, fake_run)
        self.assertFalse(captured["settings"].x_update_finish_state)

    def test_wandb_nested_summary_mapping_is_accepted(self) -> None:
        class ItemsOnly:
            def items(self):
                return (("train-7", 3),)

        self.assertEqual(
            _cursor_mapping(ItemsOnly()),
            {"train-7": 3},
        )

    def test_wandb_nested_step_summary_mapping_is_accepted(self) -> None:
        class ItemsOnly:
            def items(self):
                return (("max", 4196496),)

        self.assertEqual(_summary_step_max(ItemsOnly()), 4196496.0)
        self.assertEqual(_summary_step_max({"max": 2000000}), 2000000.0)
        self.assertIsNone(_summary_step_max({}))

    def test_remote_publication_api_errors_are_not_converted_to_empty_cursors(self) -> None:
        fake_wandb = SimpleNamespace(
            Api=lambda **_kwargs: SimpleNamespace(
                run=mock.Mock(side_effect=TimeoutError("offline"))
            )
        )
        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch("rlab.fleet_wandb_publisher.load_wandb_env"),
            mock.patch(
                "rlab.fleet_wandb_publisher.resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            self.assertRaisesRegex(TimeoutError, "offline"),
        ):
            _remote_publication_state({"wandb_run_id": "run-7", "game": "SuperMarioBros-Nes-v0"})

    def test_pristine_publication_requires_run_wide_absence_of_submission_evidence(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {
            "streams_unpublished": True,
            "batches_unsubmitted": True,
        }
        run = {"id": 66, "wandb_url": None, "train_config": {"wandb": True}}

        self.assertTrue(_publication_is_pristine(conn, run))
        statement = cursor.execute.call_args.args[0]
        self.assertIn("s.submitted_sequence <> 0", statement)
        self.assertIn("b.submitted_at IS NOT NULL", statement)

        cursor.fetchone.return_value = {
            "streams_unpublished": False,
            "batches_unsubmitted": True,
        }
        self.assertFalse(_publication_is_pristine(conn, run))
        self.assertFalse(_publication_is_pristine(conn, {**run, "wandb_url": "https://wandb"}))

    def test_pristine_first_cycle_creates_without_remote_preflight(self) -> None:
        conn = mock.MagicMock()
        run = {
            "id": 66,
            "status": "running",
            "wandb_url": None,
            "wandb_run_id": "rlab-pristine",
            "train_config": {"wandb": True, "game": "game"},
        }
        batches = [
            {
                "id": 1,
                "stream_id": "train-66",
                "batch_sequence": 1,
                "submitted_sequence": 0,
                "payload": b"payload",
            }
        ]
        projector = SimpleNamespace(
            run=SimpleNamespace(url="https://wandb/run", summary={}),
            close=mock.Mock(),
        )
        with (
            mock.patch("rlab.fleet_wandb_publisher._publication_is_pristine", return_value=True),
            mock.patch("rlab.fleet_wandb_publisher._remote_publication_state") as remote_state,
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ) as resume,
            mock.patch("rlab.fleet_wandb_publisher.decode_metric_batch", return_value=[]),
            mock.patch("rlab.fleet_wandb_publisher.resolve_env_config", return_value={}),
            mock.patch("rlab.fleet_wandb_publisher.env_config_from_args", return_value={}),
            mock.patch("rlab.fleet_wandb_publisher.mark_submitted_batches") as submitted,
        ):
            publish_claimed_run(conn, run, batches)

        remote_state.assert_not_called()
        self.assertTrue(resume.call_args.kwargs["allow_create"])
        submitted.assert_called_once_with(conn, batches)

    def test_non_modal_terminal_outbox_moves_to_finishing_only_when_drained(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        self.assertTrue(_reconcile_non_modal_publication_finishing(conn, 70))

        statement, params = cursor.execute.call_args.args
        self.assertIn("t.process_exited_at IS NOT NULL", statement)
        self.assertIn("checkpoint_eval_backend", statement)
        self.assertIn("<> 'modal'", statement)
        self.assertIn("s.final_sequence IS NULL", statement)
        self.assertIn("s.published_sequence < s.final_sequence", statement)
        self.assertEqual(params, {"train_job_id": 70})

    def test_pristine_zero_batch_finalization_creates_preassigned_run(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 70,
                "wandb_url": None,
                "live_publication_attempts": 0,
                "train_config": {
                    "wandb": True,
                    "wandb_run_id": "rlab-pristine-final",
                    "game": "Bandit-v0",
                },
                "outcome": "unknown",
                "promotion_json": None,
            },
            {"streams_unpublished": True, "batches_unsubmitted": True},
        ]
        cursor.fetchall.return_value = []
        cursor.rowcount = 1
        remote = SimpleNamespace(
            url="https://wandb/run",
            state="finished",
            summary={},
            logged_artifacts=lambda: [],
        )
        projector = SimpleNamespace(run=remote, close=mock.Mock())

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ) as resume,
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                return_value=remote,
            ) as api_run,
        ):
            self.assertTrue(finalize_finishing_run(conn, 70))

        self.assertTrue(resume.call_args.kwargs["allow_create"])
        self.assertTrue(resume.call_args.kwargs["update_finish_state"])
        api_run.assert_called_once()
        complete = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_status = 'complete'" in call.args[0]
        )
        self.assertEqual(complete.args[1]["wandb_url"], "https://wandb/run")
        select = next(
            call for call in cursor.execute.call_args_list if "LEFT JOIN eval_runs" in call.args[0]
        )
        self.assertIn("COALESCE(r.outcome, 'unknown')", select.args[0])

    def test_nonpristine_api_failure_never_falls_back_to_creation(self) -> None:
        run = {
            "id": 66,
            "status": "running",
            "wandb_run_id": "rlab-existing",
            "train_config": {"wandb": True, "game": "game"},
        }
        batches = [{"id": 1, "payload": b"payload"}]
        with (
            mock.patch("rlab.fleet_wandb_publisher._publication_is_pristine", return_value=False),
            mock.patch(
                "rlab.fleet_wandb_publisher._remote_publication_state",
                side_effect=TimeoutError("offline"),
            ),
            mock.patch("rlab.fleet_wandb_publisher.WandbProjector.resume") as resume,
            mock.patch("rlab.fleet_wandb_publisher.decode_metric_batch", return_value=[]),
            self.assertRaisesRegex(TimeoutError, "offline"),
        ):
            publish_claimed_run(mock.MagicMock(), run, batches)
        resume.assert_not_called()

    def test_mailbox_batches_have_distinct_confirmed_submitted_and_new_states(self) -> None:
        batches = [
            {"id": 1, "stream_id": "train-7", "batch_sequence": 1, "submitted_sequence": 2},
            {"id": 2, "stream_id": "train-7", "batch_sequence": 2, "submitted_sequence": 2},
            {"id": 3, "stream_id": "train-7", "batch_sequence": 3, "submitted_sequence": 2},
        ]

        confirmed, awaiting, unpublished = _partition_batches(
            batches,
            {"train-7": 1},
        )

        self.assertEqual([row["id"] for row in confirmed], [1])
        self.assertEqual([row["id"] for row in awaiting], [2])
        self.assertEqual([row["id"] for row in unpublished], [3])

    def test_artifact_only_session_preserves_durable_training_cursor_floor(self) -> None:
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
            {"stream_id": "train-7", "sequence": 39},
        ]
        run = {
            "id": 7,
            "status": "running",
            "wandb_url": "https://wandb/run",
            "train_config": {"wandb": True, "wandb_run_id": "rlab-7", "game": "game"},
        }
        batches = [
            {
                "id": 8,
                "stream_id": "artifact-v2-7-14-availability-r0",
                "batch_sequence": 1,
                "submitted_sequence": 0,
                "payload": b"payload",
            }
        ]
        projector = SimpleNamespace(
            run=SimpleNamespace(url="https://wandb/run", summary={}),
            close=mock.Mock(),
        )

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._publication_is_pristine",
                return_value=False,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher._remote_publication_state",
                return_value=WandbPublicationState(
                    state="running",
                    cursors={"train-7": 35},
                    step_max=7_500_000,
                ),
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ),
            mock.patch("rlab.fleet_wandb_publisher.decode_metric_batch", return_value=[]),
            mock.patch("rlab.fleet_wandb_publisher.resolve_env_config", return_value={}),
            mock.patch("rlab.fleet_wandb_publisher.env_config_from_args", return_value={}),
            mock.patch("rlab.fleet_wandb_publisher.mark_submitted_batches"),
        ):
            publish_claimed_run(conn, run, batches)

        self.assertEqual(
            projector.run.summary["_rlab_telemetry_cursors"],
            {
                "train-7": 39,
                "artifact-v2-7-14-availability-r0": 1,
            },
        )

    def test_stalled_crashed_run_reasserts_durable_cursor_without_relogging(self) -> None:
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
            {"stream_id": "train-7", "sequence": 39},
        ]
        run = {
            "id": 7,
            "status": "running",
            "wandb_url": "https://wandb/run",
            "train_config": {"wandb": True, "wandb_run_id": "rlab-7", "game": "game"},
        }
        batches = [
            {
                "id": 8,
                "stream_id": "train-7",
                "batch_sequence": 39,
                "submitted_sequence": 39,
                "submitted_at": datetime.now(UTC) - timedelta(minutes=5),
                "payload": b"payload",
            }
        ]
        projector = SimpleNamespace(
            run=SimpleNamespace(url="https://wandb/run", summary={}),
            close=mock.Mock(),
        )

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._publication_is_pristine",
                return_value=False,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher._remote_publication_state",
                return_value=WandbPublicationState(
                    state="crashed",
                    cursors={"train-7": 35},
                    step_max=7_500_000,
                ),
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ) as resume,
            mock.patch("rlab.fleet_wandb_publisher.decode_metric_batch", return_value=[]),
            mock.patch("rlab.fleet_wandb_publisher.mark_submitted_batches") as submitted,
        ):
            self.assertEqual(publish_claimed_run(conn, run, batches), 0)

        self.assertEqual(
            projector.run.summary["_rlab_telemetry_cursors"],
            {"train-7": 39},
        )
        self.assertFalse(resume.call_args.kwargs["update_finish_state"])
        submitted.assert_called_once_with(conn, batches, refresh_submitted_at=True)

    def test_remote_confirmation_restores_submitted_sequence_before_publish_commit(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        batch = {"id": 1, "stream_id": "artifact-v2-7-3-availability-r0", "batch_sequence": 1}

        _record_committed_effects(
            conn,
            run={"id": 7},
            batches=[batch],
            decoded={1: []},
            wandb_url=None,
        )

        advance = next(
            call
            for call in cursor.execute.call_args_list
            if "UPDATE metric_streams" in call.args[0]
        )
        self.assertIn(
            "submitted_sequence = GREATEST(submitted_sequence, %(sequence)s)",
            advance.args[0],
        )
        self.assertIn(
            "published_sequence = GREATEST(published_sequence, %(sequence)s)",
            advance.args[0],
        )

    def test_artifact_receipt_requires_exact_api_visible_membership(self) -> None:
        payload = {
            "train_config": {"wandb_run_id": "rlab-7"},
            "train_job_id": 7,
            "ledger_id": 3,
            "artifact_kind": "checkpoint",
            "publication_role": "availability",
            "promotion_revision": 0,
            "publication_stream_id": "artifact-v2-7-3-availability-r0",
            "announcement_sha256": "1" * 64,
            "checkpoint_step": 300,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_uri": "s3://bucket/model.zip",
            "metadata_uri": "s3://bucket/metadata.json",
            "metadata_sha256": "3" * 64,
            "recipe_uri": "s3://bucket/recipe.json",
            "recipe_sha256": "4" * 64,
            "artifact_aliases": ["step-300"],
        }
        artifact = SimpleNamespace(
            type="model",
            version="v4",
            qualified_name="tsilva/project/rlab-7-checkpoint:v4",
            aliases=["latest", "step-300"],
            metadata={
                "artifact_publication_schema": "v2",
                "train_job_id": 7,
                "ledger_id": 3,
                "artifact_kind": "checkpoint",
                "publication_role": "availability",
                "promotion_revision": 0,
                "publication_stream_id": "artifact-v2-7-3-availability-r0",
                "announcement_sha256": "1" * 64,
                "checkpoint_step": 300,
                "checkpoint_sha256": "2" * 64,
                "artifact_storage_uri": "s3://bucket/model.zip",
                "metadata_uri": "s3://bucket/metadata.json",
                "metadata_sha256": "3" * 64,
                "recipe_uri": "s3://bucket/recipe.json",
                "recipe_sha256": "4" * 64,
            },
        )
        remote = SimpleNamespace(logged_artifacts=lambda: [artifact])

        receipt = _artifact_receipts_from_remote(remote, [payload])[0]

        self.assertEqual(receipt["artifact_ref"], "tsilva/project/rlab-7-checkpoint:v4")
        self.assertEqual(receipt["artifact_version"], "v4")
        artifact.metadata["metadata_sha256"] = "wrong"
        with self.assertRaises(WandbArtifactVisibilityError):
            _artifact_receipts_from_remote(remote, [payload])

    def test_deduplicated_promotion_artifact_confirms_availability_receipt(self) -> None:
        payload = {
            "train_config": {"wandb_run_id": "rlab-7"},
            "train_job_id": 7,
            "ledger_id": 3,
            "artifact_kind": "checkpoint",
            "publication_role": "availability",
            "promotion_revision": 0,
            "publication_stream_id": "artifact-v2-7-3-availability-r0",
            "announcement_sha256": "1" * 64,
            "checkpoint_step": 300,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_uri": "s3://bucket/model.zip",
            "metadata_uri": "s3://bucket/metadata.json",
            "metadata_sha256": "3" * 64,
            "recipe_uri": "s3://bucket/recipe.json",
            "recipe_sha256": "4" * 64,
            "artifact_aliases": ["step-300"],
        }
        artifact = SimpleNamespace(
            type="model",
            version="v4",
            qualified_name="tsilva/project/rlab-7-checkpoint:v4",
            aliases=["latest", "promoted", "step-300"],
            metadata={
                "artifact_publication_schema": "v2",
                "train_job_id": 7,
                "ledger_id": 3,
                "artifact_kind": "checkpoint",
                "publication_role": "promotion",
                "promotion_revision": 1,
                "publication_stream_id": "artifact-v2-7-3-promotion-r1",
                "announcement_sha256": "1" * 64,
                "checkpoint_step": 300,
                "checkpoint_sha256": "2" * 64,
                "artifact_storage_uri": "s3://bucket/model.zip",
                "metadata_uri": "s3://bucket/metadata.json",
                "metadata_sha256": "3" * 64,
                "recipe_uri": "s3://bucket/recipe.json",
                "recipe_sha256": "4" * 64,
            },
        )

        receipt = _artifact_receipts_from_remote(
            SimpleNamespace(logged_artifacts=lambda: [artifact]),
            [payload],
        )[0]

        self.assertEqual(receipt["role"], "availability")
        self.assertEqual(receipt["stream_id"], "artifact-v2-7-3-availability-r0")
        self.assertEqual(receipt["artifact_ref"], "tsilva/project/rlab-7-checkpoint:v4")

    def test_queued_artifact_repairs_empty_run_id_from_authoritative_job(self) -> None:
        repaired = _repair_artifact_projection_identity(
            {
                "artifact_publication_schema": "v2",
                "train_config": {"wandb_run_id": "", "game": "Bandit-v0"},
            },
            {"wandb_run_id": "rlab-authoritative", "run_name": "run-7"},
        )

        self.assertEqual(repaired["train_config"]["wandb_run_id"], "rlab-authoritative")
        self.assertEqual(repaired["train_config"]["run_name"], "run-7")
        self.assertEqual(repaired["train_config"]["game"], "Bandit-v0")

    def test_visible_cursor_without_visible_artifact_retries_without_relogging(self) -> None:
        payload = {
            "artifact_publication_schema": "v2",
            "train_config": {"wandb_run_id": "rlab-7"},
            "train_job_id": 7,
            "ledger_id": 3,
            "artifact_kind": "checkpoint",
            "publication_role": "availability",
            "promotion_revision": 0,
            "publication_stream_id": "artifact-v2-7-3-availability-r0",
            "announcement_sha256": "1" * 64,
            "checkpoint_step": 300,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_uri": "s3://bucket/model.zip",
            "metadata_uri": "s3://bucket/metadata.json",
            "metadata_sha256": "3" * 64,
            "artifact_aliases": ["step-300"],
        }
        run = {
            "id": 7,
            "status": "running",
            "wandb_url": "https://wandb/run",
            "train_config": {"wandb": True, "wandb_run_id": "rlab-7"},
        }
        batches = [
            {
                "id": 8,
                "stream_id": "artifact-v2-7-3-availability-r0",
                "batch_sequence": 1,
                "submitted_sequence": 1,
                "payload": b"payload",
            }
        ]
        frame = {"kind": "projection", "payload": payload}
        remote = SimpleNamespace(logged_artifacts=lambda: [])

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._publication_is_pristine",
                return_value=False,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher._remote_publication_state",
                return_value=WandbPublicationState(
                    state="running",
                    cursors={"artifact-v2-7-3-availability-r0": 1},
                    step_max=None,
                ),
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.decode_metric_batch",
                return_value=[frame],
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                return_value=remote,
            ),
            mock.patch("rlab.fleet_wandb_publisher.WandbProjector.resume") as resume,
            self.assertRaises(WandbArtifactVisibilityError),
        ):
            publish_claimed_run(mock.MagicMock(), run, batches)

        resume.assert_not_called()

    def test_crashed_run_with_eight_missing_cursors_fails_confirmation(self) -> None:
        batches = [
            {
                "id": index,
                "stream_id": f"eval-{index}",
                "batch_sequence": 1,
                "submitted_at": None,
            }
            for index in range(1, 9)
        ]
        remote = WandbPublicationState(state="crashed", cursors={}, step_max=None)

        with self.assertRaisesRegex(
            WandbCursorConfirmationError,
            "remote_state=crashed",
        ) as caught:
            _raise_for_stalled_confirmations(batches, remote)

        self.assertIn("eval-1:expected=1:observed=0", str(caught.exception))
        self.assertIn("eval-8:expected=1:observed=0", str(caught.exception))

    def test_running_run_confirmation_is_bounded_at_two_minutes(self) -> None:
        now = datetime.now(UTC)
        remote = WandbPublicationState(state="running", cursors={}, step_max=None)
        recent = {
            "stream_id": "eval-recent",
            "batch_sequence": 1,
            "submitted_at": now - timedelta(seconds=119),
        }
        stale = {
            "stream_id": "eval-stale",
            "batch_sequence": 1,
            "submitted_at": now - timedelta(seconds=120),
        }

        _raise_for_stalled_confirmations([recent], remote, now=now)
        with self.assertRaisesRegex(
            WandbCursorConfirmationError,
            "reason=confirmation_timeout",
        ):
            _raise_for_stalled_confirmations([stale], remote, now=now)

    def test_third_finalizing_publication_failure_is_terminal_and_preserves_batches(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        run = {
            "id": 63,
            "status": "finalizing",
            "live_publication_attempts": 2,
        }
        batches = [{"id": 1, "stream_id": "eval-1", "batch_sequence": 1}]
        error = WandbCursorConfirmationError("cursor missing")

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.publish_claimed_run",
                side_effect=error,
            ),
            mock.patch("rlab.fleet_wandb_publisher.release_metric_batch_claims") as release,
            mock.patch("rlab.fleet_wandb_publisher.record_job_event") as event,
            mock.patch("rlab.fleet_wandb_publisher.release_wandb_run_lock"),
            self.assertRaises(WandbCursorConfirmationError),
        ):
            _drain_claim(conn, run, batches)

        release.assert_called_once_with(conn, batches, error=repr(error))
        update = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_attempts = %(attempts)s" in call.args[0]
        )
        self.assertEqual(update.args[1]["attempts"], 3)
        self.assertTrue(update.args[1]["terminal"])
        self.assertNotIn("DELETE FROM metric_batches", update.args[0])
        event.assert_called_once()

    def test_recent_artifact_visibility_lag_does_not_consume_finalization_attempt(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        run = {
            "id": 64,
            "status": "finalizing",
            "live_publication_attempts": 2,
        }
        batches = [
            {
                "id": 1,
                "stream_id": "artifact-v2-64-1-availability-r0",
                "batch_sequence": 1,
                "submitted_at": datetime.now(UTC) - timedelta(seconds=119),
            }
        ]
        error = WandbArtifactVisibilityError("artifact membership pending")

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.publish_claimed_run",
                side_effect=error,
            ),
            mock.patch("rlab.fleet_wandb_publisher.release_metric_batch_claims"),
            mock.patch("rlab.fleet_wandb_publisher.record_job_event") as event,
            mock.patch("rlab.fleet_wandb_publisher.release_wandb_run_lock"),
            self.assertRaises(WandbArtifactVisibilityError),
        ):
            _drain_claim(conn, run, batches)

        update = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_attempts = %(attempts)s" in call.args[0]
        )
        self.assertEqual(update.args[1]["attempts"], 2)
        self.assertFalse(update.args[1]["terminal"])
        self.assertEqual(update.args[1]["retry_delay"], 5)
        self.assertTrue(event.call_args.kwargs["metadata"]["visibility_propagating"])

    def test_stale_artifact_visibility_lag_consumes_finalization_attempt(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        run = {
            "id": 65,
            "status": "finalizing",
            "live_publication_attempts": 2,
        }
        batches = [
            {
                "id": 1,
                "stream_id": "artifact-v2-65-1-availability-r0",
                "batch_sequence": 1,
                "submitted_at": datetime.now(UTC) - timedelta(seconds=120),
            }
        ]
        error = WandbArtifactVisibilityError("artifact membership still missing")

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.publish_claimed_run",
                side_effect=error,
            ),
            mock.patch("rlab.fleet_wandb_publisher.release_metric_batch_claims"),
            mock.patch("rlab.fleet_wandb_publisher.record_job_event") as event,
            mock.patch("rlab.fleet_wandb_publisher.release_wandb_run_lock"),
            self.assertRaises(WandbArtifactVisibilityError),
        ):
            _drain_claim(conn, run, batches)

        update = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_attempts = %(attempts)s" in call.args[0]
        )
        self.assertEqual(update.args[1]["attempts"], 3)
        self.assertTrue(update.args[1]["terminal"])
        self.assertFalse(event.call_args.kwargs["metadata"]["visibility_propagating"])

    def test_active_training_publication_failure_never_terminalizes_learner(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        run = {"id": 7, "status": "running", "live_publication_attempts": 99}
        batches = [{"id": 1, "stream_id": "train-7", "batch_sequence": 1}]

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.publish_claimed_run",
                side_effect=RuntimeError("W&B unavailable"),
            ),
            mock.patch("rlab.fleet_wandb_publisher.release_metric_batch_claims"),
            mock.patch("rlab.fleet_wandb_publisher.record_job_event"),
            mock.patch("rlab.fleet_wandb_publisher.release_wandb_run_lock"),
            self.assertRaisesRegex(RuntimeError, "W&B unavailable"),
        ):
            _drain_claim(conn, run, batches)

        update = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_attempts = %(attempts)s" in call.args[0]
        )
        self.assertFalse(update.args[1]["terminal"])
        self.assertEqual(update.args[1]["retry_delay"], 30)

    def test_terminal_run_artifact_recovery_completes_without_changing_outcome(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        self.assertTrue(_reconcile_terminal_artifact_publication(conn, 87))

        statement = cursor.execute.call_args.args[0]
        self.assertIn("live_publication_status = 'complete'", statement)
        self.assertIn("t.status IN ('succeeded', 'failed', 'canceled'", statement)
        self.assertNotIn("SET status =", statement)
        self.assertIn("artifact_stream.stream_id LIKE 'artifact-v2-%%'", statement)

    def test_finishing_run_completes_only_after_remote_cursors_metrics_and_artifact(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                },
                "outcome": "accepted",
                "promotion_revision": 1,
                "promotion_json": {
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_step": 8_000_000,
                },
            },
        ]
        cursor.fetchall.return_value = [
            {
                "stream_id": "train-7",
                "final_sequence": 3,
                "published_sequence": 3,
            }
        ]
        cursor.rowcount = 1
        artifact = SimpleNamespace(
            aliases=["promoted"],
            metadata={
                "checkpoint_sha256": "a" * 64,
                "publication_role": "promotion",
                "promotion_revision": 1,
            },
        )
        remote = SimpleNamespace(
            state="finished",
            summary={
                "_rlab_telemetry_cursors": {"train-7": 3},
                "eval/acceptance/pass": 0.0,
                "rlab/goal/outcome": "accepted",
                LEADER_CHECKPOINT_ACCEPTANCE_PASS: 1.0,
                LEADER_CHECKPOINT_STEP: 8_000_000,
            },
            logged_artifacts=lambda: [artifact],
        )

        with mock.patch(
            "rlab.fleet_wandb_publisher._wandb_api_run",
            return_value=remote,
        ):
            self.assertTrue(finalize_finishing_run(conn, 7))

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        complete = [
            statement
            for statement in statements
            if "live_publication_status = 'complete'" in statement
        ]
        self.assertEqual(len(complete), 1)

    def test_wandb_failure_remains_retryable_finalization_work(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                },
                "outcome": "accepted",
                "promotion_revision": 1,
                "promotion_json": {"checkpoint_sha256": "a" * 64},
            },
        ]
        cursor.fetchall.return_value = [
            {
                "stream_id": "train-7",
                "final_sequence": 3,
                "published_sequence": 3,
            }
        ]

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                side_effect=RuntimeError("W&B unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "W&B unavailable"),
        ):
            finalize_finishing_run(conn, 7)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(
            any("live_publication_next_retry_at" in statement for statement in statements)
        )
        self.assertFalse(
            any("live_publication_status = 'complete'" in statement for statement in statements)
        )

    def test_stale_remote_summary_is_restamped_from_database_promotion(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "live_publication_attempts": 0,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                    "selection_rank": [
                        "max(eval/full/outcome/success/rate/min)",
                        "max(eval/full/outcome/success/rate/mean)",
                        "min(leader/checkpoint/steps_to_goal)",
                        "max(eval/full/episode/return/mean)",
                    ],
                },
                "outcome": "accepted",
                "promotion_revision": 1,
                "promotion_json": {
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_uri": "s3://bucket/8000000/model.zip",
                    "raw_metrics": {
                        "checkpoint_step": 8_000_000,
                        "eval/full/outcome/success/rate/min": 1.0,
                        "eval/full/outcome/success/rate/mean": 1.0,
                        "eval/full/episode/return/mean": 3144.17,
                    },
                },
            },
        ]
        cursor.fetchall.return_value = [
            {"stream_id": "train-7", "final_sequence": 3, "published_sequence": 3}
        ]
        cursor.rowcount = 1
        artifact = SimpleNamespace(
            aliases=["promoted"],
            metadata={
                "checkpoint_sha256": "a" * 64,
                "publication_role": "promotion",
                "promotion_revision": 1,
            },
        )
        remote = SimpleNamespace(
            state="finished",
            summary={
                "_rlab_telemetry_cursors": {"train-7": 3},
                EVAL_ACCEPTANCE_PASS: 0.0,
            },
            logged_artifacts=lambda: [artifact],
        )
        projector = SimpleNamespace(run=remote, close=lambda: None)

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                return_value=remote,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ),
        ):
            self.assertTrue(finalize_finishing_run(conn, 7))

        self.assertEqual(remote.summary["rlab/goal/outcome"], "accepted")
        self.assertEqual(remote.summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS], 1.0)
        self.assertEqual(remote.summary[LEADER_CHECKPOINT_STEP], 8_000_000)

    def test_persistent_artifact_mismatch_is_bounded_and_precise(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "live_publication_attempts": 2,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                    "selection_rank": ["max(eval/full/outcome/success/rate/min)"],
                },
                "outcome": "accepted",
                "promotion_revision": 1,
                "promotion_json": {
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_uri": "s3://bucket/model.zip",
                    "raw_metrics": {
                        "checkpoint_step": 8_000_000,
                        "eval/full/outcome/success/rate/min": 1.0,
                        "eval/full/outcome/success/rate/mean": 1.0,
                    },
                },
            },
        ]
        cursor.fetchall.return_value = [
            {"stream_id": "train-7", "final_sequence": 3, "published_sequence": 3}
        ]
        remote = SimpleNamespace(
            state="finished",
            summary={
                "_rlab_telemetry_cursors": {"train-7": 3},
                "rlab/goal/outcome": "accepted",
                LEADER_CHECKPOINT_ACCEPTANCE_PASS: 1.0,
                LEADER_CHECKPOINT_STEP: 8_000_000,
            },
            logged_artifacts=lambda: [],
        )
        projector = SimpleNamespace(run=remote, close=lambda: None)

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                return_value=remote,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.WandbProjector.resume",
                return_value=projector,
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.time.monotonic",
                side_effect=[0.0, 21.0],
            ),
            self.assertRaisesRegex(WandbFinalizationVerificationError, "artifact"),
        ):
            finalize_finishing_run(conn, 7)

        failure_update = next(
            call
            for call in cursor.execute.call_args_list
            if "live_publication_attempts = %(attempts)s" in call.args[0]
        )
        self.assertEqual(failure_update.args[1]["attempts"], 3)
        self.assertTrue(failure_update.args[1]["terminal"])
        self.assertIn("artifact", failure_update.args[1]["error"])

    def test_late_evaluations_keep_their_checkpoint_steps_without_internal_step(self) -> None:
        run = FakeRun()
        run.log({"global_step": 300, "train/throughput/loop_fps": 10.0})
        with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace()}):
            for checkpoint_step in (100, 200):
                project_payload_to_run(
                    run,
                    {
                        "train_config": {"wandb_run_id": "rlab-test"},
                        "purpose": "screen",
                        "checkpoint_uri": f"s3://bucket/{checkpoint_step}/model.zip",
                        "checkpoint_step": checkpoint_step,
                        "decision": {
                            "metrics": {
                                "global_step": checkpoint_step,
                                "eval/screen/checkpoint/step": checkpoint_step,
                                "eval/screen/outcome/success/rate/min": 1.0,
                            }
                        },
                    },
                )

        self.assertEqual([row["global_step"] for row in run.logged], [300, 100, 200])

    def test_publishes_batched_frame_without_overriding_wandb_internal_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/episode/return/shaped/mean": 12.5},
                step=2048,
                source="train",
            )
            run = FakeRun()

            count = publish_pending_frames(
                store,
                run,
                args=SimpleNamespace(),
                config=SimpleNamespace(),
                limit=100,
            )

            self.assertEqual(count, 1)
            self.assertEqual(len(run.logged), 1)
            self.assertEqual(run.logged[0]["global_step"], 2048.0)
            self.assertEqual(run.logged[0]["train/episode/return/shaped/mean"], 12.5)
            self.assertEqual(store.phase_counts()["telemetry:published"], 1)
            self.assertEqual(store.telemetry_health()["published_step"], 2048)

    def test_interrupted_publish_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/throughput/loop_fps": 1},
                step=10,
                source="train",
            )
            frame = store.pending_metric_frames()[0]
            self.assertTrue(store.claim_metric_frame(int(frame["id"])))

            self.assertEqual(store.reset_interrupted_metric_frames(), 1)
            self.assertEqual(store.pending_metric_frames()[0]["status"], "failed_retryable")

    def test_preview_failure_never_blocks_scalars_and_is_abandoned_after_two_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.enqueue_event(
                kind="checkpoint_preview",
                payload={
                    "url": "https://preview.example/checkpoint.mp4",
                    "checkpoint_step": 10,
                    "lane_count": 2,
                    "duration_seconds": 30,
                },
                step=10,
                source="modal_checkpoint_eval",
            )
            store.append_metrics(
                {"train/throughput/loop_fps": 7.0},
                step=11,
                source="train",
            )
            self.assertEqual(store.pending_metric_frames()[0]["kind"], "history")

            class FailingPreviewRun(FakeRun):
                def log(self, payload: dict[str, object]) -> None:
                    if "eval/screen/preview" in payload:
                        raise RuntimeError("HTML unavailable")
                    super().log(payload)

            run = FailingPreviewRun()
            fake_wandb = SimpleNamespace(Html=FakeHtml)
            with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
                self.assertEqual(
                    publish_pending_frames(
                        store,
                        run,
                        args=SimpleNamespace(),
                        config=SimpleNamespace(),
                        limit=100,
                    ),
                    1,
                )
                self.assertEqual(run.logged[0]["global_step"], 11.0)
                self.assertEqual(store.pending_metric_frames()[0]["kind"], "checkpoint_preview")
                self.assertEqual(
                    publish_pending_frames(
                        store,
                        run,
                        args=SimpleNamespace(),
                        config=SimpleNamespace(),
                        limit=100,
                    ),
                    0,
                )

            self.assertEqual(store.pending_metric_frames(), [])
            self.assertEqual(store.phase_counts()["telemetry:failed_terminal"], 1)

    def test_preview_logs_external_html_at_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.enqueue_event(
                kind="checkpoint_preview",
                payload={
                    "url": "https://preview.example/checkpoint.mp4",
                    "checkpoint_step": 500000,
                    "passed": True,
                    "lane_count": 2,
                    "duration_seconds": 30,
                },
                step=500000,
                source="modal_checkpoint_eval",
            )
            run = FakeRun()
            with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace(Html=FakeHtml)}):
                count = publish_pending_frames(
                    store,
                    run,
                    args=SimpleNamespace(),
                    config=SimpleNamespace(),
                    limit=100,
                )

        self.assertEqual(count, 1)
        self.assertEqual(run.logged[0]["global_step"], 500000)
        media = run.logged[0]["eval/screen/preview"]
        self.assertIn("https://preview.example/checkpoint.mp4", media.data)
        self.assertEqual(media.kwargs["inject"], False)
        self.assertEqual(media.kwargs["data_is_not_path"], True)


if __name__ == "__main__":
    unittest.main()
