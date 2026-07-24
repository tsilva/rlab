from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest

from rlab.metric_store import MetricStore, metric_store_path


def test_metric_outbox_deduplicates_stable_events_and_tracks_latest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "rlab.sqlite")
        store.init()
        first = store.append_metrics(
            {"train/episode/return/shaped/mean": 1.0},
            step=10,
            source="learner",
        )
        second = store.append_metrics(
            {"train/episode/return/shaped/mean": 1.0},
            step=10,
            source="learner",
        )

        assert first == second == 1
        assert len(store.pending_metric_frames()) == 1
        assert store.latest_metric("train/episode/return/shaped/mean") == 1.0


def test_frame_publish_failure_is_retryable_and_success_drains_outbox() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "rlab.sqlite")
        store.init()
        store.append_metrics({"global_step": 10}, step=10, source="learner")
        frame = store.pending_metric_frames()[0]
        assert store.claim_metric_frame(frame["id"])
        store.mark_metric_frame_failed(frame["id"], "offline")
        assert store.pending_metric_frames()[0]["attempts"] == 1
        assert store.claim_metric_frame(frame["id"])
        store.mark_metric_frame_published(frame["id"], step=10)
        assert store.metric_outbox_stats()["frames"] == 0
        assert store.outbox_health()["published_step"] == 10


def test_interrupted_publish_claim_is_recovered() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "rlab.sqlite")
        store.init()
        store.append_metrics({"global_step": 1}, step=1, source="learner")
        frame_id = store.pending_metric_frames()[0]["id"]
        assert store.claim_metric_frame(frame_id)
        assert store.reset_interrupted_metric_frames() == 1
        assert store.pending_metric_frames()[0]["status"] == "failed_retryable"


def test_recovery_manifest_is_secret_free_and_immutable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "rlab.sqlite")
        store.init()
        digest = store.register_recovery_manifest(
            {"version": "supervisor-sqlite-recovery-v1", "run_id": "run"}
        )
        assert len(digest) == 64
        assert (
            store.register_recovery_manifest(
                {"version": "supervisor-sqlite-recovery-v1", "run_id": "run"}
            )
            == digest
        )
        with pytest.raises(ValueError, match="secret-like"):
            store.register_recovery_manifest({"api_key": "forbidden"})
        with pytest.raises(RuntimeError, match="conflicts"):
            store.register_recovery_manifest(
                {"version": "supervisor-sqlite-recovery-v1", "run_id": "other"}
            )


def test_checkpoint_rows_are_idempotent_and_record_eval_policy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "model_100_steps.zip"
        checkpoint.write_bytes(b"checkpoint")
        store = MetricStore(metric_store_path(root))
        store.init()
        first = store.record_checkpoint(
            run_name="run",
            kind="checkpoint",
            step=100,
            path=checkpoint,
            metadata_path=None,
            eval_required=True,
        )
        replay = store.record_checkpoint(
            run_name="run",
            kind="checkpoint",
            step=100,
            path=checkpoint,
            metadata_path=None,
            eval_required=True,
        )
        assert replay == first
        assert store.checkpoints()[0]["eval_required"] == 1
        with pytest.raises(ValueError, match="eval_required"):
            store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=100,
                path=checkpoint,
                metadata_path=None,
                eval_required=False,
            )


def test_checkpoint_publication_status_is_local_delivery_state() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "model.zip"
        checkpoint.write_bytes(b"checkpoint")
        store = MetricStore(metric_store_path(root))
        store.init()
        checkpoint_id = store.record_checkpoint(
            run_name="run",
            kind="final",
            step=200,
            path=checkpoint,
            metadata_path=None,
            eval_required=True,
        )
        store.mark_checkpoint_upload_failed(checkpoint_id, "network")
        assert store.checkpoints()[0]["upload_status"] == "failed_retryable"
        store.mark_checkpoint_uploaded(
            checkpoint_id,
            "https://models.example/runs/run/checkpoints/model.zip",
        )
        assert store.checkpoints()[0]["upload_status"] == "uploaded"


def test_wal_accepts_concurrent_learner_writes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        store = MetricStore(Path(temporary) / "rlab.sqlite", timeout=5)
        store.init()
        failures: list[BaseException] = []

        def writer(offset: int) -> None:
            try:
                for step in range(offset, offset + 20):
                    store.append_metrics(
                        {"train/episode/return/shaped/mean": float(step)},
                        step=step,
                        source=f"learner-{offset}",
                    )
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=writer, args=(index * 100,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert len(store.pending_metric_frames(limit=100)) == 80
