from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from rlab.artifacts import (
    checkpoint_step,
    load_model_metadata,
    model_metadata_path,
    write_model_metadata,
)
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.env_registry import resolve_env_provider
from rlab.metric_names import (
    CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP,
    CHECKPOINT_EVAL_CANDIDATE_EPISODES,
    CHECKPOINT_EVAL_CANDIDATE_PASS,
    CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    SEED_PROTOCOL,
    stage_job_descriptor,
)
from rlab.modal_eval_storage import ObjectStore, file_sha256, object_store_base_uri
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.train_config import materialized_train_args


MAX_UPLOAD_ATTEMPTS = 3


def reconcile_orphan_models(store: MetricStore, args, run_dir: Path) -> int:
    known = {Path(str(row["path"])).resolve() for row in store.checkpoints()}
    config = None
    recovered = 0
    for model_path in sorted(run_dir.glob("*.zip")):
        if model_path.name.startswith(".") or model_path.resolve() in known:
            continue
        metadata = load_model_metadata(model_path)
        kind = str(metadata.get("kind") or "")
        if not kind:
            kind = (
                "final"
                if model_path.stem == "final_model"
                else "interrupted"
                if "interrupted" in model_path.stem
                else "checkpoint"
            )
        step_value = metadata.get("checkpoint_step")
        step = int(step_value) if step_value is not None else checkpoint_step(model_path)
        metadata_path = model_metadata_path(model_path)
        if not metadata_path.is_file():
            if config is None:
                config = resolve_env_config(env_config_from_args(args, include_states=True))
            metadata_path = write_model_metadata(
                model_path,
                args,
                config,
                kind,
                checkpoint_step_value=step,
            )
        store.record_checkpoint(
            run_name=str(getattr(args, "run_name", run_dir.name)),
            kind=kind,
            step=step,
            path=model_path,
            metadata_path=metadata_path,
            sha256=None,
        )
        recovered += 1
    return recovered


def _storage_uri(args) -> str:
    configured = str(getattr(args, "wandb_artifact_storage_uri", "") or "").strip()
    return configured or object_store_base_uri()


def _max_steps(args) -> int:
    explicit = int(getattr(args, "post_train_eval_max_steps", 0) or 0)
    if explicit > 0:
        return explicit
    environment = getattr(args, "checkpoint_eval_environment", None)
    if isinstance(environment, dict):
        task = environment.get("task")
        if isinstance(task, dict):
            termination = task.get("termination")
            if isinstance(termination, dict):
                value = int(termination.get("max_episode_steps") or 0)
                if value > 0:
                    return value
    raise ValueError("checkpoint eval max steps are not materialized")


def _eval_payload(args) -> dict[str, Any]:
    environment = getattr(args, "checkpoint_eval_environment", None)
    stages = getattr(args, "checkpoint_eval_stages", None)
    asset = getattr(args, "checkpoint_eval_asset_manifest", None)
    if not isinstance(environment, dict):
        raise ValueError("Modal checkpoint eval requires a materialized environment")
    provider = str(getattr(args, "env_provider", "") or "").strip()
    requires_rom_asset = (
        resolve_env_provider(provider).uses_stable_retro_roms if provider else True
    )
    if requires_rom_asset and not isinstance(asset, dict):
        raise ValueError("Modal checkpoint eval requires a materialized asset manifest")
    if not requires_rom_asset:
        asset = None
    if stages is None:
        stages = []
    if not isinstance(stages, list):
        raise ValueError("Modal checkpoint eval stages must be a list")
    return {
        "environment": environment,
        "stages": stages,
        "n_envs": int(getattr(args, "checkpoint_eval_n_envs", 1)),
        "max_steps": _max_steps(args),
        "seed": int(getattr(args, "checkpoint_eval_seed", DEFAULT_EVAL_SEED)),
        "seed_protocol": str(getattr(args, "checkpoint_eval_seed_protocol", SEED_PROTOCOL)),
        "asset": asset,
        "promotion_episodes": int(getattr(args, "post_train_eval_episodes", 100)),
    }


def checkpoint_announcement(
    args,
    row: dict[str, Any],
    *,
    sha256: str,
    model_uri: str,
    metadata_uri: str,
    metadata_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "train_job_id": int(getattr(args, "queue_train_job_id", 0)),
        "ledger_id": int(row["id"]),
        "kind": str(row["kind"]),
        "step": int(row.get("step") or 0),
        "sha256": sha256,
        "model_uri": model_uri,
        "metadata_uri": metadata_uri,
        "metadata_sha256": metadata_sha256,
        "runtime_image_ref": str(getattr(args, "runtime_image_ref", "")),
        "wandb_run_id": str(getattr(args, "wandb_run_id", "")),
        "eval": _eval_payload(args),
    }


def process_upload(
    store: MetricStore, object_store: ObjectStore, args, row: dict[str, Any]
) -> bool:
    checkpoint_id = int(row["id"])
    if not store.claim_artifact_upload(checkpoint_id):
        return False
    model_path = Path(str(row["path"]))
    metadata_path = Path(str(row.get("metadata_path") or ""))
    try:
        sha256 = str(row.get("sha256") or "") or file_sha256(model_path)
        store.set_checkpoint_sha256(checkpoint_id, sha256)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
        metadata_sha = file_sha256(metadata_path)
        prefix = f"checkpoints/{int(getattr(args, 'queue_train_job_id', 0))}/{sha256}"
        model_uri = object_store.put_file(
            f"{prefix}/model.zip", model_path, sha256=sha256, content_type="application/zip"
        )
        metadata_uri = object_store.put_file(
            f"{prefix}/{metadata_sha}/metadata.json",
            metadata_path,
            sha256=metadata_sha,
            content_type="application/json",
        )
        announcement = checkpoint_announcement(
            args,
            row,
            sha256=sha256,
            model_uri=model_uri,
            metadata_uri=metadata_uri,
            metadata_sha256=metadata_sha,
        )
        if str(getattr(args, "telemetry_transport", "legacy_local")) == "neon_mailbox_v1":
            from rlab.telemetry_mailbox import WorkerMailbox

            WorkerMailbox.from_env().append_event(
                "checkpoint_ready",
                announcement,
                event_id=f"checkpoint-ready:{announcement['train_job_id']}:{checkpoint_id}",
            )
        else:
            object_store.put_json(
                f"artifact-announcements/{announcement['train_job_id']}/{checkpoint_id:08d}.json",
                announcement,
                create_only=True,
            )
        store.mark_artifact_uploaded(checkpoint_id, artifact_ref=None, storage_uri=model_uri)
        return True
    except Exception as exc:
        attempts = int(row.get("attempts") or 0) + 1
        if attempts >= MAX_UPLOAD_ATTEMPTS:
            tombstone = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "train_job_id": int(getattr(args, "queue_train_job_id", 0)),
                "ledger_id": checkpoint_id,
                "kind": "tombstone",
                "error": repr(exc)[:1000],
            }
            tombstone_key = (
                f"artifact-announcements/{tombstone['train_job_id']}/{checkpoint_id:08d}.json"
            )
            try:
                if str(getattr(args, "telemetry_transport", "legacy_local")) == ("neon_mailbox_v1"):
                    from rlab.telemetry_mailbox import WorkerMailbox

                    WorkerMailbox.from_env().append_event(
                        "checkpoint_tombstone",
                        tombstone,
                        event_id=f"checkpoint-tombstone:{tombstone['train_job_id']}:{checkpoint_id}",
                    )
                elif object_store.get_json_optional(tombstone_key) is None:
                    object_store.put_json(tombstone_key, tombstone, create_only=True)
            except Exception as tombstone_exc:
                store.mark_artifact_failed(
                    checkpoint_id,
                    f"upload failed: {exc!r}; tombstone failed: {tombstone_exc!r}",
                )
            else:
                store.mark_artifact_terminal_failure(checkpoint_id, error=repr(exc))
        else:
            store.mark_artifact_failed(checkpoint_id, repr(exc))
        print(f"checkpoint coordinator upload failed id={checkpoint_id}: {exc}", flush=True)
        return False


def import_decisions(store: MetricStore, object_store: ObjectStore, args) -> int:
    imported = 0
    for row in store.checkpoints():
        if str(row.get("kind")) != "checkpoint" or not row.get("sha256"):
            continue
        artifact = checkpoint_announcement(
            args,
            row,
            sha256=str(row["sha256"]),
            model_uri="",
            metadata_uri="",
            metadata_sha256=file_sha256(Path(str(row["metadata_path"]))),
        )
        for stage_index, _stage in enumerate(artifact["eval"]["stages"]):
            descriptor = stage_job_descriptor(artifact, stage_index=stage_index)
            local_stage_status = store.checkpoint_eval_stage_status(
                int(row["id"]), str(descriptor["stage_name"])
            )
            if local_stage_status == "succeeded":
                continue
            if local_stage_status in {"failed_gate", "skipped_stale"}:
                break
            decision = object_store.get_json_optional(
                f"eval-decisions/{artifact['train_job_id']}/{descriptor['job_key']}.json"
            )
            if decision is None:
                break
            if str(decision.get("status") or "") == "skipped_stale":
                store.apply_modal_eval_skip(
                    int(row["id"]),
                    stage_name=str(descriptor["stage_name"]),
                    stage_index=stage_index,
                    episodes=int(descriptor["contract"]["episodes"]),
                    n_envs=int(descriptor["contract"]["n_envs"]),
                )
                imported += 1
                break
            metrics = decision.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("Modal eval decision is missing metrics")
            if descriptor["candidate_stop"] and bool(decision.get("passed")):
                metrics.update(
                    {
                        CHECKPOINT_EVAL_CANDIDATE_PASS: 1.0,
                        CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX: float(stage_index),
                        CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP: float(row.get("step") or 0),
                        CHECKPOINT_EVAL_CANDIDATE_EPISODES: float(
                            descriptor["contract"]["episodes"]
                        ),
                    }
                )
            store.apply_modal_eval_decision(
                int(row["id"]),
                stage_name=str(descriptor["stage_name"]),
                stage_index=stage_index,
                episodes=int(descriptor["contract"]["episodes"]),
                n_envs=int(descriptor["contract"]["n_envs"]),
                metrics=metrics,
                passed=bool(decision.get("passed")),
                candidate_stop=bool(descriptor["candidate_stop"]),
                publish=(
                    str(getattr(args, "telemetry_transport", "legacy_local")) != "neon_mailbox_v1"
                ),
            )
            preview = decision.get("preview")
            if (
                str(descriptor["purpose"]) == "screen"
                and str(getattr(args, "telemetry_transport", "legacy_local")) != "neon_mailbox_v1"
                and isinstance(preview, dict)
                and str(preview.get("status")) == "succeeded"
            ):
                try:
                    store.enqueue_event(
                        kind="checkpoint_preview",
                        payload={
                            "url": str(preview.get("public_url") or ""),
                            "checkpoint_step": int(row.get("step") or 0),
                            "passed": bool(decision.get("passed")),
                            "lane_count": int(preview.get("lane_count") or 0),
                            "duration_seconds": float(preview.get("duration_seconds") or 0.0),
                            "width": int(preview.get("width") or 0),
                            "height": int(preview.get("height") or 0),
                            "observation_source": "preprocessed_policy_observation",
                        },
                        step=int(row.get("step") or 0),
                        source="modal_checkpoint_eval",
                        event_id=f"checkpoint-preview:{decision['job_key']}",
                    )
                except Exception as exc:
                    print(f"checkpoint preview enqueue failed: {exc}", flush=True)
            imported += 1
            if not bool(decision.get("passed")):
                break
    return imported


def write_complete_marker(store: MetricStore, object_store: ObjectStore, args) -> bool:
    rows = store.checkpoints()
    phases = store.phase_counts()
    pending = sum(
        count
        for key, count in phases.items()
        if key.startswith("artifacts:")
        and key not in {"artifacts:uploaded", "artifacts:failed_terminal"}
    )
    if pending:
        return False
    train_job_id = int(getattr(args, "queue_train_job_id", 0))
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "train_job_id": train_job_id,
        "last_ledger_id": max((int(row["id"]) for row in rows), default=0),
        "checkpoint_count": len(rows),
    }
    if str(getattr(args, "telemetry_transport", "legacy_local")) == "neon_mailbox_v1":
        from rlab.telemetry_mailbox import WorkerMailbox

        try:
            WorkerMailbox.from_env().append_event(
                "checkpoint_stream_closed",
                payload,
                event_id=f"checkpoint-stream-closed:{train_job_id}",
            )
        except Exception as exc:
            print(f"checkpoint stream closure delivery failed; retrying: {exc}", flush=True)
            return False
    else:
        object_store.put_json(
            f"artifact-announcements/{train_job_id}/complete.json",
            payload,
            create_only=True,
        )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate queue checkpoint storage and Modal decisions"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path, required=True)
    exit_mode = parser.add_mutually_exclusive_group(required=True)
    exit_mode.add_argument("--stop-file", type=Path)
    exit_mode.add_argument("--drain-and-exit", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    try:
        os.nice(10)
    except AttributeError, OSError:
        pass
    args = materialized_train_args(cli.train_config_json)
    store = MetricStore(metric_store_path(cli.run_dir))
    store.init()
    object_store = ObjectStore(_storage_uri(args))
    while True:
        activity = reconcile_orphan_models(store, args, cli.run_dir)
        for row in store.pending_artifact_uploads(limit=max(1, cli.limit)):
            activity += int(process_upload(store, object_store, args, row))
        if str(getattr(args, "telemetry_transport", "legacy_local")) != "neon_mailbox_v1":
            activity += import_decisions(store, object_store, args)
        drain_requested = cli.drain_and_exit or (
            cli.stop_file is not None and cli.stop_file.exists()
        )
        if drain_requested and write_complete_marker(store, object_store, args):
            return 0
        if not activity:
            time.sleep(max(0.25, cli.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
