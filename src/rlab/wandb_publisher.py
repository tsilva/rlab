from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from rlab.artifacts import (
    build_s3_artifact_uri,
    init_wandb,
    log_wandb_model_artifact,
    wandb_artifact_storage_uri,
    write_wandb_url,
)
from rlab.file_utils import file_sha256
from rlab.checkpoint_eval_worker import (
    add_eval_by_start_table,
    log_checkpoint_eval_metrics,
    update_best_checkpoint_summary,
)
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.env_metadata import env_config_from_config_dict
from rlab.metric_store import MetricStore, metric_store_path
from rlab.metric_names import (
    EVAL_SCREEN_PREVIEW,
    LEADER_CHECKPOINT_ACCEPTANCE_PASS,
    validate_metric_payload,
)
from rlab.modal_eval_storage import ObjectStore, object_store_base_uri
from rlab.policy_bundle import (
    CHECKPOINT_FILENAME,
    MODEL_FILENAME,
    RECIPE_FILENAME,
    load_policy_bundle,
    model_document_as_metadata,
)
from rlab.train_config import materialized_train_args
from rlab.wandb_utils import (
    configure_wandb_metrics,
    load_wandb_env,
    resolve_wandb_namespace,
)
from rlab.wandb_artifacts import (
    artifact_collection_name,
    artifact_write_aliases,
    artifact_write_ref,
)


class WandbProjector:
    """Own one resumed W&B run for both live outbox and final projections."""

    def __init__(self, run, *, run_dir: str | None = None) -> None:
        self.run = run
        self.run_dir = run_dir

    @classmethod
    def start_live(cls, args, *, run_dir: str, config) -> WandbProjector:
        return cls(init_wandb(args, run_dir, config), run_dir=run_dir)

    @classmethod
    def resume(
        cls,
        train_config: Mapping[str, Any],
        *,
        allow_create: bool = False,
        update_finish_state: bool = True,
    ) -> WandbProjector:
        run_id = str(train_config.get("wandb_run_id") or "")
        if not run_id:
            raise ValueError("W&B projection requires the producing run id")
        load_wandb_env()
        import wandb

        entity, project = resolve_wandb_namespace(
            train_config.get("wandb_entity"),
            train_config.get("wandb_project"),
            str(train_config.get("game") or ""),
            env_provider=train_config.get("env_provider"),
        )
        raw_tags = train_config.get("wandb_tags") or ()
        tags = (
            [part.strip() for part in str(raw_tags).split(",") if part.strip()]
            if isinstance(raw_tags, str)
            else [str(tag) for tag in raw_tags]
        )
        run = configure_wandb_metrics(
            wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                resume="allow" if allow_create else "must",
                mode=str(train_config.get("wandb_mode") or "online"),
                name=str(train_config.get("run_name") or "") or None,
                group=str(train_config.get("wandb_group") or "") or None,
                tags=tags,
                config=dict(train_config) if allow_create else None,
                settings=wandb.Settings(x_update_finish_state=update_finish_state),
            )
        )
        return cls(run)

    def close(self) -> None:
        if self.run_dir is not None:
            write_wandb_url(self.run, self.run_dir)
        if self.run is not None:
            self.run.finish()


def project_payload_to_run(
    run,
    payload: Mapping[str, Any],
    *,
    allow_artifact_references: bool = True,
) -> Any | None:
    train_config = dict(payload["train_config"])
    run_id = str(train_config["wandb_run_id"])
    import wandb

    projection_kind = str(payload.get("projection_kind") or "evaluation")
    if projection_kind == "artifact_reference":
        if not allow_artifact_references:
            raise ValueError("mailbox telemetry does not project W&B artifact references")
        if bool(train_config.get("no_wandb_artifacts", False)):
            return None
        kind = str(payload["artifact_kind"])
        checkpoint_step = int(payload["checkpoint_step"])
        publication_schema = str(payload.get("artifact_publication_schema") or "")
        checkpoint_uri = str(payload["checkpoint_uri"])
        versioned_bundle = bool(payload.get("recipe_uri"))
        raw_aliases = payload.get("artifact_aliases")
        aliases = (
            [str(alias) for alias in raw_aliases]
            if isinstance(raw_aliases, list) and raw_aliases
            else artifact_write_aliases(kind, checkpoint_step)
        )
        with tempfile.TemporaryDirectory(prefix="rlab-wandb-artifact-") as temporary:
            root = Path(temporary)
            members: dict[str, dict[str, int | str]] | None = None
            if publication_schema == "v3":
                if payload.get("content_mode") != "wandb_native_v1" or not versioned_bundle:
                    raise ValueError("v3 artifact publication requires a native policy bundle")
                store = ObjectStore(object_store_base_uri())
                member_specs = (
                    (
                        CHECKPOINT_FILENAME,
                        checkpoint_uri,
                        str(payload["checkpoint_sha256"]),
                    ),
                    (
                        MODEL_FILENAME,
                        str(payload["metadata_uri"]),
                        str(payload["metadata_sha256"]),
                    ),
                    (
                        RECIPE_FILENAME,
                        str(payload["recipe_uri"]),
                        str(payload["recipe_sha256"]),
                    ),
                )
                members = {}
                for filename, uri, expected_sha256 in member_specs:
                    data = store.get_bytes(uri)
                    observed_sha256 = hashlib.sha256(data).hexdigest()
                    if observed_sha256 != expected_sha256:
                        raise ValueError(f"W&B native artifact member hash mismatch: {filename}")
                    member_path = root / filename
                    member_path.write_bytes(data)
                    members[filename] = {
                        "sha256": observed_sha256,
                        "size_bytes": len(data),
                    }
                bundle = load_policy_bundle(root)
                model_metadata = model_document_as_metadata(bundle.model)
            else:
                model_metadata = dict(payload["model_metadata"])
            if not isinstance(model_metadata.get("training_metadata"), dict):
                raise ValueError("artifact projection requires checkpoint training_metadata")
            artifact = wandb.Artifact(
                artifact_collection_name(kind, run_id=run_id),
                type="model",
                metadata={
                    **model_metadata,
                    "source_filename": model_metadata.get("filename", ""),
                    "filename": CHECKPOINT_FILENAME,
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "metadata_sha256": payload["metadata_sha256"],
                    "checkpoint_step": checkpoint_step,
                    "metadata_uri": payload["metadata_uri"],
                    **(
                        {
                            "recipe_uri": payload["recipe_uri"],
                            "recipe_sha256": payload["recipe_sha256"],
                        }
                        if versioned_bundle
                        else {}
                    ),
                    "artifact_storage_uri": checkpoint_uri,
                    **(
                        {
                            "artifact_publication_schema": publication_schema,
                            "content_mode": payload.get("content_mode"),
                            "artifact_members": members,
                            "train_job_id": int(payload["train_job_id"]),
                            "ledger_id": int(payload["ledger_id"]),
                            "artifact_kind": kind,
                            "publication_role": str(payload["publication_role"]),
                            "promotion_revision": int(payload.get("promotion_revision") or 0),
                            "publication_stream_id": str(payload["publication_stream_id"]),
                            "announcement_sha256": str(payload["announcement_sha256"]),
                        }
                        if publication_schema in {"v2", "v3"}
                        else {}
                    ),
                },
            )
            if publication_schema == "v3":
                for filename in (CHECKPOINT_FILENAME, MODEL_FILENAME, RECIPE_FILENAME):
                    artifact.add_file(str(root / filename), name=filename)
            elif versioned_bundle:
                artifact.add_reference(checkpoint_uri, name=CHECKPOINT_FILENAME)
                artifact.add_reference(str(payload["metadata_uri"]), name=MODEL_FILENAME)
                artifact.add_reference(str(payload["recipe_uri"]), name=RECIPE_FILENAME)
            else:
                artifact.add_reference(checkpoint_uri)
            logged_artifact = run.log_artifact(artifact, aliases=aliases)
            if publication_schema == "v3":
                wait = getattr(logged_artifact, "wait", None)
                if callable(wait):
                    wait()
            return logged_artifact
    decision = dict(payload["decision"])
    purpose = str(payload["purpose"])
    checkpoint_uri = str(payload["checkpoint_uri"])
    schema_version = int(train_config.get("metrics_schema_version", 4) or 4)
    if purpose == "promotion":
        environment = train_config.get("checkpoint_eval_environment")
        if not isinstance(environment, dict):
            raise ValueError("W&B projection is missing the materialized environment")
        config = env_config_from_config_dict(environment)
        if config is None:
            raise ValueError("W&B projection environment is invalid")
        log_checkpoint_eval_metrics(
            run,
            args=SimpleNamespace(**train_config),
            metrics=dict(decision["raw_metrics"]),
            checkpoint_path=checkpoint_uri,
            checkpoint_step_value=int(payload["checkpoint_step"]),
            artifact_ref=checkpoint_uri,
            eval_source="modal",
            config=resolve_env_config(config),
            update_leader=bool(payload.get("canonical_promotion", False)),
            force_leader=bool(payload.get("canonical_promotion", False)),
        )
    else:
        projected_metrics = dict(decision["metrics"])
        if purpose == "acceptance" and bool(decision.get("passed", False)):
            add_eval_by_start_table(
                projected_metrics,
                metrics=dict(decision["raw_metrics"]),
                checkpoint_step=int(payload["checkpoint_step"]),
            )
        run.log(projected_metrics)
        if purpose == "acceptance" and bool(payload.get("canonical_promotion", False)):
            if not bool(decision.get("passed", False)):
                raise ValueError("canonical acceptance projection must contain a passed decision")
            update_best_checkpoint_summary(
                run,
                metrics=dict(decision["raw_metrics"]),
                checkpoint_path=checkpoint_uri,
                checkpoint_step_value=int(payload["checkpoint_step"]),
                artifact_ref=checkpoint_uri,
                eval_source="modal:acceptance",
                selection_rank=train_config.get("selection_rank") or (),
                force=True,
                metrics_schema_version=schema_version,
                include_completion=schema_version == 4,
            )
            if schema_version == 4:
                run.summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS] = 1.0
            run.summary["rlab/goal/outcome"] = "accepted"
        preview = decision.get("preview")
        if (
            purpose == "screen"
            and isinstance(preview, Mapping)
            and str(preview.get("status") or "") == "succeeded"
        ):
            url = str(preview.get("public_url") or "")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("checkpoint preview URL must be absolute HTTPS")
            checkpoint_step = int(payload.get("checkpoint_step") or 0)
            caption = (
                f"checkpoint {checkpoint_step:,} · screen "
                f"{'passed' if bool(decision.get('passed')) else 'did not pass'} · "
                f"{int(preview.get('lane_count') or 0)} lanes · "
                f"{float(preview.get('duration_seconds') or 0.0):.1f}s · "
                "preprocessed observations"
            )
            markup = (
                '<div style="font-family:system-ui,sans-serif">'
                '<video controls muted loop playsinline preload="metadata" '
                'style="display:block;width:100%;max-width:720px;image-rendering:pixelated" '
                f'src="{html.escape(url, quote=True)}"></video>'
                f'<div style="margin-top:6px;font-size:12px">{html.escape(caption)}</div>'
                "</div>"
            )
            run.log(
                {
                    "global_step": checkpoint_step,
                    EVAL_SCREEN_PREVIEW: wandb.Html(
                        markup,
                        inject=False,
                        data_is_not_path=True,
                    ),
                }
            )
    return None


def project_payload(payload: Mapping[str, Any]) -> None:
    """Project one durable post-training payload through the legacy W&B owner."""

    train_config = dict(payload["train_config"])
    if not bool(train_config.get("wandb", False)):
        return
    projector = WandbProjector.resume(train_config)
    try:
        project_payload_to_run(projector.run, payload)
    finally:
        projector.close()


def artifact_ref(
    args: argparse.Namespace,
    kind: str,
    aliases: list[str],
    *,
    wandb_run=None,
) -> str | None:
    if not getattr(args, "wandb", False) or getattr(args, "no_wandb_artifacts", False):
        return None
    entity, project = resolve_wandb_namespace(
        getattr(args, "wandb_entity", None),
        getattr(args, "wandb_project", None),
        str(args.game),
        env_provider=getattr(args, "env_provider", None),
    )
    if not entity or not project:
        return None
    alias = aliases[-1] if aliases else "latest"
    run_id = getattr(wandb_run, "id", None) or getattr(args, "wandb_run_id", None)
    return artifact_write_ref(
        namespace=f"{entity}/{project}",
        kind=kind,
        run_id=run_id,
        alias=alias,
    )


def process_upload(
    *,
    store: MetricStore,
    args: argparse.Namespace,
    config,
    row: dict[str, Any],
    wandb_run=None,
) -> bool:
    checkpoint_id = int(row["id"])
    if not store.claim_artifact_upload(checkpoint_id):
        return False
    path = Path(str(row["path"]))
    kind = str(row["kind"])
    step = row.get("step")
    step_value = int(step) if step is not None else None
    aliases = artifact_write_aliases(kind, step_value)
    try:
        log_wandb_model_artifact(
            wandb_run,
            args,
            config,
            path,
            kind=kind,
            aliases=aliases,
            metric_step=step_value,
        )
        storage_uri = None
        if wandb_artifact_storage_uri(args):
            storage_uri = build_s3_artifact_uri(
                wandb_artifact_storage_uri(args),
                args,
                path,
                kind,
                run_id=getattr(wandb_run, "id", None) or getattr(args, "wandb_run_id", None),
                object_sha256=file_sha256(path),
            )
        store.mark_artifact_uploaded(
            checkpoint_id,
            artifact_ref=artifact_ref(args, kind, aliases, wandb_run=wandb_run),
            storage_uri=storage_uri,
        )
        return True
    except Exception as exc:
        store.mark_artifact_failed(checkpoint_id, repr(exc))
        print(f"artifact upload failed checkpoint_id={checkpoint_id}: {exc}", flush=True)
        return False


def _publish_frame(run, row: dict[str, Any], *, args, config) -> None:
    if run is None:
        return
    payload = json.loads(str(row["payload_json"]))
    kind = str(row["kind"])
    if kind == "history":
        validate_metric_payload(
            payload,
            schema_version=int(getattr(args, "metrics_schema_version", 4) or 4),
        )
        run.log(payload)
        return
    if kind == "histogram":
        import wandb

        converted: dict[str, object] = {"global_step": payload["global_step"]}
        for name, values in payload.get("histograms", {}).items():
            converted[str(name)] = wandb.Histogram(values)
        if len(converted) > 1:
            validate_metric_payload(
                converted,
                schema_version=int(getattr(args, "metrics_schema_version", 4) or 4),
            )
            run.log(converted)
        return
    if kind == "checkpoint_eval":
        log_checkpoint_eval_metrics(
            run,
            args=args,
            metrics=dict(payload["metrics"]),
            checkpoint_path=Path(str(payload["checkpoint_path"])),
            checkpoint_step_value=int(payload["checkpoint_step"]),
            artifact_ref=str(payload["artifact_ref"]),
            eval_source=str(payload.get("eval_source") or "async_worker"),
            config=config,
        )
        return
    if kind == "checkpoint_preview":
        import wandb

        url = str(payload.get("url") or "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("checkpoint preview URL must be absolute HTTPS")
        checkpoint_step = int(payload.get("checkpoint_step") or payload.get("global_step") or 0)
        passed = "passed" if bool(payload.get("passed")) else "did not pass"
        lanes = int(payload.get("lane_count") or 0)
        duration = float(payload.get("duration_seconds") or 0.0)
        caption = (
            f"checkpoint {checkpoint_step:,} · screen {passed} · {lanes} lanes · "
            f"{duration:.1f}s · preprocessed observations"
        )
        markup = (
            '<div style="font-family:system-ui,sans-serif">'
            '<video controls muted loop playsinline preload="metadata" '
            'style="display:block;width:100%;max-width:720px;image-rendering:pixelated" '
            f'src="{html.escape(url, quote=True)}"></video>'
            f'<div style="margin-top:6px;font-size:12px">{html.escape(caption)}</div>'
            "</div>"
        )
        run.log(
            {
                "global_step": checkpoint_step,
                EVAL_SCREEN_PREVIEW: wandb.Html(
                    markup,
                    inject=False,
                    data_is_not_path=True,
                ),
            }
        )
        return
    raise ValueError(f"unsupported telemetry frame kind: {kind}")


def publish_pending_frames(
    store: MetricStore,
    run,
    *,
    args,
    config,
    limit: int,
) -> int:
    published = 0
    for row in store.pending_metric_frames(limit=limit):
        frame_id = int(row["id"])
        if not store.claim_metric_frame(frame_id):
            continue
        try:
            _publish_frame(run, row, args=args, config=config)
        except Exception as exc:
            attempts = int(row.get("attempts") or 0) + 1
            if str(row.get("kind")) == "checkpoint_preview" and attempts >= 2:
                store.mark_metric_frame_terminal_failure(frame_id, repr(exc))
            else:
                store.mark_metric_frame_failed(frame_id, repr(exc))
            print(f"W&B frame publish failed id={frame_id}: {exc}", flush=True)
            if str(row.get("kind")) != "checkpoint_preview":
                break
            continue
        store.mark_metric_frame_published(
            frame_id,
            step=int(row["step"]) if row.get("step") is not None else None,
        )
        published += 1
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish one rlab telemetry stream to W&B.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli_args = build_parser().parse_args(argv)
    args = materialized_train_args(cli_args.train_config_json)
    config = resolve_env_config(env_config_from_args(args, include_states=True))
    store = MetricStore(metric_store_path(cli_args.run_dir))
    store.init()
    wandb_enabled = bool(getattr(args, "wandb", False))
    if wandb_enabled:
        store.reset_interrupted_metric_frames()
    modal_eval = str(getattr(args, "checkpoint_eval_backend", "local")) == "modal"
    if not modal_eval:
        store.reset_interrupted_artifact_uploads()
    projector = None
    try:
        retry_delay = max(cli_args.poll_seconds, 0.1)
        while True:
            store.touch_publisher()
            if wandb_enabled and projector is None:
                try:
                    projector = WandbProjector.start_live(
                        args, run_dir=str(cli_args.run_dir), config=config
                    )
                    write_wandb_url(projector.run, str(cli_args.run_dir))
                    retry_delay = max(cli_args.poll_seconds, 0.1)
                except Exception as exc:
                    store.record_publisher_error(repr(exc))
                    print(f"W&B initialization failed; retrying: {exc}", flush=True)
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2.0, 60.0)
                    continue
            activity = (
                publish_pending_frames(
                    store,
                    projector.run if projector is not None else None,
                    args=args,
                    config=config,
                    limit=max(cli_args.limit, 1),
                )
                if wandb_enabled
                else 0
            )
            if not modal_eval:
                for row in store.pending_artifact_uploads(limit=max(cli_args.limit, 1)):
                    uploaded = process_upload(
                        store=store,
                        args=args,
                        config=config,
                        row=row,
                        wandb_run=projector.run if projector is not None else None,
                    )
                    activity += int(bool(uploaded))
            if cli_args.stop_file.exists():
                if (not wandb_enabled or not store.pending_metric_frames(limit=1)) and (
                    modal_eval or not store.pending_artifact_uploads(limit=1)
                ):
                    return 0
            if not activity:
                has_backlog = bool(
                    (wandb_enabled and store.pending_metric_frames(limit=1))
                    or (not modal_eval and store.pending_artifact_uploads(limit=1))
                )
                time.sleep(retry_delay)
                retry_delay = (
                    min(retry_delay * 2.0, 60.0) if has_backlog else max(cli_args.poll_seconds, 0.1)
                )
            else:
                retry_delay = max(cli_args.poll_seconds, 0.1)
    finally:
        if projector is not None:
            projector.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
