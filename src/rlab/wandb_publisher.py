from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rlab.artifacts import (
    build_s3_artifact_uri,
    init_wandb,
    log_wandb_model_artifact,
    wandb_artifact_collection_name,
    wandb_artifact_storage_uri,
    write_wandb_url,
)
from rlab.checkpoint_eval_worker import log_checkpoint_eval_metrics
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.metric_store import MetricStore, metric_store_path
from rlab.metric_names import validate_metric_payload
from rlab.metric_names import EVAL_SCREEN_PREVIEW
from rlab.train_config import materialized_train_args
from rlab.wandb_utils import resolve_wandb_namespace


def artifact_aliases(kind: str, step: int | None) -> list[str]:
    if kind == "checkpoint":
        aliases = ["latest"]
        if step is not None:
            aliases.append(f"step-{step}")
        return aliases
    if kind == "interrupted":
        aliases = ["interrupted", "latest"]
        if step is not None:
            aliases.append(f"step-{step}")
        return aliases
    return [kind, "latest"]


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
    name = wandb_artifact_collection_name(kind, run_id=run_id)
    return f"{entity}/{project}/{name}:{alias}"


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
    aliases = artifact_aliases(kind, step_value)
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
                run_id=getattr(wandb_run, "id", None)
                or getattr(args, "wandb_run_id", None),
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
        validate_metric_payload(payload)
        run.log(payload)
        return
    if kind == "histogram":
        import wandb

        converted: dict[str, object] = {"global_step": payload["global_step"]}
        for name, values in payload.get("histograms", {}).items():
            converted[str(name)] = wandb.Histogram(values)
        if len(converted) > 1:
            validate_metric_payload(converted)
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
    run = None
    try:
        retry_delay = max(cli_args.poll_seconds, 0.1)
        while True:
            store.touch_publisher()
            if wandb_enabled and run is None:
                try:
                    run = init_wandb(args, str(cli_args.run_dir), config)
                    write_wandb_url(run, str(cli_args.run_dir))
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
                    run,
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
                        wandb_run=run,
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
        write_wandb_url(run, str(cli_args.run_dir))
        if run is not None:
            run.finish()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
