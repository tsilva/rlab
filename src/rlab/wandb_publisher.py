from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rlab.artifact_worker import process_upload
from rlab.artifacts import init_wandb, write_wandb_url
from rlab.checkpoint_eval_worker import log_checkpoint_eval_metrics
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.metric_store import MetricStore, metric_store_path
from rlab.train_config import materialized_train_args


def _publish_frame(run, row: dict[str, Any], *, args, config) -> None:
    if run is None:
        return
    payload = json.loads(str(row["payload_json"]))
    kind = str(row["kind"])
    if kind == "history":
        run.log(payload)
        return
    if kind == "histogram":
        import wandb

        converted: dict[str, object] = {"global_step": payload["global_step"]}
        for name, values in payload.get("histograms", {}).items():
            converted[str(name)] = wandb.Histogram(values)
        if len(converted) > 1:
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
            store.mark_metric_frame_failed(frame_id, repr(exc))
            print(f"W&B frame publish failed id={frame_id}: {exc}", flush=True)
            break
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
    store.reset_interrupted_metric_frames()
    store.reset_interrupted_artifact_uploads()
    run = None
    try:
        retry_delay = max(cli_args.poll_seconds, 0.1)
        while True:
            store.touch_publisher()
            if run is None:
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
            activity = publish_pending_frames(
                store,
                run,
                args=args,
                config=config,
                limit=max(cli_args.limit, 1),
            )
            for row in store.pending_artifact_uploads(limit=max(cli_args.limit, 1)):
                uploaded = process_upload(
                    store=store,
                    args=args,
                    config=config,
                    run_dir=cli_args.run_dir,
                    row=row,
                    wandb_run=run,
                )
                activity += int(bool(uploaded))
            if cli_args.stop_file.exists():
                if not store.pending_metric_frames(limit=1) and not store.pending_artifact_uploads(
                    limit=1
                ):
                    return 0
            if not activity:
                has_backlog = bool(
                    store.pending_metric_frames(limit=1) or store.pending_artifact_uploads(limit=1)
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
