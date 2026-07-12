from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from rlab.artifacts import (
    build_s3_artifact_uri,
    log_wandb_model_artifact,
    sanitize_artifact_name,
    wandb_artifact_storage_uri,
)
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.metric_store import MetricStore, metric_store_path
from rlab.train_config import materialized_train_args
from rlab.wandb_utils import DEFAULT_WANDB_ENTITY, resolve_wandb_project


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


def artifact_ref(args: argparse.Namespace, kind: str, aliases: list[str]) -> str | None:
    if not getattr(args, "wandb", False) or getattr(args, "no_wandb_artifacts", False):
        return None
    entity = str(getattr(args, "wandb_entity", "") or DEFAULT_WANDB_ENTITY).strip()
    project = resolve_wandb_project(getattr(args, "wandb_project", None), str(args.game))
    if not entity or not project:
        return None
    alias = aliases[-1] if aliases else "latest"
    return f"{entity}/{project}/{sanitize_artifact_name(args.run_name)}-{kind}:{alias}"


def process_upload(
    *,
    store: MetricStore,
    args: argparse.Namespace,
    config,
    run_dir: Path,
    row: dict[str, Any],
    wandb_run=None,
):
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
            storage_uri = build_s3_artifact_uri(wandb_artifact_storage_uri(args), args, path, kind)
        store.mark_artifact_uploaded(
            checkpoint_id,
            artifact_ref=artifact_ref(args, kind, aliases),
            storage_uri=storage_uri,
        )
        return True
    except Exception as exc:
        store.mark_artifact_failed(checkpoint_id, repr(exc))
        print(f"artifact worker upload failed checkpoint_id={checkpoint_id}: {exc}", flush=True)
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flush rlab checkpoint artifacts asynchronously.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_args = materialized_train_args(args.train_config_json)
    config = resolve_env_config(env_config_from_args(train_args, include_states=True))
    store = MetricStore(metric_store_path(args.run_dir))
    store.init()
    if getattr(train_args, "wandb", False) and not getattr(train_args, "no_wandb_artifacts", False):
        raise RuntimeError(
            "artifact_worker cannot own a W&B run; start rlab.wandb_publisher instead"
        )
    while True:
        rows = store.pending_artifact_uploads(limit=max(args.limit, 1))
        if not rows and args.stop_file.exists():
            return 0
        for row in rows:
            process_upload(
                store=store,
                args=train_args,
                config=config,
                run_dir=args.run_dir,
                row=row,
            )
        if not rows:
            time.sleep(max(args.poll_seconds, 0.25))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
