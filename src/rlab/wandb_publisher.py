from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.env_metadata import env_config_metadata, training_metadata
from rlab.metric_names import (
    EVAL_FULL_BY_START,
    LEADER_CHECKPOINT_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_BEST_RETURN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_PROGRESS_MAX,
    LEADER_CHECKPOINT_RANK_VALUES,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_UPDATED_AT,
    METRICS_SCHEMA_VERSION,
    validate_metric_payload,
)
from rlab.metric_store import MetricStore
from rlab.wandb_utils import (
    configure_wandb_metrics,
    game_family_for_environment,
    load_wandb_env,
    resolve_wandb_namespace,
)

WANDB_FINISH_TIMEOUT_SECONDS = 300.0


def _write_wandb_identity(run, run_dir: str) -> None:
    if run is None:
        return
    for attribute, filename in (
        ("url", "wandb_url.txt"),
        ("id", "wandb_run_id.txt"),
    ):
        value = getattr(run, attribute, None)
        if value:
            Path(run_dir, filename).write_text(f"{value}\n", encoding="utf-8")


def _start_wandb(args, *, run_dir: str, config):
    if not args.wandb:
        raise ValueError("supervised training requires W&B metric publication")
    load_wandb_env()
    wandb_dir = os.path.abspath(run_dir)
    for env_name, path in {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(wandb_dir, "wandb", "cache"),
        "WANDB_CONFIG_DIR": os.path.join(wandb_dir, "wandb", "config"),
        "WANDB_DATA_DIR": os.path.join(wandb_dir, "wandb", "data"),
    }.items():
        os.environ.setdefault(env_name, path)
        os.makedirs(os.environ[env_name], exist_ok=True)

    import wandb

    entity, project = resolve_wandb_namespace(
        getattr(args, "wandb_entity", None),
        getattr(args, "wandb_project", None),
        config.game,
        env_provider=config.env_provider,
    )
    args.wandb_entity = entity
    args.wandb_project = project
    args.game_family = game_family_for_environment(config.env_provider, config.game)
    tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
    family_tag = f"game_family:{args.game_family}"
    if family_tag not in tags:
        tags.append(family_tag)
    args.wandb_tags = ",".join(tags)
    wandb_config: dict[str, Any] = {**vars(args), **env_config_metadata(config)}
    wandb_config["metrics_schema_version"] = int(
        getattr(args, "metrics_schema_version", METRICS_SCHEMA_VERSION)
        or METRICS_SCHEMA_VERSION
    )
    training = training_metadata(
        config,
        rom_asset_manifest=getattr(args, "rom_asset_manifest", None),
    )
    wandb_config["environment"] = training["environment"]
    wandb_config["environment_hash"] = training["environment_hash"]
    return configure_wandb_metrics(
        wandb.init(
            project=project,
            entity=entity,
            group=args.wandb_group,
            name=args.run_name,
            notes=args.run_description or None,
            tags=tags,
            config=wandb_config,
            dir=wandb_dir,
            sync_tensorboard=False,
            save_code=False,
            mode=args.wandb_mode,
            id=str(args.wandb_run_id),
            resume="allow",
            settings=wandb.Settings(
                finish_timeout=WANDB_FINISH_TIMEOUT_SECONDS,
                finish_timeout_raises=True,
            ),
        )
    )


class WandbProjector:
    """The sole W&B SDK owner for one logical dstack run."""

    def __init__(self, run, *, run_dir: str | None = None) -> None:
        self.run = run
        self.run_dir = run_dir

    @classmethod
    def start_live(cls, args, *, run_dir: str, config) -> WandbProjector:
        return cls(_start_wandb(args, run_dir=run_dir, config=config), run_dir=run_dir)

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
                settings=wandb.Settings(
                    finish_timeout=WANDB_FINISH_TIMEOUT_SECONDS,
                    finish_timeout_raises=True,
                    x_update_finish_state=update_finish_state,
                ),
            )
        )
        return cls(run)

    def close(self) -> None:
        if self.run_dir is not None:
            _write_wandb_identity(self.run, self.run_dir)
        if self.run is not None:
            self.run.finish()


def _publish_frame(run, row: Mapping[str, Any], *, args) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")
    payload = json.loads(str(row["payload_json"]))
    kind = str(row["kind"])
    event_seq = int(row["id"])
    event_id = str(row["event_id"])
    step = int(row["step"] or payload.get("global_step") or 0)
    source = str(row.get("source") or "")

    if kind == "history":
        validate_metric_payload(
            payload,
            schema_version=int(getattr(args, "metrics_schema_version", 6) or 6),
        )
        payload["orchestration/event_seq"] = event_seq
        payload["orchestration/event_id"] = event_id
        if source.startswith("eval"):
            payload["eval/checkpoint_step"] = step
        elif not source.startswith("orchestration"):
            payload["train/global_step"] = step
        # Use the durable outbox sequence as W&B's internal step. If the SDK call
        # succeeded but the local acknowledgement was interrupted, replaying the
        # same sequence is rejected by W&B as an already-committed step instead
        # of appending a second scientific point.
        run.log(payload, step=event_seq)
        return

    if kind == "histogram":
        import wandb

        converted: dict[str, object] = {
            "global_step": payload["global_step"],
            "train/global_step": step,
            "orchestration/event_seq": event_seq,
            "orchestration/event_id": event_id,
        }
        for name, values in payload.get("histograms", {}).items():
            converted[str(name)] = wandb.Histogram(values)
        validate_metric_payload(
            converted,
            schema_version=int(getattr(args, "metrics_schema_version", 6) or 6),
        )
        run.log(converted, step=event_seq)
        return

    if kind == "eval_by_start":
        import wandb

        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("eval_by_start frame must contain rows")
        run.log(
            {
                "global_step": step,
                "eval/checkpoint_step": step,
                "orchestration/event_seq": event_seq,
                "orchestration/event_id": event_id,
                EVAL_FULL_BY_START: wandb.Table(
                    columns=[
                        "checkpoint_step",
                        "start_id",
                        "episodes",
                        "success_count",
                        "success_rate",
                        "return_mean",
                        "return_std",
                        "return_median",
                        "reason",
                        "reason_count",
                        "reason_rate",
                    ],
                    data=[[step, *list(result)] for result in rows],
                ),
            },
            step=event_seq,
        )
        return

    raise ValueError(f"unsupported supervisor telemetry frame kind: {kind}")


def publish_pending_frames(
    store: MetricStore,
    run,
    *,
    args,
    config,
    limit: int,
) -> int:
    del config
    published = 0
    for row in store.pending_metric_frames(limit=limit):
        frame_id = int(row["id"])
        if not store.claim_metric_frame(frame_id):
            continue
        try:
            _publish_frame(run, row, args=args)
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


def publish_promotion_summary(
    run,
    *,
    checkpoint_step: int,
    checkpoint_url: str,
    metrics: Mapping[str, Any],
    updated_at: str,
) -> None:
    if run is None:
        raise RuntimeError("W&B run is unavailable")

    def numeric(name: str, default: float = 0.0) -> float:
        value = metrics.get(name)
        return default if not isinstance(value, int | float) else float(value)

    progress = max(
        (
            float(value)
            for name, value in metrics.items()
            if str(name).startswith("eval/full/progress/")
            and str(name).endswith("/max")
            and isinstance(value, int | float)
        ),
        default=0.0,
    )
    success_min = numeric("eval/full/outcome/success/rate/min", 1.0)
    success_mean = numeric("eval/full/outcome/success/rate/mean", success_min)
    return_mean = numeric("eval/full/episode/return/mean")
    best_return = numeric("eval/full/episode/return/best", return_mean)
    run.summary.update(
        {
            "rlab/goal/outcome": "accepted",
            LEADER_CHECKPOINT_ACCEPTANCE_PASS: 1.0,
            LEADER_CHECKPOINT_SUCCESS_RATE_MIN: success_min,
            LEADER_CHECKPOINT_SUCCESS_RATE_MEAN: success_mean,
            LEADER_CHECKPOINT_OBJECTIVE: success_min,
            LEADER_CHECKPOINT_RETURN_MEAN: return_mean,
            LEADER_CHECKPOINT_BEST_RETURN: best_return,
            LEADER_CHECKPOINT_RANK_VALUES: json.dumps(
                [int(checkpoint_step), return_mean],
                separators=(",", ":"),
            ),
            LEADER_CHECKPOINT_PROGRESS_MAX: progress,
            LEADER_CHECKPOINT_STEP: int(checkpoint_step),
            LEADER_CHECKPOINT_ARTIFACT_REF: checkpoint_url,
            LEADER_CHECKPOINT_EVAL_SOURCE: "modal:acceptance",
            LEADER_CHECKPOINT_UPDATED_AT: updated_at,
        }
    )
