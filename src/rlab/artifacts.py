from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rlab.env import EnvConfig, resolve_env_config
from rlab.env_metadata import (
    PLAYBACK_ENV_ARG_KEYS,
    assert_metadata_runtime_versions,
    env_config_from_config_dict,
    env_config_from_metadata,
    env_config_metadata,
    training_metadata,
)
from rlab.metric_names import (
    GLOBAL_STEP,
    METRICS_SCHEMA_VERSION,
    TRAIN_ARTIFACT_SAVE_SECONDS,
    TRAIN_ARTIFACT_UPLOAD_SECONDS,
)
from rlab.wandb_artifacts import artifact_collection_name, model_metadata_path, safe_artifact_stem
from rlab.wandb_utils import (
    configure_wandb_metrics,
    game_family_for_environment,
    load_wandb_env,
    resolve_wandb_namespace,
)


MODEL_METADATA_VERSION = 6


@dataclass(frozen=True)
class ArtifactLogTiming:
    artifact_name: str
    kind: str
    checkpoint_step: int | None
    metadata_seconds: float
    storage_upload_seconds: float
    wandb_log_seconds: float
    log_seconds: float
    stall_seconds: float
    local_save_seconds: float | None = None


def stable_json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_model_metadata(
    args: argparse.Namespace,
    config: EnvConfig,
    model_path: Path,
    kind: str,
    checkpoint_step_value: int | None = None,
) -> dict[str, Any]:
    training = training_metadata(config)
    step = checkpoint_step(model_path)
    if step is None:
        step = checkpoint_step_value
    return {
        "metadata_version": MODEL_METADATA_VERSION,
        "kind": kind,
        "filename": model_path.name,
        "run_name": getattr(args, "run_name", ""),
        "run_description": getattr(args, "run_description", ""),
        "wandb_run_id": getattr(args, "wandb_run_id", ""),
        "wandb_project": getattr(args, "wandb_project", ""),
        "batch_id": getattr(args, "batch_id", ""),
        "campaign_id": getattr(args, "campaign_id", ""),
        "game_family": getattr(args, "game_family", ""),
        "retry_of_job_id": getattr(args, "retry_of_job_id", 0),
        "goal_slug": getattr(args, "goal_slug", ""),
        "goal_path": getattr(args, "goal_path", ""),
        "goal_sha256": getattr(args, "goal_sha256", ""),
        "recipe_slug": getattr(args, "recipe_slug", ""),
        "recipe_path": getattr(args, "recipe_path", ""),
        "recipe_sha256": getattr(args, "recipe_sha256", ""),
        "queue_train_job_id": getattr(args, "queue_train_job_id", 0),
        "runtime_image_ref": getattr(args, "runtime_image_ref", ""),
        "machine": getattr(args, "machine", ""),
        "seed": getattr(args, "seed", None),
        "repo_git_commit": str(
            getattr(args, "source_sha", "") or getattr(args, "repo_git_commit", "") or ""
        ).strip(),
        "checkpoint_step": step,
        "training_backend_id": str(getattr(args, "training_backend_id", "") or "").strip(),
        "training_backend_config_hash": str(
            getattr(args, "training_backend_config_hash", "") or ""
        ).strip(),
        "algorithm_id": str(getattr(args, "algorithm_id", "") or "").strip(),
        "model_class": str(getattr(args, "model_class", "") or "").strip(),
        "training_metadata": training,
        "training_metadata_hash": stable_json_hash(training),
    }


def write_model_metadata(
    model_path: Path,
    args: argparse.Namespace,
    config: EnvConfig,
    kind: str,
    checkpoint_step_value: int | None = None,
) -> Path | None:
    if not model_path.is_file():
        return None
    return write_model_metadata_payload(
        model_path,
        build_model_metadata(
            args,
            config,
            model_path,
            kind,
            checkpoint_step_value=checkpoint_step_value,
        ),
    )


def write_model_metadata_payload(
    model_path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    path = model_metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    path = model_metadata_path(model_path)
    if not path.is_file():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"warning: could not parse model metadata {path}: {exc}", file=sys.stderr)
        return {}
    return metadata if isinstance(metadata, dict) else {}


def playback_env_config(
    config: EnvConfig,
    *,
    respect_task_termination: bool = False,
) -> EnvConfig:
    if respect_task_termination:
        return config
    task = deepcopy(config.task)
    termination = dict(task.get("termination", {}))
    termination.update(
        failure=[],
        success=[],
        timeout=[],
        max_episode_steps=0,
    )
    task["termination"] = termination

    # The Mario kernel treats a configured stalled event as a task truncation
    # even when it is not listed as a failure. Interactive playback must only
    # stop for provider-native termination, so omit that task-owned timer while
    # preserving observational events such as life loss and level change.
    events = dict(task.get("events", {}))
    events.pop("stalled", None)
    task["events"] = events
    return replace(config, task=task)


def load_playback_env_config(
    model_path: Path,
    *,
    respect_task_termination: bool = False,
) -> EnvConfig:
    metadata = load_model_metadata(model_path)
    assert_metadata_runtime_versions(metadata)
    saved_config = env_config_from_metadata(metadata)
    if not saved_config:
        raise SystemExit(
            f"{model_path} is missing playback metadata. Recreate or re-upload the "
            "checkpoint with current model metadata before using rlab play."
        )
    config = env_config_from_config_dict(saved_config)
    if config is None:
        raise SystemExit(f"{model_path} playback metadata does not contain an environment config")
    return playback_env_config(
        resolve_env_config(config),
        respect_task_termination=respect_task_termination,
    )


def apply_config_defaults(
    args: argparse.Namespace,
    config: dict[str, Any],
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> None:
    for arg_name, config_keys in PLAYBACK_ENV_ARG_KEYS.items():
        if arg_name not in parser_defaults or not hasattr(args, arg_name):
            continue
        if arg_name in explicit_dests:
            continue
        current_value = getattr(args, arg_name)
        default_value = parser_defaults[arg_name]
        if current_value != default_value and current_value not in ("", None):
            continue
        for config_key in config_keys:
            if config_key in config and config[config_key] is not None:
                setattr(args, arg_name, config[config_key])
                break


def init_wandb(args: argparse.Namespace, run_dir: str, config: EnvConfig):
    if not args.wandb:
        return None

    load_wandb_env()

    wandb_dir = os.path.abspath(run_dir)
    wandb_aux_dir = os.path.join(wandb_dir, "wandb")
    wandb_env_dirs = {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(wandb_aux_dir, "cache"),
        "WANDB_CONFIG_DIR": os.path.join(wandb_aux_dir, "config"),
        "WANDB_DATA_DIR": os.path.join(wandb_aux_dir, "data"),
        "WANDB_ARTIFACT_DIR": os.path.join(wandb_aux_dir, "artifacts"),
    }
    for env_name, path in wandb_env_dirs.items():
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
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    family_tag = f"game_family:{args.game_family}"
    if family_tag not in tags:
        tags.append(family_tag)
    args.wandb_tags = ",".join(tags)
    wandb_config: dict[str, Any] = {**vars(args), **env_config_metadata(config)}
    wandb_config["metrics_schema_version"] = METRICS_SCHEMA_VERSION
    wandb_config["algorithm_id"] = str(getattr(args, "algorithm_id", "") or "").strip()
    wandb_config["model_class"] = str(getattr(args, "model_class", "") or "").strip()
    wandb_config["training_backend_id"] = str(
        getattr(args, "training_backend_id", "") or ""
    ).strip()
    wandb_config["training_backend_config_hash"] = str(
        getattr(args, "training_backend_config_hash", "") or ""
    ).strip()
    training = training_metadata(config)
    wandb_config["environment"] = training["environment"]
    wandb_config["environment_hash"] = training["environment_hash"]
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        group=args.wandb_group,
        name=args.run_name,
        notes=args.run_description or None,
        tags=tags,
        config=wandb_config,
        dir=wandb_dir,
        sync_tensorboard=False,
        save_code=True,
        mode=args.wandb_mode,
        id=str(getattr(args, "wandb_run_id", "") or "") or None,
        resume="allow" if str(getattr(args, "wandb_run_id", "") or "") else None,
    )
    return configure_wandb_metrics(wandb_run)


def wandb_artifacts_enabled(wandb_run, args: argparse.Namespace) -> bool:
    return wandb_run is not None and not args.no_wandb_artifacts


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def format_wandb_run_path(run_path) -> str:
    if isinstance(run_path, (list, tuple)):
        return "/".join(str(part) for part in run_path)
    return str(run_path)


def strip_env_file_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


CHECKPOINT_BUCKET_URI_PLACEHOLDERS = frozenset(
    {"${CHECKPOINT_BUCKET_URI}", "$CHECKPOINT_BUCKET_URI", "CHECKPOINT_BUCKET_URI"}
)


def resolve_artifact_storage_uri(
    configured: str,
    environment: Mapping[str, str],
    *,
    allow_environment_fallback: bool = True,
) -> str:
    configured_uri = strip_env_file_quotes(configured)
    if configured_uri in CHECKPOINT_BUCKET_URI_PLACEHOLDERS:
        return strip_env_file_quotes(environment.get("CHECKPOINT_BUCKET_URI", ""))
    if configured_uri or not allow_environment_fallback:
        return configured_uri
    return strip_env_file_quotes(
        environment.get("WANDB_ARTIFACT_STORAGE_URI", "")
    ) or strip_env_file_quotes(environment.get("CHECKPOINT_BUCKET_URI", ""))


def wandb_artifact_storage_uri(args: argparse.Namespace) -> str:
    return resolve_artifact_storage_uri(
        args.wandb_artifact_storage_uri,
        os.environ,
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def artifact_rom_prefix(game: str) -> str:
    return safe_artifact_stem(game, "rlab")


def artifact_storage_prefix(base_prefix: str, game: str) -> str:
    prefix = base_prefix.rstrip("/")
    rom_prefix = artifact_rom_prefix(game)
    if not prefix:
        return rom_prefix
    if prefix == rom_prefix or prefix.endswith(f"/{rom_prefix}"):
        return prefix
    return f"{prefix}/{rom_prefix}"


def build_s3_artifact_uri(
    base_uri: str,
    args: argparse.Namespace,
    model_path: Path,
    kind: str,
    *,
    run_id: object = None,
) -> str:
    bucket, prefix = parse_s3_uri(base_uri)
    prefix = artifact_storage_prefix(prefix, args.game)
    collection_name = artifact_collection_name(
        kind,
        run_id=run_id or getattr(args, "wandb_run_id", ""),
    )
    key_parts = [
        prefix,
        collection_name,
        model_path.name,
    ]
    key = "/".join(part for part in key_parts if part)
    return f"s3://{bucket}/{key}"


def upload_s3_artifact(model_path: Path, destination_uri: str) -> None:
    bucket, key = parse_s3_uri(destination_uri)

    import boto3

    endpoint_url = strip_env_file_quotes(
        os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3", "")
    )
    client_kwargs = {"endpoint_url": endpoint_url or None}
    access_key = strip_env_file_quotes(os.environ.get("AWS_ACCESS_KEY_ID", ""))
    secret_key = strip_env_file_quotes(os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    region = strip_env_file_quotes(os.environ.get("AWS_REGION", ""))
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key:
        client_kwargs["aws_secret_access_key"] = secret_key
    if region:
        client_kwargs["region_name"] = region
    s3_client = boto3.client("s3", **client_kwargs)
    s3_client.upload_file(
        str(model_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/zip"},
    )


def artifact_timing_payload(timing: ArtifactLogTiming) -> dict[str, float]:
    payload = {
        TRAIN_ARTIFACT_UPLOAD_SECONDS: timing.storage_upload_seconds + timing.wandb_log_seconds,
    }
    if timing.local_save_seconds is not None:
        payload[TRAIN_ARTIFACT_SAVE_SECONDS] = timing.local_save_seconds
    return payload


def log_artifact_timing_metrics(
    wandb_run,
    timing: ArtifactLogTiming,
    *,
    metric_step: int | None,
) -> None:
    if wandb_run is None or metric_step is None:
        return
    wandb_run.log(
        {
            GLOBAL_STEP: metric_step,
            **artifact_timing_payload(timing),
        },
    )


def artifact_stall_seconds(
    *,
    finished_at: float,
    started_at: float,
    stall_started_at: float | None,
    local_save_seconds: float | None,
) -> float:
    if stall_started_at is not None:
        return finished_at - stall_started_at
    return finished_at - started_at + (local_save_seconds or 0.0)


def log_wandb_model_artifact(
    wandb_run,
    args: argparse.Namespace,
    config: EnvConfig,
    model_path: Path,
    kind: str,
    aliases: list[str] | None = None,
    *,
    metric_step: int | None = None,
    local_save_seconds: float | None = None,
    stall_started_at: float | None = None,
    clock: Callable[[], float] | None = None,
    purge_after_upload: bool = False,
) -> ArtifactLogTiming | None:
    if not model_path.is_file():
        return None
    timer = clock or time.perf_counter
    started_at = timer()
    run_id = getattr(wandb_run, "id", None) or getattr(args, "wandb_run_id", None)
    artifact_name = (
        artifact_collection_name(kind, run_id=run_id) if str(run_id or "").strip() else ""
    )
    step = checkpoint_step(model_path)
    artifact_step = step if step is not None else metric_step

    metadata_started_at = timer()
    metadata = build_model_metadata(
        args,
        config,
        model_path,
        kind,
        checkpoint_step_value=artifact_step,
    )
    sidecar_path = write_model_metadata_payload(model_path, metadata)
    metadata_seconds = timer() - metadata_started_at

    wandb_output_enabled = wandb_artifacts_enabled(wandb_run, args)
    storage_base_uri = (
        wandb_artifact_storage_uri(args) if not getattr(args, "no_wandb_artifacts", False) else ""
    )
    if not wandb_output_enabled and not storage_base_uri:
        finished_at = timer()
        return ArtifactLogTiming(
            artifact_name=artifact_name,
            kind=kind,
            checkpoint_step=artifact_step,
            metadata_seconds=metadata_seconds,
            storage_upload_seconds=0.0,
            wandb_log_seconds=0.0,
            log_seconds=finished_at - started_at,
            stall_seconds=artifact_stall_seconds(
                finished_at=finished_at,
                started_at=started_at,
                stall_started_at=stall_started_at,
                local_save_seconds=local_save_seconds,
            ),
            local_save_seconds=local_save_seconds,
        )

    if not artifact_name:
        raise ValueError("new artifact writes require an immutable W&B run id")

    if run_id:
        metadata["wandb_run_id"] = run_id
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        metadata["wandb_run_path"] = format_wandb_run_path(run_path)

    reference_uri = None
    storage_upload_seconds = 0.0
    if storage_base_uri:
        reference_uri = build_s3_artifact_uri(
            storage_base_uri,
            args,
            model_path,
            kind,
            run_id=run_id,
        )
        upload_started_at = timer()
        upload_s3_artifact(model_path, reference_uri)
        storage_upload_seconds = timer() - upload_started_at
        metadata["artifact_storage_uri"] = reference_uri

    if not wandb_output_enabled:
        finished_at = timer()
        timing = ArtifactLogTiming(
            artifact_name=artifact_name,
            kind=kind,
            checkpoint_step=artifact_step,
            metadata_seconds=metadata_seconds,
            storage_upload_seconds=storage_upload_seconds,
            wandb_log_seconds=0.0,
            log_seconds=finished_at - started_at,
            stall_seconds=artifact_stall_seconds(
                finished_at=finished_at,
                started_at=started_at,
                stall_started_at=stall_started_at,
                local_save_seconds=local_save_seconds,
            ),
            local_save_seconds=local_save_seconds,
        )
        print(
            f"artifact stored: {artifact_name} ({reference_uri}); "
            f"artifact_stall_seconds={timing.stall_seconds:.3f}"
        )
        if purge_after_upload and getattr(args, "wandb_mode", "online") == "online":
            purge_model_artifact_files(model_path)
        return timing

    import wandb

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata=metadata,
    )
    if reference_uri:
        artifact.add_reference(reference_uri, name=model_path.name)
    else:
        artifact.add_file(str(model_path), name=model_path.name)
    if sidecar_path is not None:
        artifact.add_file(str(sidecar_path), name=sidecar_path.name)
    wandb_log_started_at = timer()
    logged_artifact = wandb_run.log_artifact(artifact, aliases=aliases)
    if logged_artifact is not None and hasattr(logged_artifact, "wait"):
        logged_artifact.wait()
    wandb_log_seconds = timer() - wandb_log_started_at
    finished_at = timer()
    timing = ArtifactLogTiming(
        artifact_name=artifact_name,
        kind=kind,
        checkpoint_step=artifact_step,
        metadata_seconds=metadata_seconds,
        storage_upload_seconds=storage_upload_seconds,
        wandb_log_seconds=wandb_log_seconds,
        log_seconds=finished_at - started_at,
        stall_seconds=artifact_stall_seconds(
            finished_at=finished_at,
            started_at=started_at,
            stall_started_at=stall_started_at,
            local_save_seconds=local_save_seconds,
        ),
        local_save_seconds=local_save_seconds,
    )
    log_artifact_timing_metrics(
        wandb_run,
        timing,
        metric_step=metric_step if metric_step is not None else artifact_step,
    )
    location = reference_uri or str(model_path)
    print(
        f"wandb artifact logged: {artifact_name} ({location}); "
        f"artifact_stall_seconds={timing.stall_seconds:.3f}"
    )
    if purge_after_upload and getattr(args, "wandb_mode", "online") == "online":
        purge_model_artifact_files(model_path)
    return timing


def purge_model_artifact_files(model_path: Path) -> tuple[Path, ...]:
    purged: list[Path] = []
    for path in (model_path, model_metadata_path(model_path)):
        try:
            if path.is_file():
                path.unlink()
                purged.append(path)
        except OSError as exc:
            print(f"warning: could not purge uploaded artifact file {path}: {exc}", file=sys.stderr)
    if purged:
        print(
            "purged uploaded artifact files: " + ", ".join(str(path) for path in purged),
            flush=True,
        )
    return tuple(purged)


def write_wandb_url(wandb_run, run_dir: str) -> None:
    if wandb_run is None:
        return

    run_url = getattr(wandb_run, "url", None)
    if run_url:
        Path(run_dir, "wandb_url.txt").write_text(f"{run_url}\n", encoding="utf-8")
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        Path(run_dir, "wandb_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        Path(run_dir, "wandb_run_path.txt").write_text(
            f"{format_wandb_run_path(run_path)}\n",
            encoding="utf-8",
        )


def write_run_description(args: argparse.Namespace, run_dir: str) -> None:
    description = args.run_description.strip()
    Path(run_dir, "run_description.txt").write_text(
        f"{description}\n" if description else "",
        encoding="utf-8",
    )
