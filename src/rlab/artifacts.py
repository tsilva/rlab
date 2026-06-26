from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rlab.env import EnvConfig, state_distribution_metadata
from rlab.env_config import parse_done_on_info
from rlab.wandb_utils import load_wandb_env


MODEL_METADATA_VERSION = 2

PLAYBACK_ENV_ARG_KEYS = {
    "game": ("game",),
    "state": ("state",),
    "states": ("states",),
    "state_probs": ("state_probs",),
    "task_conditioning": ("task_conditioning",),
    "task_conditioning_info_vars": ("task_conditioning_info_vars",),
    "task_conditioning_info_values": ("task_conditioning_info_values",),
    "frame_skip": ("frame_skip",),
    "max_pool_frames": ("max_pool_frames",),
    "sticky_action_prob": ("sticky_action_prob",),
    "max_steps": ("max_steps", "max_episode_steps"),
    "observation_size": ("observation_size",),
    "hud_crop_top": ("hud_crop_top",),
    "obs_resize_algorithm": ("obs_resize_algorithm",),
    "use_retro_reward": ("use_retro_reward",),
    "clip_rewards": ("clip_rewards",),
    "reward_mode": ("reward_mode",),
    "progress_reward_cap": ("progress_reward_cap",),
    "progress_reward_scale": ("progress_reward_scale",),
    "terminal_reward": ("terminal_reward",),
    "reward_scale": ("reward_scale",),
    "time_penalty": ("time_penalty",),
    "death_penalty": ("death_penalty",),
    "completion_reward": ("completion_reward",),
    "score_progress_clipped": ("score_progress_clipped",),
    "no_progress_timeout_steps": ("no_progress_timeout_steps",),
    "no_progress_min_delta": ("no_progress_min_delta",),
    "completion_x_threshold": ("completion_x_threshold",),
    "done_on_info_json": ("done_on_info",),
    "action_set": ("action_set",),
}


def explicit_arg_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_dests: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_dests[option] = action.dest
    return {
        option_dests[arg.split("=", 1)[0]]
        for arg in argv
        if arg.split("=", 1)[0] in option_dests
    }


def env_config_metadata(config: EnvConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["states"] = list(config.states)
    metadata["state_probs"] = list(config.state_probs)
    metadata["task_conditioning_info_vars"] = list(config.task_conditioning_info_vars)
    metadata["task_conditioning_info_values"] = [
        list(value) for value in config.task_conditioning_info_values
    ]
    if config.state_probs:
        metadata["state_sampling_mode"] = "probability"
    elif config.states:
        metadata["state_sampling_mode"] = "fixed_per_env"
    else:
        metadata["state_sampling_mode"] = "single"
    metadata["state_distribution"] = state_distribution_metadata(config)
    return metadata


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def training_preprocessing_metadata(config: EnvConfig) -> dict[str, Any]:
    return {
        "pipeline": "stable_retro_native_vec_env",
        "obs_resize": [config.observation_size, config.observation_size],
        "obs_crop": [config.hud_crop_top, 0, 0, 0] if config.hud_crop_top else None,
        "obs_grayscale": True,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "frame_skip": config.frame_skip,
        "frame_stack": 4,
        "maxpool_last_two": config.max_pool_frames,
        "copy_observations": False,
        "policy_observation_layout": "dict_image_task"
        if config.task_conditioning
        else "channel_first",
    }


def training_metadata(config: EnvConfig) -> dict[str, Any]:
    return {
        "env_config": env_config_metadata(config),
        "preprocessing": training_preprocessing_metadata(config),
        "versions": {
            "stable_retro_turbo": _package_version("stable-retro-turbo"),
            "stable_baselines3": _package_version("stable-baselines3"),
        },
    }


def stable_json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def build_model_metadata(
    args: argparse.Namespace,
    config: EnvConfig,
    model_path: Path,
    kind: str,
) -> dict[str, Any]:
    training = training_metadata(config)
    return {
        "metadata_version": MODEL_METADATA_VERSION,
        "kind": kind,
        "filename": model_path.name,
        "run_name": getattr(args, "run_name", ""),
        "run_description": getattr(args, "run_description", ""),
        "runtime_image_ref": getattr(args, "runtime_image_ref", ""),
        "run_target": getattr(args, "run_target", ""),
        "checkpoint_step": checkpoint_step(model_path),
        "env_config": training["env_config"],
        "training_metadata": training,
        "training_metadata_hash": stable_json_hash(training),
    }


def write_model_metadata(
    model_path: Path,
    args: argparse.Namespace,
    config: EnvConfig,
    kind: str,
) -> Path | None:
    if not model_path.is_file():
        return None
    path = model_metadata_path(model_path)
    path.write_text(
        json.dumps(build_model_metadata(args, config, model_path, kind), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
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


def env_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    training = metadata.get("training_metadata")
    if isinstance(training, dict):
        env_config = training.get("env_config", {})
        if isinstance(env_config, dict) and env_config:
            return env_config
    env_config = metadata.get("env_config", {})
    return env_config if isinstance(env_config, dict) else {}


def require_training_metadata(model_path: Path) -> dict[str, Any]:
    metadata = load_model_metadata(model_path)
    version = metadata.get("metadata_version")
    training = metadata.get("training_metadata")
    if version != MODEL_METADATA_VERSION or not isinstance(training, dict):
        raise ValueError(
            f"{model_path} is missing v{MODEL_METADATA_VERSION} training metadata; "
            "recreate or re-upload the checkpoint with current artifact metadata"
        )
    env_config = training.get("env_config")
    if not isinstance(env_config, dict) or not env_config:
        raise ValueError(f"{model_path} training metadata is missing env_config")
    return training


def require_env_config_from_model_metadata(model_path: Path) -> EnvConfig:
    training = require_training_metadata(model_path)
    config = env_config_from_config_dict(training["env_config"])
    if config is None:
        raise ValueError(f"{model_path} training metadata cannot be converted to EnvConfig")
    return config


def env_config_from_config_dict(
    config: dict[str, Any],
    fallback: EnvConfig | None = None,
) -> EnvConfig | None:
    field_names = set(EnvConfig.__dataclass_fields__)
    config_values = asdict(fallback) if fallback is not None else {}
    matched = False

    for field_name in field_names:
        if field_name in config and config[field_name] is not None:
            config_values[field_name] = config[field_name]
            matched = True

    if "done_on_info" in config and config.get("done_on_info") is not None:
        config_values["done_on_info"] = parse_done_on_info(config["done_on_info"])
        matched = True

    if "max_steps" in config and config.get("max_steps") is not None:
        config_values["max_episode_steps"] = config["max_steps"]
        matched = True

    if "states" in config and config.get("states") is not None:
        states = config["states"]
        config_values["states"] = tuple(states) if isinstance(states, list) else states
        matched = True
    if "state_probs" in config and config.get("state_probs") is not None:
        state_probs = config["state_probs"]
        config_values["state_probs"] = (
            tuple(state_probs) if isinstance(state_probs, list) else state_probs
        )
        matched = True
    if "task_conditioning_info_vars" in config and config.get("task_conditioning_info_vars") is not None:
        info_vars = config["task_conditioning_info_vars"]
        config_values["task_conditioning_info_vars"] = (
            tuple(info_vars) if isinstance(info_vars, list) else info_vars
        )
        matched = True
    if (
        "task_conditioning_info_values" in config
        and config.get("task_conditioning_info_values") is not None
    ):
        info_values = config["task_conditioning_info_values"]
        config_values["task_conditioning_info_values"] = (
            tuple(tuple(row) for row in info_values)
            if isinstance(info_values, list)
            else info_values
        )
        matched = True

    if not matched and fallback is None:
        return None
    return EnvConfig(**config_values)


def env_config_from_model_metadata(
    model_path: Path,
    fallback: EnvConfig | None = None,
) -> EnvConfig | None:
    saved_config = env_config_from_metadata(load_model_metadata(model_path))
    if not saved_config:
        return fallback
    return env_config_from_config_dict(saved_config, fallback=fallback)


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


def apply_model_config_defaults(
    args: argparse.Namespace,
    model_path: Path,
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> bool:
    saved_config = env_config_from_metadata(load_model_metadata(model_path))
    if not saved_config:
        return False
    apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
    return True


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

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb_config: dict[str, Any] = {
        **vars(args),
        "game": config.game,
        "state": config.state,
        "states": list(config.states),
        "state_probs": list(config.state_probs),
        "task_conditioning": config.task_conditioning,
        "task_conditioning_info_vars": list(config.task_conditioning_info_vars),
        "task_conditioning_info_values": [
            list(value) for value in config.task_conditioning_info_values
        ],
        "state_sampling_mode": (
            "probability" if config.state_probs else "fixed_per_env" if config.states else "single"
        ),
        "state_distribution": state_distribution_metadata(config),
        "frame_skip": config.frame_skip,
        "max_pool_frames": config.max_pool_frames,
        "sticky_action_prob": config.sticky_action_prob,
        "max_episode_steps": config.max_episode_steps,
        "observation_size": config.observation_size,
        "hud_crop_top": config.hud_crop_top,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "use_retro_reward": config.use_retro_reward,
        "reward_mode": config.reward_mode,
        "progress_reward_cap": config.progress_reward_cap,
        "progress_reward_scale": config.progress_reward_scale,
        "terminal_reward": config.terminal_reward,
        "reward_scale": config.reward_scale,
        "time_penalty": config.time_penalty,
        "death_penalty": config.death_penalty,
        "completion_reward": config.completion_reward,
        "score_progress_clipped": config.score_progress_clipped,
        "no_progress_timeout_steps": config.no_progress_timeout_steps,
        "no_progress_min_delta": config.no_progress_min_delta,
        "completion_x_threshold": config.completion_x_threshold,
        "done_on_info": config.done_on_info,
        "action_set": config.action_set,
    }
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        notes=args.run_description or None,
        tags=tags,
        config=wandb_config,
        dir=wandb_dir,
        sync_tensorboard=True,
        save_code=True,
        mode=args.wandb_mode,
    )
    wandb_run.define_metric("global_step")
    wandb_run.define_metric("*", step_metric="global_step")
    return wandb_run


def wandb_artifacts_enabled(wandb_run, args: argparse.Namespace) -> bool:
    return wandb_run is not None and not args.no_wandb_artifacts


def sanitize_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "rlab"


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def format_wandb_run_path(run_path) -> str:
    if isinstance(run_path, (list, tuple)):
        return "/".join(str(part) for part in run_path)
    return str(run_path)


def wandb_artifact_storage_uri(args: argparse.Namespace) -> str:
    configured_uri = args.wandb_artifact_storage_uri.strip()
    if configured_uri == "${CHECKPOINT_BUCKET_URI}":
        configured_uri = ""
    return (
        configured_uri
        or os.environ.get("WANDB_ARTIFACT_STORAGE_URI", "").strip()
        or os.environ.get("CHECKPOINT_BUCKET_URI", "").strip()
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/prefix URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def artifact_rom_prefix(game: str) -> str:
    return sanitize_artifact_name(game)


def artifact_storage_prefix(base_prefix: str, game: str) -> str:
    prefix = base_prefix.rstrip("/")
    rom_prefix = artifact_rom_prefix(game)
    if not prefix:
        return rom_prefix
    if prefix == rom_prefix or prefix.endswith(f"/{rom_prefix}"):
        return prefix
    return f"{prefix}/{rom_prefix}"


def build_s3_artifact_uri(base_uri: str, args: argparse.Namespace, model_path: Path, kind: str) -> str:
    bucket, prefix = parse_s3_uri(base_uri)
    prefix = artifact_storage_prefix(prefix, args.game)
    key_parts = [
        prefix,
        sanitize_artifact_name(args.run_name),
        kind,
        model_path.name,
    ]
    key = "/".join(part for part in key_parts if part)
    return f"s3://{bucket}/{key}"


def upload_s3_artifact(model_path: Path, destination_uri: str) -> None:
    bucket, key = parse_s3_uri(destination_uri)

    import boto3

    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    s3_client.upload_file(
        str(model_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/zip"},
    )


def log_wandb_model_artifact(
    wandb_run,
    args: argparse.Namespace,
    config: EnvConfig,
    model_path: Path,
    kind: str,
    aliases: list[str] | None = None,
) -> None:
    if not model_path.is_file():
        return
    sidecar_path = write_model_metadata(model_path, args, config, kind)
    if not wandb_artifacts_enabled(wandb_run, args):
        return

    import wandb

    artifact_name = f"{sanitize_artifact_name(args.run_name)}-{kind}"
    step = checkpoint_step(model_path)
    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "run_description": args.run_description,
        "kind": kind,
        "filename": model_path.name,
        "checkpoint_step": step,
        "metadata_version": MODEL_METADATA_VERSION,
    }
    training = training_metadata(config)
    metadata["env_config"] = training["env_config"]
    metadata["training_metadata"] = training
    metadata["training_metadata_hash"] = stable_json_hash(training)
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        metadata["wandb_run_id"] = run_id
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        metadata["wandb_run_path"] = format_wandb_run_path(run_path)

    storage_base_uri = wandb_artifact_storage_uri(args)
    reference_uri = None
    if storage_base_uri:
        reference_uri = build_s3_artifact_uri(storage_base_uri, args, model_path, kind)
        upload_s3_artifact(model_path, reference_uri)
        metadata["artifact_storage_uri"] = reference_uri

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
    wandb_run.log_artifact(artifact, aliases=aliases)
    location = reference_uri or str(model_path)
    print(f"wandb artifact logged: {artifact_name} ({location})")


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
