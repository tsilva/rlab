from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from rlab.artifacts import artifact_storage_prefix, checkpoint_step, sanitize_artifact_name
from rlab.cli import TRAIN_COMMAND_FIELDS, build_train_command
from rlab.compute_targets import (
    canonical_target_name,
    ensure_skypilot_target,
    instance_defaults,
    instance_label,
    launch_infra,
    load_instance_config,
    load_json_file,
    target_kind,
    target_name,
)
from rlab.metric_names import (
    TRAIN_DONE_ALL,
)
from rlab.runtime_refs import normalize_runtime_image_ref, runtime_image_digest_slug
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT


REQUIRED_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT_URL",
    "AWS_REGION",
    "CHECKPOINT_BUCKET_URI",
    "WANDB_API_KEY",
)
DATABASE_ENV_KEYS = (
    "TRAIN_QUEUE_DATABASE_URL",
    "DATABASE_URL",
    "DIRECT_DATABASE_URL",
)
REQUIRED_STABLE_RETRO_TURBO_VERSION = "1.0.0.post21"


@dataclass(frozen=True)
class Check:
    level: str
    message: str


@dataclass(frozen=True)
class LaunchSummary:
    command: list[str]
    task_path: Path
    cluster: str
    wandb_group_prefix: str


@dataclass(frozen=True)
class SparseEvent:
    kind: str
    message: str


@dataclass(frozen=True)
class EndpointCheck:
    endpoint: str
    ok: bool
    message: str


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def merged_env(dotenv_path: Path) -> dict[str, str]:
    values = load_dotenv(dotenv_path)
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_file(path)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must define at least one run in runs[]")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be an object")
        if not run.get("suffix"):
            raise ValueError(f"runs[{index}] must define suffix")
    return manifest


def load_runner_profile(path: Path) -> dict[str, Any]:
    profile = load_json_file(path)
    if not str(profile.get("profile_id", "")).strip():
        raise ValueError("runner profile must define profile_id")
    if not str(profile.get("game", "")).strip():
        raise ValueError("runner profile must define game")
    if not str(profile.get("rom_source", "")).strip():
        raise ValueError("runner profile must define rom_source")
    return profile


def manifest_game(manifest: dict[str, Any]) -> str:
    game = str(manifest.get("game", "")).strip()
    base = manifest.get("base_train", {})
    if not game and isinstance(base, dict):
        game = str(base.get("game", "")).strip()
    if not game:
        raise ValueError("manifest must define game or base_train.game")
    return game


def runner_profile_game(profile: dict[str, Any]) -> str:
    game = str(profile.get("game", "")).strip()
    if not game:
        raise ValueError("runner profile must define game")
    return game


def shell_quote(part: str) -> str:
    if part.startswith('"') and part.endswith('"'):
        return part
    if "$" in part and re.fullmatch(r"[A-Za-z0-9_./:${}-]+", part):
        return f'"{part}"'
    return shlex.quote(part)


def shell_join(parts: list[str]) -> str:
    return " ".join(shell_quote(part) for part in parts)


def training_options(manifest: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if manifest.get("game"):
        options["game"] = manifest["game"]
    if manifest.get("state"):
        options["state"] = manifest["state"]
    base = manifest.get("base_train", {})
    if isinstance(base, dict):
        options.update(base)
    options.update({key: value for key, value in run.items() if key not in {"suffix"}})
    return options


def sanitize_slug(value: str) -> str:
    return sanitize_artifact_name(value).replace(".", "-").replace("_", "-")


def runner_cluster_name(
    *,
    profile_id: str,
    target: str,
    runtime_image_ref: str,
) -> str:
    digest_slug = runtime_image_digest_slug(runtime_image_ref)
    target_slug = sanitize_slug(target)[:24].strip("-") or "target"
    profile_slug = sanitize_slug(profile_id)[:48].strip("-") or "profile"
    return f"rlab-runner-{target_slug}-{profile_slug}-{digest_slug}"


def parse_wandb_run_ref(run_ref: str) -> tuple[str, str, str]:
    parts = [part for part in run_ref.strip("/").split("/") if part]
    if len(parts) != 3:
        raise ValueError("W&B run ref must be entity/project/run_id")
    return parts[0], parts[1], parts[2]


def fetch_wandb_run_config(run_ref: str) -> dict[str, Any]:
    import wandb

    run = wandb.Api().run(run_ref)
    return dict(run.config)


def _manifest_train_value(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return value


def manifest_from_wandb_config(
    run_ref: str,
    config: dict[str, Any],
    rom_source: str,
    *,
    name: str | None = None,
    cluster: str | None = None,
    artifact_storage_uri: str = "${CHECKPOINT_BUCKET_URI}",
) -> dict[str, Any]:
    entity, project, run_id = parse_wandb_run_ref(run_ref)
    game = str(config.get("game", "")).strip()
    if not game:
        raise ValueError("W&B config does not contain a game id; pass a manifest instead")
    rom_source_path = Path(rom_source)
    experiment_slug = sanitize_slug(name or f"repro-{run_id}-4090")
    base_train: dict[str, Any] = {}
    skipped_fields = {"run_name", "run_description", "wandb_group", "wandb_tags", "resume"}
    for key in TRAIN_COMMAND_FIELDS:
        if key in skipped_fields or key not in config:
            continue
        value = _manifest_train_value(config[key])
        if value is not None and value != "":
            base_train[key] = value

    base_train.setdefault("wandb", True)
    base_train.setdefault("wandb_mode", "online")
    base_train.setdefault("wandb_project", project)
    base_train.setdefault("wandb_entity", entity)
    base_train.setdefault("eval_freq", 0)
    base_train.setdefault("eval_episodes", 0)
    base_train.setdefault("device", "cuda")
    base_train.setdefault("wandb_artifact_storage_uri", artifact_storage_uri)

    run_description = (
        f"Reproduction of W&B run {run_ref} using its logged training config. "
        f"ROM path is supplied by this launch manifest so the workflow remains ROM-agnostic."
    )
    seed = config.get("seed", 0)
    tags = ["repro", f"source-run-{sanitize_slug(run_id)}", "rtx4090"]
    existing_tags = str(config.get("wandb_tags", "")).split(",")
    for tag in existing_tags:
        tag = tag.strip()
        if tag and tag not in tags:
            tags.append(tag)

    manifest = {
        "name": experiment_slug,
        "cluster": cluster or f"rlab-{experiment_slug}",
        "game": game,
        "state": str(config.get("state", "") or ""),
        "rom_source": str(rom_source_path),
        "rom_mount_path": f"~/roms/{game}/{rom_source_path.name}",
        "run_name_prefix": f"{experiment_slug}_{sanitize_slug(run_id)}",
        "wandb_group_prefix": f"{experiment_slug}-{sanitize_slug(run_id)}",
        "log_dir": f"logs/{experiment_slug}",
        "venv_name": f"{experiment_slug}-venv",
        "wandb_project": project,
        "wandb_tags": tags,
        "base_train": base_train,
        "runs": [
            {
                "suffix": f"seed{seed}",
                "seed": seed,
                "run_description": run_description,
            }
        ],
    }
    return manifest


def run_name_expr(manifest: dict[str, Any], run: dict[str, Any]) -> str:
    prefix = str(manifest.get("run_name_prefix", manifest["name"]))
    suffix = str(run["suffix"])
    return f"{prefix}_{suffix}_${{TS}}"


def render_train_command(py_expr: str, options: dict[str, Any]) -> str:
    cmd = build_train_command(options)
    if cmd[:3] != ["python", "-m", "rlab.train"]:
        raise ValueError("unexpected train command prefix")
    rendered = [py_expr, "-m", "rlab.train", *cmd[3:]]
    return shell_join(rendered)


def yaml_string(value: str) -> str:
    return json.dumps(value)


def indent_block(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def local_mount_path(repo_root: Path, path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_absolute():
        return source
    return (repo_root.resolve() / source).resolve()


def render_task_yaml(
    manifest: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    *,
    target_override: str | None = None,
) -> str:
    target = target_name(manifest, target_override)
    instance = instance_defaults(instance_config, target)
    ensure_skypilot_target(instance)
    game = manifest_game(manifest)
    sky_name = str(manifest.get("name", "rlab-rtx4090"))
    stable_retro_turbo_version = str(
        manifest.get("stable_retro_turbo_version", REQUIRED_STABLE_RETRO_TURBO_VERSION)
    ).strip()
    venv_name = str(manifest.get("venv_name", f"{sky_name}-venv"))
    log_dir = str(manifest.get("log_dir", f"logs/{sky_name}"))
    wandb_group_prefix = str(manifest.get("wandb_group_prefix", sky_name))
    image_id = str(instance.get("image_id", "")).strip()
    cpus = str(manifest.get("cpus", instance.get("cpus", "12+")))
    memory = str(manifest.get("memory", instance.get("memory", "48+")))
    accelerator = str(instance.get("accelerator", "RTX4090"))
    target_description = str(instance.get("infra") or target)
    if not manifest.get("rom_source"):
        raise ValueError("manifest must define rom_source for SkyPilot file_mounts")
    rom_source = local_mount_path(repo_root, str(manifest["rom_source"]))
    rom_mount_path = str(manifest.get("rom_mount_path", f"~/roms/{game}/{rom_source.name}"))
    extra_file_mounts = manifest.get("extra_file_mounts", {})
    if extra_file_mounts and not isinstance(extra_file_mounts, dict):
        raise ValueError("extra_file_mounts must be a mapping of remote path to local path")
    file_mount_lines = [
        "file_mounts:",
        f"  {rom_mount_path}: {rom_source}",
    ]
    for remote_path, local_path in extra_file_mounts.items():
        file_mount_lines.append(f"  {remote_path}: {local_mount_path(repo_root, str(local_path))}")
    smoke_options = training_options(manifest, manifest["runs"][0])
    smoke_state = str(smoke_options.get("state", ""))
    smoke_states = _manifest_train_value(smoke_options.get("states", ""))
    smoke_state_probs = _manifest_train_value(smoke_options.get("state_probs", ""))
    smoke_states_tuple = (
        tuple(part.strip() for part in str(smoke_states).split(",") if part.strip())
        if smoke_states
        else ()
    )
    smoke_state_probs_tuple = (
        tuple(float(part.strip()) for part in str(smoke_state_probs).split(",") if part.strip())
        if smoke_state_probs
        else ()
    )
    smoke_n_envs = (
        len(smoke_states_tuple)
        if smoke_states_tuple and not smoke_state_probs_tuple
        else int(smoke_options.get("n_envs", 2))
    )
    smoke_hud_crop_top = int(smoke_options.get("hud_crop_top", -1))
    smoke_reward_mode = str(smoke_options.get("reward_mode", "auto"))
    smoke_action_set = str(smoke_options.get("action_set", "auto"))
    smoke_done_on_info_json = str(smoke_options.get("done_on_info_json", ""))
    smoke_done_on_info = json.loads(smoke_done_on_info_json) if smoke_done_on_info_json else {}
    smoke_completion_x_threshold = int(smoke_options.get("completion_x_threshold", -1))

    setup_block = f"""set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y libxcb1 libgl1 libglib2.0-0
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n apt-get update
    sudo -n apt-get install -y libxcb1 libgl1 libglib2.0-0
  else
    apt-get update
    apt-get install -y libxcb1 libgl1 libglib2.0-0
  fi
fi

BOOTSTRAP_PYTHON="${{SKY_RUNTIME_DIR:-$HOME}}/skypilot-runtime/bin/python"
if [ ! -x "$BOOTSTRAP_PYTHON" ]; then
  BOOTSTRAP_PYTHON=/home/sky/skypilot-runtime/bin/python
fi
JOB_VENV="$HOME/{venv_name}"
UV_VENV="$HOME/{venv_name}-uv"

"$BOOTSTRAP_PYTHON" -m venv --clear "$JOB_VENV"
"$BOOTSTRAP_PYTHON" -m venv "$UV_VENV"
"$UV_VENV/bin/python" -m pip install --upgrade pip
"$UV_VENV/bin/python" -m pip install --upgrade "uv>=0.9,<1"

export UV_PROJECT_ENVIRONMENT="$JOB_VENV"
"$UV_VENV/bin/uv" sync --locked --no-dev
"$JOB_VENV/bin/python" -m ensurepip --upgrade
"$JOB_VENV/bin/python" -m pip install --no-deps --force-reinstall "stable-retro-turbo=={stable_retro_turbo_version}"
"$JOB_VENV/bin/python" -m stable_retro.import "$HOME/roms"

"$JOB_VENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
import stable_retro as retro

game = {game!r}
mounted_dir = Path.home() / "roms" / game
rom_path = retro.data.get_romfile_path(game)
if rom_path is not None and mounted_dir.exists():
    package_dir = Path(rom_path).parent
    for source in mounted_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, package_dir / source.name)
    print("synced_mounted_game_data", mounted_dir, "->", package_dir)
PY

"$JOB_VENV/bin/python" - <<'PY'
import importlib.metadata
import cv2
import stable_retro as retro
import torch
from rlab.env import EnvConfig, make_training_vec_env, resolve_env_config

print("stable-retro-turbo", importlib.metadata.version("stable-retro-turbo"))
print("cv2", cv2.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("rom", retro.data.get_romfile_path({game!r}))
config = resolve_env_config(EnvConfig(
    game={game!r},
    state={smoke_state!r},
    states={smoke_states_tuple!r},
    state_probs={smoke_state_probs_tuple!r},
    hud_crop_top={smoke_hud_crop_top!r},
    reward_mode={smoke_reward_mode!r},
    action_set={smoke_action_set!r},
    done_on_info={smoke_done_on_info!r},
    completion_x_threshold={smoke_completion_x_threshold!r},
))
env = make_training_vec_env(config, n_envs={smoke_n_envs!r}, seed=0)
try:
    obs = env.reset()
    print("training_observation_space", env.observation_space)
    print("training_reset_shape", getattr(obs, "shape", None), getattr(obs, "dtype", None))
finally:
    env.close()
PY"""

    run_lines = [
        "set -euo pipefail",
        "",
        f'PY="$HOME/{venv_name}/bin/python"',
        'TS="$(date -u +%Y%m%d_%H%M%S)"',
        f'GROUP="{wandb_group_prefix}-${{TS}}"',
        f'LOG_DIR="{log_dir}"',
        'mkdir -p "$LOG_DIR"',
        "",
        'echo "wandb_group=${GROUP}"',
        f'echo "runs={len(manifest["runs"])} target={target_description}"',
        "",
        "status=0",
    ]

    for index, run in enumerate(manifest["runs"], start=1):
        options = training_options(manifest, run)
        run_name = run_name_expr(manifest, run)
        options["run_name"] = run_name
        options.setdefault("wandb", True)
        options.setdefault("wandb_project", manifest.get("wandb_project", DEFAULT_WANDB_PROJECT))
        options.setdefault("wandb_group", "${GROUP}")
        tags = manifest.get("wandb_tags")
        if isinstance(tags, list):
            options.setdefault("wandb_tags", ",".join(str(tag) for tag in tags))
        options.setdefault("wandb_mode", "online")
        cmd = render_train_command('"$PY"', options)
        log_path = f"${{LOG_DIR}}/{run_name}.log"
        pid_var = f"pid_{index}"
        run_lines.extend(
            [
                "",
                f'echo "starting {run_name}"',
                "(",
                "  set -euo pipefail",
                f"  {cmd} 2>&1 | tee {shell_quote(log_path)}",
                "  status=${PIPESTATUS[0]}",
                f'  echo "{run_name} exit status: ${{status}}"',
                '  exit "$status"',
                f") & {pid_var}=$!",
            ]
        )

    run_lines.append("")
    for index in range(1, len(manifest["runs"]) + 1):
        run_lines.append(f'wait "$pid_{index}" || status=$?')
    run_lines.extend(
        [
            "",
            'echo "wandb_group=${GROUP}"',
            'for path in runs/*_${TS}/early_stop.txt; do',
            '  if [ -f "$path" ]; then',
            '    echo "--- ${path} ---"',
            '    cat "$path"',
            "  fi",
            "done",
            f'echo "logs_dir={log_dir}"',
            'exit "$status"',
        ]
    )

    resource_lines = [
        "resources:",
        f"  accelerators: {{{accelerator}: 1}}",
        f"  cpus: {cpus}",
        f"  memory: {memory}",
    ]
    if image_id:
        resource_lines.append(f"  image_id: {yaml_string(image_id)}")

    yaml = [
        f"name: {sky_name}",
        "",
        "workdir: .",
        "",
        *file_mount_lines,
        "",
        *resource_lines,
        "",
        "setup: |",
        indent_block(setup_block, 2),
        "",
        "run: |",
        indent_block("\n".join(run_lines), 2),
    ]
    return "\n".join(yaml) + "\n"


def build_launch_command(
    cluster: str,
    task_path: Path,
    *,
    infra: str | None = None,
    detach_run: bool = False,
    extra_env: tuple[str, ...] = (),
    extra_secrets: tuple[str, ...] = (),
) -> list[str]:
    cmd = ["sky", "launch", "-c", cluster, "-y", str(task_path)]
    if infra:
        cmd.extend(["--infra", infra])
    if detach_run:
        cmd.append("--detach-run")
    for key in ("AWS_REGION", "AWS_S3_ENDPOINT_URL", "CHECKPOINT_BUCKET_URI", *extra_env):
        cmd.extend(["--env", key])
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY", *extra_secrets):
        cmd.extend(["--secret", key])
    return cmd


def available_database_env_keys(env: dict[str, str]) -> tuple[str, ...]:
    return tuple(key for key in DATABASE_ENV_KEYS if env.get(key))


def build_runner_launch_command(
    cluster: str,
    task_path: Path,
    *,
    env: dict[str, str],
    infra: str | None = None,
    detach_run: bool = False,
) -> list[str]:
    database_secrets = available_database_env_keys(env)
    if not database_secrets:
        raise ValueError(
            "runner launch requires a database secret from one of "
            f"{', '.join(DATABASE_ENV_KEYS)} in the local environment or .env",
        )
    return build_launch_command(
        cluster,
        task_path,
        infra=infra,
        detach_run=detach_run,
        extra_secrets=database_secrets,
    )


def write_rendered_task(
    manifest: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    output_path: Path,
    *,
    target_override: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_task_yaml(manifest, instance_config, repo_root, target_override=target_override),
        encoding="utf-8",
    )
    return output_path


def _file_mount_lines(
    *,
    game: str,
    rom_source: Path,
    rom_mount_path: str,
    repo_root: Path,
    extra_file_mounts: Any,
) -> list[str]:
    if extra_file_mounts and not isinstance(extra_file_mounts, dict):
        raise ValueError("extra_file_mounts must be a mapping of remote path to local path")
    file_mount_lines = [
        "file_mounts:",
        f"  {rom_mount_path}: {rom_source}",
    ]
    if isinstance(extra_file_mounts, dict):
        for remote_path, local_path in extra_file_mounts.items():
            file_mount_lines.append(f"  {remote_path}: {local_mount_path(repo_root, str(local_path))}")
    return file_mount_lines


def render_runner_task_yaml(
    profile: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    *,
    target_override: str | None = None,
) -> str:
    target = target_name(profile, target_override)
    instance = instance_defaults(instance_config, target)
    ensure_skypilot_target(instance)
    canonical_target = canonical_target_name(instance_config, target)
    game = runner_profile_game(profile)
    profile_id = str(profile["profile_id"]).strip()
    sky_name = str(profile.get("name", sanitize_slug(profile_id)))
    stable_retro_turbo_version = str(
        profile.get("stable_retro_turbo_version", REQUIRED_STABLE_RETRO_TURBO_VERSION)
    ).strip()
    venv_name = str(profile.get("venv_name", f"{sky_name}-venv"))
    log_dir = str(profile.get("log_dir", "logs/train_runner"))
    workers = int(profile.get("workers", instance.get("children", 1)))
    max_jobs = int(profile.get("max_jobs", 0))
    poll_seconds = float(profile.get("poll_seconds", 30))
    lease_seconds = int(profile.get("lease_seconds", 1800))
    heartbeat_seconds = float(profile.get("heartbeat_seconds", 30.0))
    once = bool(profile.get("once", False))
    direct = bool(profile.get("direct", False))
    status_goal = str(profile.get("status_goal", "")).strip()
    image_id = str(profile.get("image_id", instance.get("image_id", ""))).strip()
    prebuilt_image = bool(profile.get("prebuilt_image", instance.get("prebuilt_image", False)))
    runtime_image_ref = str(profile.get("runtime_image_ref", image_id if prebuilt_image else "")).strip()
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    if prebuilt_image:
        if image_id and image_id != runtime_image_ref:
            raise ValueError("prebuilt runner image_id must match runtime_image_ref when both are set")
        image_id = runtime_image_ref
    cpus = str(profile.get("cpus", instance.get("cpus", "12+")))
    memory = str(profile.get("memory", instance.get("memory", "48+")))
    accelerator = str(instance.get("accelerator", "RTX4090"))
    target_description = str(instance.get("infra") or target)
    rom_source = local_mount_path(repo_root, str(profile["rom_source"]))
    rom_mount_path = str(profile.get("rom_mount_path", f"~/roms/{game}/{rom_source.name}"))
    file_mount_lines = _file_mount_lines(
        game=game,
        rom_source=rom_source,
        rom_mount_path=rom_mount_path,
        repo_root=repo_root,
        extra_file_mounts=profile.get("extra_file_mounts", {}),
    )

    smoke = profile.get("smoke", {})
    if smoke is False:
        smoke = {"enabled": False}
    if smoke and not isinstance(smoke, dict):
        raise ValueError("smoke must be an object or false")
    smoke_enabled = bool(smoke.get("enabled", True))
    smoke_env = smoke.get("env", {})
    if smoke_env and not isinstance(smoke_env, dict):
        raise ValueError("smoke.env must be an object")
    smoke_state = str(smoke_env.get("state", profile.get("state", "")))
    smoke_states = _manifest_train_value(smoke_env.get("states", profile.get("states", "")))
    smoke_state_probs = _manifest_train_value(
        smoke_env.get("state_probs", profile.get("state_probs", ""))
    )
    smoke_states_tuple = (
        tuple(part.strip() for part in str(smoke_states).split(",") if part.strip())
        if smoke_states
        else ()
    )
    smoke_state_probs_tuple = (
        tuple(float(part.strip()) for part in str(smoke_state_probs).split(",") if part.strip())
        if smoke_state_probs
        else ()
    )
    smoke_n_envs = int(
        smoke_env.get(
            "n_envs",
            len(smoke_states_tuple) if smoke_states_tuple and not smoke_state_probs_tuple else 2,
        )
    )
    smoke_task_conditioning = bool(smoke_env.get("task_conditioning", False))
    smoke_task_conditioning_info_vars = tuple(
        part.strip()
        for part in str(
            _manifest_train_value(smoke_env.get("task_conditioning_info_vars", ""))
        ).split(",")
        if part.strip()
    )
    smoke_task_conditioning_info_values = tuple(
        tuple(
            int(item.strip()) if item.strip().lstrip("-").isdigit() else item.strip()
            for item in row.split(",")
            if item.strip()
        )
        for row in str(
            _manifest_train_value(smoke_env.get("task_conditioning_info_values", ""))
        ).split(";")
        if row.strip()
    )
    smoke_hud_crop_top = int(smoke_env.get("hud_crop_top", -1))
    smoke_reward_mode = str(smoke_env.get("reward_mode", "auto"))
    smoke_action_set = str(smoke_env.get("action_set", "auto"))
    smoke_done_on_info_json = str(smoke_env.get("done_on_info_json", ""))
    smoke_done_on_info = json.loads(smoke_done_on_info_json) if smoke_done_on_info_json else {}
    smoke_completion_x_threshold = int(smoke_env.get("completion_x_threshold", -1))
    smoke_max_pool_frames = bool(smoke_env.get("max_pool_frames", True))
    smoke_learn = bool(smoke.get("learn", smoke_task_conditioning))

    smoke_block = ""
    if smoke_enabled:
        smoke_block = f'''

"$JOB_VENV/bin/python" - <<'PY'
import importlib.metadata
import stable_retro as retro
import torch
from rlab.env import EnvConfig, make_training_vec_env, resolve_env_config, resolve_mixed_state_config

print("stable-retro-turbo", importlib.metadata.version("stable-retro-turbo"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
states_preview = list(retro.data.list_states({game!r}))[:12]
print("states_preview", states_preview)
config = resolve_mixed_state_config(resolve_env_config(EnvConfig(
    game={game!r},
    state={smoke_state!r},
    states={smoke_states_tuple!r},
    state_probs={smoke_state_probs_tuple!r},
    task_conditioning={smoke_task_conditioning!r},
    task_conditioning_info_vars={smoke_task_conditioning_info_vars!r},
    task_conditioning_info_values={smoke_task_conditioning_info_values!r},
    hud_crop_top={smoke_hud_crop_top!r},
    reward_mode={smoke_reward_mode!r},
    action_set={smoke_action_set!r},
    done_on_info={smoke_done_on_info!r},
    completion_x_threshold={smoke_completion_x_threshold!r},
    max_pool_frames={smoke_max_pool_frames!r},
)), n_envs={smoke_n_envs!r})
env = make_training_vec_env(config, n_envs={smoke_n_envs!r}, seed=0)
try:
    obs = env.reset()
    print("runner_smoke_observation_space", env.observation_space)
    print("runner_smoke_reset", type(obs).__name__, getattr(obs, "shape", None))
    if isinstance(obs, dict):
        print("runner_smoke_keys", sorted(obs))
{"    from stable_baselines3 import PPO" if smoke_learn else ""}
{"    PPO('MultiInputPolicy' if isinstance(obs, dict) else 'CnnPolicy', env, n_steps=8, batch_size=4, n_epochs=1, learning_rate=1e-4, device='cpu', verbose=0).learn(total_timesteps=16)" if smoke_learn else ""}
{"    print('runner_smoke_learn=ok')" if smoke_learn else ""}
finally:
    env.close()
PY'''

    if prebuilt_image:
        setup_block = """set -euo pipefail

export RLAB_ROM_DIR="$HOME/roms"
rlab-container-entrypoint rlab-container-smoke"""
    else:
        setup_block = f"""set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y libxcb1 libgl1 libglib2.0-0
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n apt-get update
    sudo -n apt-get install -y libxcb1 libgl1 libglib2.0-0
  else
    apt-get update
    apt-get install -y libxcb1 libgl1 libglib2.0-0
  fi
fi

BOOTSTRAP_PYTHON="${{SKY_RUNTIME_DIR:-$HOME}}/skypilot-runtime/bin/python"
if [ ! -x "$BOOTSTRAP_PYTHON" ]; then
  BOOTSTRAP_PYTHON=/home/sky/skypilot-runtime/bin/python
fi
JOB_VENV="$HOME/{venv_name}"
UV_VENV="$HOME/{venv_name}-uv"

"$BOOTSTRAP_PYTHON" -m venv --clear "$JOB_VENV"
"$BOOTSTRAP_PYTHON" -m venv "$UV_VENV"
"$UV_VENV/bin/python" -m pip install --upgrade pip
"$UV_VENV/bin/python" -m pip install --upgrade "uv>=0.9,<1"

export UV_PROJECT_ENVIRONMENT="$JOB_VENV"
"$UV_VENV/bin/uv" sync --locked --no-dev
"$JOB_VENV/bin/python" -m ensurepip --upgrade
"$JOB_VENV/bin/python" -m pip install --no-deps --force-reinstall "stable-retro-turbo=={stable_retro_turbo_version}"
"$JOB_VENV/bin/python" -m stable_retro.import "$HOME/roms"

"$JOB_VENV/bin/python" - <<'PY'
from pathlib import Path
import shutil
import stable_retro as retro

game = {game!r}
mounted_dir = Path.home() / "roms" / game
rom_path = retro.data.get_romfile_path(game)
if rom_path is not None and mounted_dir.exists():
    package_dir = Path(rom_path).parent
    for source in mounted_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, package_dir / source.name)
    print("synced_mounted_game_data", mounted_dir, "->", package_dir)
PY{smoke_block}"""

    command = [
        "python" if prebuilt_image else '"$PY"',
        "-m",
        "rlab.train_runner",
        "--profile",
        profile_id,
        "--runtime-image-ref",
        runtime_image_ref,
        "--run-target",
        canonical_target,
        "--workers",
        str(workers),
        "--max-jobs",
        str(max_jobs),
        "--log-dir",
        log_dir,
        "--poll-seconds",
        str(poll_seconds),
        "--lease-seconds",
        str(lease_seconds),
        "--heartbeat-seconds",
        str(heartbeat_seconds),
    ]
    if once:
        command.append("--once")
    if direct:
        command.append("--direct")
    if status_goal:
        command.extend(["--status-goal", status_goal])
    if prebuilt_image:
        command = ["rlab-container-entrypoint", *command]

    run_lines = [
        "set -euo pipefail",
        "",
        'export RLAB_ROM_DIR="$HOME/roms"',
    ]
    if not prebuilt_image:
        run_lines.append(f'PY="$HOME/{venv_name}/bin/python"')
    run_lines.extend(
        [
            f"mkdir -p {shell_quote(log_dir)}",
            f'echo "train_profile={profile_id}"',
            f'echo "runtime_image_ref={runtime_image_ref}"',
            f'echo "target={canonical_target} infra={target_description} workers={workers} max_jobs={max_jobs}"',
            shell_join(command),
        ]
    )
    run_block = "\n".join(run_lines)

    resource_lines = [
        "resources:",
        f"  accelerators: {{{accelerator}: 1}}",
        f"  cpus: {cpus}",
        f"  memory: {memory}",
    ]
    if image_id:
        resource_lines.append(f"  image_id: {yaml_string(image_id)}")

    yaml = [
        f"name: {sky_name}",
        "",
    ]
    if not prebuilt_image:
        yaml.extend(["workdir: .", ""])
    yaml.extend(
        [
            *file_mount_lines,
            "",
            *resource_lines,
            "",
            "setup: |",
            indent_block(setup_block, 2),
            "",
            "run: |",
            indent_block(run_block, 2),
        ]
    )
    return "\n".join(yaml) + "\n"


def write_rendered_runner_task(
    profile: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    output_path: Path,
    *,
    target_override: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_runner_task_yaml(profile, instance_config, repo_root, target_override=target_override),
        encoding="utf-8",
    )
    return output_path


def preflight_runner_profile(
    profile: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    env: dict[str, str] | None = None,
    *,
    target_override: str | None = None,
) -> list[Check]:
    env_values = env if env is not None else merged_env(repo_root / ".env")
    checks: list[Check] = []
    target = target_name(profile, target_override)
    try:
        instance = instance_defaults(instance_config, target)
    except ValueError as exc:
        return [Check("error", str(exc))]
    target_label = instance_label(instance)
    kind = target_kind(instance)
    if instance.get("available") is False:
        reason = str(instance.get("disabled_reason") or "target is marked unavailable")
        checks.append(Check("error", f"target {target!r} is unavailable: {reason}"))
    if kind not in {"", "skypilot"}:
        checks.append(
            Check("error", f"target {target!r} has kind {kind!r}; use the matching compute launcher")
        )
    max_children = int(instance.get("max_children", instance.get("children", 1)))
    profile_id = str(profile.get("profile_id", "")).strip()
    if not profile_id:
        checks.append(Check("error", "runner profile must define profile_id"))
    try:
        runner_profile_game(profile)
    except ValueError as exc:
        checks.append(Check("error", str(exc)))
    if not profile.get("rom_source"):
        checks.append(Check("error", "runner profile must define rom_source"))
    image_id = str(profile.get("image_id", instance.get("image_id", ""))).strip()
    prebuilt_image = bool(profile.get("prebuilt_image", instance.get("prebuilt_image", False)))
    runtime_image_ref = str(profile.get("runtime_image_ref", image_id if prebuilt_image else "")).strip()
    try:
        runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    except ValueError as exc:
        checks.append(Check("error", str(exc)))
    if prebuilt_image and not image_id:
        checks.append(Check("error", "prebuilt_image=true requires image_id on the profile or target"))
    if prebuilt_image and image_id and runtime_image_ref and image_id != runtime_image_ref:
        checks.append(Check("error", "prebuilt runner image_id must match runtime_image_ref"))
    if prebuilt_image and str(instance.get("infra", "")).startswith("ssh/"):
        checks.append(
            Check(
                "warning",
                "prebuilt_image=true on an SSH node pool may not run inside Docker; "
                "prefer direct docker or a Kubernetes/RunPod target for container isolation",
            )
        )

    missing = [key for key in REQUIRED_ENV_KEYS if not env_values.get(key)]
    if missing:
        checks.append(Check("error", f"missing env/secrets: {', '.join(missing)}"))
    if not available_database_env_keys(env_values):
        checks.append(
            Check(
                "error",
                "missing train queue database URL: "
                f"set one of {', '.join(DATABASE_ENV_KEYS)}",
            )
        )

    workers = int(profile.get("workers", instance.get("children", 1)))
    if workers > max_children:
        checks.append(Check("warning", f"{workers} workers exceeds {target_label} default {max_children}"))

    max_jobs = int(profile.get("max_jobs", 0))
    if max_jobs > 0 and not profile.get("once"):
        checks.append(Check("warning", "max_jobs is bounded but once=false; runner exits after max_jobs"))

    pyproject = repo_root / "pyproject.toml"
    expected_version = str(
        profile.get("stable_retro_turbo_version", REQUIRED_STABLE_RETRO_TURBO_VERSION)
    ).strip()
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        expected_pin = f"stable-retro-turbo=={expected_version}"
        if expected_pin not in content:
            checks.append(
                Check(
                    "warning",
                    "pyproject.toml does not pin stable-retro-turbo "
                    f"{expected_version}; runner setup will force-reinstall this version",
                )
            )
    else:
        checks.append(Check("warning", "pyproject.toml was not found"))

    rom_source = repo_root / str(profile.get("rom_source", ""))
    if profile.get("rom_source") and not rom_source.exists():
        checks.append(Check("error", f"ROM source path does not exist: {rom_source}"))
    extra_file_mounts = profile.get("extra_file_mounts", {})
    if extra_file_mounts and not isinstance(extra_file_mounts, dict):
        checks.append(Check("error", "extra_file_mounts must be a mapping of remote path to local path"))
    elif isinstance(extra_file_mounts, dict):
        for local_path in extra_file_mounts.values():
            source = repo_root / str(local_path)
            if not source.exists():
                checks.append(Check("error", f"extra file mount source path does not exist: {source}"))
    rom_mount_path = str(profile.get("rom_mount_path", ""))
    if rom_source.suffix and rom_mount_path and not Path(rom_mount_path).name.endswith(rom_source.suffix):
        checks.append(
            Check(
                "error",
                "rom_mount_path must preserve the ROM file extension "
                f"({rom_source.name} -> {rom_mount_path})",
            )
        )

    smoke = profile.get("smoke", {})
    if smoke and not isinstance(smoke, dict):
        checks.append(Check("error", "smoke must be an object or false"))
    elif isinstance(smoke, dict):
        smoke_env = smoke.get("env", {})
        if smoke_env and not isinstance(smoke_env, dict):
            checks.append(Check("error", "smoke.env must be an object"))

    if not any(check.level == "error" for check in checks):
        checks.append(Check("ok", "runner preflight passed with no blocking errors"))
    return checks


def preflight_checks(
    manifest: dict[str, Any],
    instance_config: dict[str, Any],
    repo_root: Path,
    env: dict[str, str] | None = None,
    *,
    target_override: str | None = None,
) -> list[Check]:
    env_values = env if env is not None else merged_env(repo_root / ".env")
    checks: list[Check] = []
    target = target_name(manifest, target_override)
    try:
        instance = instance_defaults(instance_config, target)
    except ValueError as exc:
        return [Check("error", str(exc))]
    target_label = instance_label(instance)
    kind = target_kind(instance)
    if instance.get("available") is False:
        reason = str(instance.get("disabled_reason") or "target is marked unavailable")
        checks.append(Check("error", f"target {target!r} is unavailable: {reason}"))
    if kind not in {"", "skypilot"}:
        checks.append(
            Check("error", f"target {target!r} has kind {kind!r}; use the matching compute launcher")
        )
    max_children = int(instance.get("max_children", instance.get("children", 1)))
    expected_env_threads = int(instance.get("env_threads", 4))
    try:
        manifest_game(manifest)
    except ValueError as exc:
        checks.append(Check("error", str(exc)))
    if not manifest.get("rom_source"):
        checks.append(Check("error", "manifest must define rom_source"))

    missing = [key for key in REQUIRED_ENV_KEYS if not env_values.get(key)]
    if missing:
        checks.append(Check("error", f"missing env/secrets: {', '.join(missing)}"))

    runs = manifest["runs"]
    if len(runs) > max_children:
        checks.append(Check("warning", f"{len(runs)} runs exceeds {target_label} default {max_children}"))

    for index, run in enumerate(runs):
        options = training_options(manifest, run)
        if not str(options.get("run_description", "")).strip():
            checks.append(Check("error", f"runs[{index}] has an empty run_description"))
        if options.get("eval_freq", 0) not in (0, "0"):
            checks.append(Check("warning", f"runs[{index}] enables training-loop eval"))
        if int(options.get("env_threads", expected_env_threads)) != expected_env_threads:
            checks.append(
                Check(
                    "warning",
                    f"runs[{index}] env_threads={options.get('env_threads')} differs from "
                    f"{target_label} default {expected_env_threads}",
                )
            )

    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        expected_version = str(
            manifest.get("stable_retro_turbo_version", REQUIRED_STABLE_RETRO_TURBO_VERSION)
        ).strip()
        expected_pin = f"stable-retro-turbo=={expected_version}"
        if expected_pin not in content:
            checks.append(
                Check(
                    "warning",
                    "pyproject.toml does not pin stable-retro-turbo "
                    f"{expected_version}; launch setup will force-reinstall this version",
                )
            )
    else:
        checks.append(Check("warning", "pyproject.toml was not found"))

    rom_source = repo_root / str(manifest.get("rom_source", ""))
    if manifest.get("rom_source") and not rom_source.exists():
        checks.append(Check("error", f"ROM source path does not exist: {rom_source}"))
    extra_file_mounts = manifest.get("extra_file_mounts", {})
    if extra_file_mounts and not isinstance(extra_file_mounts, dict):
        checks.append(Check("error", "extra_file_mounts must be a mapping of remote path to local path"))
    elif isinstance(extra_file_mounts, dict):
        for local_path in extra_file_mounts.values():
            source = repo_root / str(local_path)
            if not source.exists():
                checks.append(Check("error", f"extra file mount source path does not exist: {source}"))
    rom_mount_path = str(manifest.get("rom_mount_path", ""))
    if rom_source.suffix and rom_mount_path and not Path(rom_mount_path).name.endswith(rom_source.suffix):
        checks.append(
            Check(
                "error",
                "rom_mount_path must preserve the ROM file extension "
                f"({rom_source.name} -> {rom_mount_path})",
            )
        )
    base_train = manifest.get("base_train", {})
    base_storage_uri = base_train.get("wandb_artifact_storage_uri", "") if isinstance(base_train, dict) else ""
    storage_uri = str(manifest.get("wandb_artifact_storage_uri") or base_storage_uri).strip()
    if storage_uri == "${CHECKPOINT_BUCKET_URI}":
        storage_uri = env_values.get("CHECKPOINT_BUCKET_URI", "").strip()
    if storage_uri and storage_uri != "${CHECKPOINT_BUCKET_URI}":
        if not storage_uri.startswith("s3://"):
            checks.append(Check("error", "wandb_artifact_storage_uri must be an s3:// URI"))
        else:
            prefix = storage_uri.removeprefix("s3://").split("/", 1)[1] if "/" in storage_uri[5:] else ""
            expected_prefix = artifact_storage_prefix(prefix, manifest_game(manifest))
            if expected_prefix != prefix.rstrip("/"):
                checks.append(
                    Check(
                        "warning",
                        "artifact storage URI does not include the game id; training will append "
                        f"{manifest_game(manifest)!r} below the configured prefix",
                    )
                )

    if not any(check.level == "error" for check in checks):
        checks.append(Check("ok", "preflight passed with no blocking errors"))
    return checks


def launch_summary(
    manifest_path: Path,
    output_path: Path,
    repo_root: Path,
    instance_config_path: Path | None = None,
    *,
    target_override: str | None = None,
    detach_run: bool = False,
) -> LaunchSummary:
    manifest = load_manifest(manifest_path)
    instance_config = load_instance_config(repo_root, instance_config_path)
    target = target_name(manifest, target_override)
    instance = instance_defaults(instance_config, target)
    task_path = write_rendered_task(
        manifest,
        instance_config,
        repo_root,
        output_path,
        target_override=target,
    )
    cluster = str(manifest.get("cluster", manifest["name"]))
    return LaunchSummary(
        command=build_launch_command(cluster, task_path, infra=launch_infra(instance), detach_run=detach_run),
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(manifest.get("wandb_group_prefix", manifest["name"])),
    )


METRIC_ROW_RE = re.compile(r"\|\s+(?P<key>[A-Za-z0-9_./-]+)\s+\|\s+(?P<value>[^|]+?)\s+\|")


def _parse_float(value: str) -> float | None:
    try:
        return float(value.strip().replace(",", ""))
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value.strip().replace(",", "")))
    except ValueError:
        return None


class SparseLogMonitor:
    def __init__(self) -> None:
        self._done_thresholds = [1, 10, 100, 1_000, 10_000]
        self._seen_done_thresholds: set[int] = set()
        self._seen_wandb_urls: set[str] = set()
        self._seen_artifact_steps: set[int] = set()
        self._last_step: int | None = None
        self._metric_section = ""

    def observe(self, line: str) -> list[SparseEvent]:
        events: list[SparseEvent] = []
        text = line.strip()
        if not text:
            return events
        if "wandb.ai/" in text:
            for url in re.findall(r"https://wandb\.ai/\S+", text):
                if url not in self._seen_wandb_urls:
                    self._seen_wandb_urls.add(url)
                    events.append(SparseEvent("wandb", f"wandb: {url}"))
        if "wandb artifact logged:" in text:
            step = checkpoint_step(Path(text.split("(", 1)[0]))
            should_emit = "final" in text
            if step is not None:
                should_emit = step == 0 or step % 1_000_000 == 0
                self._seen_artifact_steps.add(step)
            if should_emit:
                events.append(SparseEvent("artifact", text))
        if "early stop" in text.lower() or "early_stop" in text:
            events.append(SparseEvent("early_stop", text))
        if "exit status:" in text or "failed" in text.lower() or "error" in text.lower():
            events.append(SparseEvent("status", text))

        metric = METRIC_ROW_RE.search(text)
        if metric:
            key = metric.group("key").strip()
            value = metric.group("value").strip()
            if key.endswith("/") and not value:
                self._metric_section = key.rstrip("/")
                return events
            metric_key = (
                key
                if key == "total_timesteps" or "/" in key or not self._metric_section
                else f"{self._metric_section}/{key}"
            )
            if metric_key in {"total_timesteps", "time/total_timesteps"}:
                self._last_step = _parse_int(value)
            elif metric_key == TRAIN_DONE_ALL:
                total = _parse_int(value)
                if total is not None:
                    for threshold in self._done_thresholds:
                        if total >= threshold and threshold not in self._seen_done_thresholds:
                            self._seen_done_thresholds.add(threshold)
                            step = f" at {self._last_step} steps" if self._last_step else ""
                            events.append(
                                SparseEvent("done", f"done count crossed {threshold}{step}: {total}"),
                            )
        return events


def sparse_log_events(text: str) -> list[SparseEvent]:
    monitor = SparseLogMonitor()
    events: list[SparseEvent] = []
    for line in text.splitlines():
        events.extend(monitor.observe(line))
    return events


def execute_launch(
    summary: LaunchSummary,
    repo_root: Path,
    dotenv_path: Path,
    *,
    sparse: bool = False,
    log_path: Path | None = None,
    down_on_complete: bool = False,
) -> int:
    env = os.environ.copy()
    env.update(load_dotenv(dotenv_path))
    if not sparse:
        returncode = subprocess.run(summary.command, cwd=repo_root, env=env, check=False).returncode
    else:
        output_path = log_path or repo_root / "logs" / f"{summary.task_path.stem}.sky.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        monitor = SparseLogMonitor()
        print(f"sparse_log: {output_path}")
        with output_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                summary.command,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                for event in monitor.observe(line):
                    print(f"{event.kind}: {event.message}")
            returncode = process.wait()
    if down_on_complete:
        subprocess.run(cleanup_command(summary.cluster), cwd=repo_root, env=env, check=False)
    return returncode


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def metric_rows(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    section = ""
    for line in text.splitlines():
        metric = METRIC_ROW_RE.search(line)
        if not metric:
            continue
        key = metric.group("key").strip()
        value = metric.group("value").strip()
        if key.endswith("/") and not value:
            section = key.rstrip("/")
            continue
        metric_key = key if key == "total_timesteps" or "/" in key or not section else f"{section}/{key}"
        rows[metric_key] = value
    return rows


def parse_log_summary(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, str] = {}
    exit_matches = re.findall(r"exit status: (\d+)", text)
    if exit_matches:
        result["exit_status"] = exit_matches[-1]
    artifact_locations = re.findall(r"wandb artifact logged: .+ \(([^)]+)\)", text)
    if artifact_locations:
        result["artifact_location"] = artifact_locations[-1]
        result["artifact_plane"] = "r2" if artifact_locations[-1].startswith("s3://") else "wandb"
    wandb_matches = re.findall(r"https://wandb\.ai/\S+", text)
    if wandb_matches:
        result["wandb_url"] = wandb_matches[-1]
    metrics = metric_rows(text)
    if total_steps := metrics.get("time/total_timesteps") or metrics.get("total_timesteps"):
        result["timesteps"] = total_steps.replace(",", "")
    if done_total := metrics.get(TRAIN_DONE_ALL):
        result["done_all"] = done_total.replace(",", "")
    return result


def launch_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = parse_log_summary(path)
    events = sparse_log_events(text)
    artifacts = re.findall(r"wandb artifact logged: ([^ ]+) \(([^)]+)\)", text)
    report: dict[str, Any] = {
        "log": str(path),
        "summary": summary,
        "events": [{"kind": event.kind, "message": event.message} for event in events],
        "artifacts": [
            {"name": name, "location": location, "plane": "r2" if location.startswith("s3://") else "wandb"}
            for name, location in artifacts
        ],
    }
    return report


def write_launch_report(log_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = launch_report(log_path)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def collect_results(log_dir: Path, runs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for log_path in sorted(log_dir.glob("*.log")):
        run_name = log_path.stem
        row = {"run_name": run_name, "log": str(log_path)}
        row.update(parse_log_summary(log_path))
        run_dir = runs_dir / run_name
        row.update(parse_key_value_file(run_dir / "early_stop.txt"))
        if (run_dir / "wandb_url.txt").exists():
            row["wandb_url"] = (run_dir / "wandb_url.txt").read_text(encoding="utf-8").strip()
        if (run_dir / "wandb_run_id.txt").exists():
            row["wandb_run_id"] = (run_dir / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
        rows.append(row)
    return rows


def format_results_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run_name",
        "exit_status",
        "timesteps",
        "done_all",
        "artifact_plane",
        "wandb_url",
    ]
    if not rows:
        return "No log files found."
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows)) for header in headers
    }
    lines = [
        " | ".join(header.ljust(widths[header]) for header in headers),
        " | ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(" | ".join(row.get(header, "").ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def cleanup_command(cluster: str) -> list[str]:
    return ["sky", "down", "-y", cluster]


def configured_api_endpoints(instance_config: dict[str, Any], instance_name: str = "rtx4090") -> list[str]:
    instances = instance_config.get("instances", {})
    instance = instances.get(instance_name, {}) if isinstance(instances, dict) else {}
    values: list[str] = []
    if isinstance(instance, dict):
        endpoints = instance.get("api_endpoints", [])
        if isinstance(endpoints, list):
            values.extend(str(endpoint) for endpoint in endpoints if str(endpoint).strip())
        api_url = instance.get("api_url")
        if api_url:
            values.append(str(api_url))
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def check_api_endpoint(endpoint: str, timeout: float = 2.0) -> EndpointCheck:
    candidates = [endpoint.rstrip("/"), f"{endpoint.rstrip('/')}/api/health"]
    last_error = "unreachable"
    for candidate in candidates:
        try:
            with urlopen(candidate, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status < 500:
                    return EndpointCheck(endpoint, True, f"healthy via {candidate}")
                last_error = f"HTTP {status}"
        except (OSError, URLError) as exc:
            last_error = str(exc)
    return EndpointCheck(endpoint, False, last_error)


def ensure_skypilot_api(
    instance_config: dict[str, Any],
    *,
    repo_root: Path,
    instance_name: str = "rtx4090",
    execute: bool = False,
) -> tuple[list[EndpointCheck], list[str] | None]:
    checks = [check_api_endpoint(endpoint) for endpoint in configured_api_endpoints(instance_config, instance_name)]
    selected = next((check.endpoint for check in checks if check.ok), None)
    command = ["sky", "api", "login", "-e", selected] if selected else None
    if execute and command is not None:
        subprocess.run(command, cwd=repo_root, check=False)
    return checks, command
