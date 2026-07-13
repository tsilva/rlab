from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from rlab.artifacts import resolve_artifact_storage_uri
from rlab.job_metadata import (
    append_unique_wandb_tag,
    normalize_level_states,
    normalize_wandb_tags,
)
from rlab.provider_config import provider_num_envs
from rlab.recipe_schema import require_explicit_queue_train_config
from rlab.metric_store import MetricStore, metric_store_path
from rlab.seeds import validate_training_seed


ARTIFACT_RE = re.compile(r"wandb artifact logged: (?P<name>[^ ]+) \((?P<location>[^)]+)\)")
METRIC_ROW_RE = re.compile(r"\|\s+(?P<key>[A-Za-z0-9_./-]+)\s+\|\s+(?P<value>[^|]+?)\s+\|")
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+")


def normalize_train_config(
    job: dict[str, Any],
    *,
    require_explicit_train_fields: bool = True,
) -> dict[str, Any]:
    """Merge queue-row execution metadata into an already materialized recipe train_config.

    Recipe composition and train-config materialization are owned by
    rlab.recipe_documents; this function only adds execution metadata needed by
    the one-job container path.
    """

    config = dict(job.get("train_config") or {})
    run_name = job.get("run_name") or config.get("run_name") or f"train_job_{job['id']}"
    config["run_name"] = run_name
    if job.get("run_description"):
        config["run_description"] = job["run_description"]
    if job.get("wandb_group"):
        config["wandb_group"] = job["wandb_group"]
    tags = normalize_wandb_tags(config.get("wandb_tags") or job.get("wandb_tags"))
    goal_slug = str(job.get("goal_slug") or "").strip()
    if goal_slug:
        config["goal_slug"] = goal_slug
        append_unique_wandb_tag(tags, f"goal_id:{goal_slug}")
    recipe_slug = str(job.get("recipe_slug") or "").strip()
    if recipe_slug:
        config["recipe_slug"] = recipe_slug
        append_unique_wandb_tag(tags, f"recipe_id:{recipe_slug}")
    recipe_path = str(job.get("recipe_path") or "").strip()
    if recipe_path:
        config["recipe_path"] = recipe_path
    if job.get("seed") is not None:
        config["seed"] = int(job["seed"])
    if job.get("id") is not None:
        config["queue_train_job_id"] = int(job["id"])
    for level in normalize_level_states(config):
        append_unique_wandb_tag(tags, f"level_id:{level}")
    if tags:
        config["wandb_tags"] = ",".join(tags)
    if "seed" in config and config["seed"] is not None:
        validate_training_seed(
            config["seed"],
            label="train_config.seed",
            seed_span=provider_num_envs(config, explicit_n_envs=config.get("n_envs")),
        )
    if job.get("runtime_image_ref"):
        config["runtime_image_ref"] = job["runtime_image_ref"]
    if job.get("machine"):
        config["machine"] = job["machine"]
    if require_explicit_train_fields:
        require_explicit_queue_train_config(config)
    config["wandb_artifact_storage_uri"] = resolve_artifact_storage_uri(
        str(config.get("wandb_artifact_storage_uri") or ""),
        os.environ,
        allow_environment_fallback=False,
    )
    return config


def write_train_config_file(
    job: dict[str, Any], path: Path, *, runs_dir: Path | str | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = normalize_train_config(job)
    if runs_dir is not None:
        config["runs_dir"] = str(runs_dir)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def train_command_for_job(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "rlab.train", "--train-config-json", str(config_path)]


def read_text_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_metric_value(value: str) -> int | float | str:
    text = value.strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    if number.is_integer() and not any(marker in text.lower() for marker in (".", "e")):
        return int(number)
    return number


def parse_log_metrics(log_text: str) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    section = ""
    for line in log_text.splitlines():
        match = METRIC_ROW_RE.search(line)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key.endswith("/") and not value:
            section = key.rstrip("/")
            continue
        metric_key = key if key == "total_timesteps" or "/" in key or not section else f"{section}/{key}"
        parsed = parse_metric_value(value)
        metrics[metric_key] = parsed
        if metric_key in {"total_timesteps", "time/total_timesteps"}:
            metrics[key] = parsed
    return metrics


def parse_wandb_run_url(log_text: str) -> str | None:
    matches = WANDB_RUN_URL_RE.findall(log_text)
    return matches[-1] if matches else None


def collect_result_metadata(job: dict[str, Any], log_path: Path) -> dict[str, Any]:
    config = normalize_train_config(
        job,
        require_explicit_train_fields=False,
    )
    run_name = str(config["run_name"])
    runs_dir = str(config.get("runs_dir") or "runs")
    run_dir = Path(runs_dir) / run_name
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    artifact_refs = [
        {"name": match.group("name"), "location": match.group("location")}
        for match in ARTIFACT_RE.finditer(log_text)
    ]
    metrics = parse_log_metrics(log_text)
    metrics.update(parse_key_value_file(run_dir / "early_stop.txt"))
    checkpoint_eval_summary_path = run_dir / "checkpoint_eval_summary.json"
    checkpoint_eval_summary = []
    if checkpoint_eval_summary_path.is_file():
        try:
            loaded_summary = json.loads(checkpoint_eval_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, list):
                checkpoint_eval_summary = loaded_summary
        except json.JSONDecodeError:
            checkpoint_eval_summary = []
    phase_counts = {}
    telemetry_health = {}
    store_path = metric_store_path(run_dir)
    if store_path.is_file():
        try:
            store = MetricStore(store_path, timeout=0.05)
            phase_counts = store.phase_counts()
            telemetry_health = store.telemetry_health()
        except Exception:
            phase_counts = {}
            telemetry_health = {}
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_model_path": str(run_dir / "final_model.zip")
        if (run_dir / "final_model.zip").is_file()
        else None,
        "wandb_run_id": read_text_file(run_dir / "wandb_run_id.txt"),
        "wandb_url": read_text_file(run_dir / "wandb_url.txt") or parse_wandb_run_url(log_text),
        "artifact_refs": artifact_refs,
        "checkpoint_eval_summary": checkpoint_eval_summary,
        "metrics_json": metrics,
        "phase_counts": phase_counts,
        "telemetry_health": telemetry_health,
    }
