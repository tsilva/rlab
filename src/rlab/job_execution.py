from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.artifacts import resolve_artifact_storage_uri
from rlab.job_metadata import (
    append_unique_wandb_tag,
    normalize_level_states,
    normalize_wandb_tags,
)
from rlab.provider_config import provider_num_envs
from rlab.policy_bundle import (
    RECIPE_DOCUMENT_TYPE,
    load_recipe_document,
    write_canonical_json,
)
from rlab.recipe_schema import require_explicit_queue_train_config
from rlab.metric_store import MetricStore, metric_store_path
from rlab.seeds import validate_training_seed
from rlab.wandb_utils import game_family_for_environment


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
    batch_id = str(job.get("batch_id") or config.get("batch_id") or "").strip()
    if batch_id:
        config["batch_id"] = batch_id
    campaign_id = str(job.get("campaign_id") or config.get("campaign_id") or "").strip()
    if campaign_id:
        config["campaign_id"] = campaign_id
        append_unique_wandb_tag(tags, f"campaign_id:{campaign_id}")
    retry_of_job_id = job.get("retry_of_job_id") or config.get("retry_of_job_id")
    if retry_of_job_id:
        config["retry_of_job_id"] = int(retry_of_job_id)
        append_unique_wandb_tag(tags, f"retry_of_job_id:{int(retry_of_job_id)}")
    game_family = game_family_for_environment(
        config.get("env_provider"),
        config.get("game"),
    )
    config["game_family"] = game_family
    append_unique_wandb_tag(tags, f"game_family:{game_family}")
    goal_slug = str(job.get("goal_slug") or "").strip()
    if goal_slug:
        config["goal_slug"] = goal_slug
        append_unique_wandb_tag(tags, f"goal_id:{goal_slug}")
    goal_path = str(job.get("goal_path") or config.get("goal_path") or "").strip()
    if goal_path:
        config["goal_path"] = goal_path
    goal_sha256 = str(job.get("goal_sha256") or config.get("goal_sha256") or "").strip()
    if goal_sha256:
        config["goal_sha256"] = goal_sha256
    recipe_slug = str(job.get("recipe_slug") or "").strip()
    if recipe_slug:
        config["recipe_slug"] = recipe_slug
        append_unique_wandb_tag(tags, f"recipe_id:{recipe_slug}")
    recipe_path = str(job.get("recipe_path") or "").strip()
    if recipe_path:
        config["recipe_path"] = recipe_path
    recipe_sha256 = str(job.get("recipe_sha256") or config.get("recipe_sha256") or "").strip()
    if recipe_sha256:
        config["recipe_sha256"] = recipe_sha256
    recipe_payload = job.get("recipe_payload_json")
    if isinstance(recipe_payload, Mapping):
        composition = recipe_payload.get("_composition")
        if isinstance(composition, Mapping):
            config["recipe_composition"] = dict(composition)
        elif recipe_payload.get("document_type") == RECIPE_DOCUMENT_TYPE:
            provenance = recipe_payload.get("provenance")
            if isinstance(provenance, Mapping):
                config["recipe_composition"] = {
                    "source_files": list(provenance.get("source_files") or [])
                }
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
    if job.get("repo_git_commit"):
        config["source_sha"] = str(job["repo_git_commit"])
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
    recipe_payload = job.get("recipe_payload_json")
    if (
        isinstance(recipe_payload, Mapping)
        and recipe_payload.get("document_type") == RECIPE_DOCUMENT_TYPE
    ):
        recipe_path = path.with_name("recipe.json")
        write_canonical_json(recipe_path, recipe_payload)
        load_recipe_document(recipe_path)
        config["recipe_json_path"] = str(recipe_path)
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


def collect_result_metadata(job: dict[str, Any]) -> dict[str, Any]:
    config = normalize_train_config(
        job,
        require_explicit_train_fields=False,
    )
    run_name = str(config["run_name"])
    runs_dir = str(config.get("runs_dir") or "runs")
    run_dir = Path(runs_dir) / run_name
    projection: dict[str, Any] = {
        "artifact_refs": [],
        "metrics_json": {},
        "phase_counts": {},
        "telemetry_health": {},
    }
    checkpoint_eval_summary_path = run_dir / "checkpoint_eval_summary.json"
    checkpoint_eval_summary = []
    if checkpoint_eval_summary_path.is_file():
        try:
            loaded_summary = json.loads(checkpoint_eval_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded_summary, list):
                checkpoint_eval_summary = loaded_summary
        except json.JSONDecodeError:
            checkpoint_eval_summary = []
    store_path = metric_store_path(run_dir)
    if store_path.is_file():
        try:
            projection = MetricStore(store_path, timeout=0.05).result_projection()
        except Exception:
            pass
    metrics = dict(projection["metrics_json"])
    metrics.update(parse_key_value_file(run_dir / "early_stop.txt"))
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_model_path": str(run_dir / "final_model.zip")
        if (run_dir / "final_model.zip").is_file()
        else None,
        "wandb_run_id": read_text_file(run_dir / "wandb_run_id.txt"),
        "wandb_url": read_text_file(run_dir / "wandb_url.txt"),
        "artifact_refs": projection["artifact_refs"],
        "checkpoint_eval_summary": checkpoint_eval_summary,
        "metrics_json": metrics,
        "phase_counts": projection["phase_counts"],
        "telemetry_health": projection["telemetry_health"],
    }
