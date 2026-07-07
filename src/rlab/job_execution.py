from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.seeds import validate_training_seed
from rlab.spec_schema import require_explicit_queue_train_config
from rlab.wandb_artifacts import artifact_download_dir, download_model_artifact


ARTIFACT_RE = re.compile(r"wandb artifact logged: (?P<name>[^ ]+) \((?P<location>[^)]+)\)")
METRIC_ROW_RE = re.compile(r"\|\s+(?P<key>[A-Za-z0-9_./-]+)\s+\|\s+(?P<value>[^|]+?)\s+\|")
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+")
LEVEL_STATE_RE = re.compile(r"^Level\d+-\d+$")
RESUME_ARTIFACT_ROOT = Path("artifacts/train_resumes")


def strip_env_file_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def normalize_wandb_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = value.split(",")
    elif isinstance(value, list | tuple):
        raw_tags = value
    else:
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def append_unique_wandb_tag(tags: list[str], tag: str) -> list[str]:
    tag = tag.strip()
    if tag and tag not in set(tags):
        tags.append(tag)
    return tags


def normalize_level_states(config: Mapping[str, Any]) -> list[str]:
    raw_states = config.get("states") or []
    if isinstance(raw_states, str):
        candidates = [raw_states]
    elif isinstance(raw_states, list | tuple):
        candidates = [str(state) for state in raw_states]
    else:
        candidates = []

    state = str(config.get("state") or "").strip()
    if state:
        candidates.append(state)

    levels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        level = candidate.strip()
        if not LEVEL_STATE_RE.match(level) or level in seen:
            continue
        seen.add(level)
        levels.append(level)
    return levels


def normalize_train_config(
    job: dict[str, Any],
    *,
    resolve_resume_artifact: bool = True,
    require_explicit_train_fields: bool = True,
) -> dict[str, Any]:
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
            seed_span=config.get("n_envs", 1),
        )
    if job.get("runtime_image_ref"):
        config["runtime_image_ref"] = job["runtime_image_ref"]
    if job.get("run_target"):
        config["run_target"] = job["run_target"]
    resume_artifact = config.pop("resume_artifact", None)
    if resume_artifact:
        if config.get("resume"):
            raise ValueError("Use only one of resume or resume_artifact in train_config")
        if resolve_resume_artifact:
            resume_ref = str(resume_artifact)
            config["resume"] = str(
                download_model_artifact(
                    resume_ref,
                    artifact_download_dir(RESUME_ARTIFACT_ROOT, resume_ref),
                )
            )
    if require_explicit_train_fields:
        require_explicit_queue_train_config(config)
    if config.get("wandb_artifact_storage_uri") in {
        "${CHECKPOINT_BUCKET_URI}",
        "$CHECKPOINT_BUCKET_URI",
        "CHECKPOINT_BUCKET_URI",
    }:
        config["wandb_artifact_storage_uri"] = strip_env_file_quotes(
            os.environ.get("CHECKPOINT_BUCKET_URI", "")
        )
    return config


def write_train_config_file(job: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_train_config(job), indent=2, sort_keys=True) + "\n",
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
        resolve_resume_artifact=False,
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
    }


def should_purge_successful_run_data(job: dict[str, Any], result: Mapping[str, Any]) -> bool:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    if not bool(config.get("wandb")):
        return False
    if str(config.get("wandb_mode") or "online") != "online":
        return False
    if bool(config.get("no_wandb_artifacts")):
        return False
    return bool(result.get("artifact_refs"))


def purge_successful_run_data(job: dict[str, Any], result: Mapping[str, Any]) -> bool:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    runs_dir = Path(str(config.get("runs_dir") or "runs"))
    raw_run_dir = str(result.get("run_dir") or "").strip()
    if not raw_run_dir:
        return False
    run_dir = Path(raw_run_dir)
    try:
        runs_root = runs_dir.resolve()
        target = run_dir.resolve()
    except OSError as exc:
        print(f"warning: could not resolve run cleanup paths run_dir={run_dir}: {exc}", flush=True)
        return False
    if target == runs_root or not target.is_relative_to(runs_root):
        print(
            f"warning: refusing to purge run_dir outside runs_dir: run_dir={target} runs_dir={runs_root}",
            flush=True,
        )
        return False
    if not target.exists():
        return False
    try:
        shutil.rmtree(target)
    except OSError as exc:
        print(f"warning: could not purge successful run data {target}: {exc}", flush=True)
        return False
    print(f"purged successful run data: {target}", flush=True)
    return True
