from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from rlab.artifacts import (
    env_config_from_config_dict,
    env_config_from_model_metadata,
    model_metadata_path,
)
from rlab.job_queue import (
    claim_eval_job,
    connect,
    database_url,
    finish_eval_job,
    heartbeat_eval_job,
    new_worker_id,
    print_status,
    queue_status,
    record_eval_wandb_status,
)
from rlab.device import resolve_sb3_device
from rlab.env import EnvConfig, resolve_env_config
from rlab.eval_metrics import flat_numeric_metrics
from rlab.eval_runner import evaluate_model_episodes
from rlab.json_utils import json_safe
from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_BEST_X,
    EVAL_BEST_VIDEO,
    EVAL_CHECKPOINT_ARTIFACT,
    EVAL_CHECKPOINT_STEP,
    EVAL_CONFIG_HUD_CROP_TOP,
    EVAL_DEATH_COUNT,
    EVAL_DEATH_RATE,
    EVAL_PROGRESS_LEVEL_X_MAX,
    EVAL_PROGRESS_LEVEL_X_MEAN,
    EVAL_PROGRESS_X_MAX,
    EVAL_PROGRESS_X_MEAN,
    EVAL_REWARD_MAX,
    EVAL_REWARD_MEAN,
    EVAL_REWARD_STD,
    GLOBAL_STEP,
)
from rlab.model_sources import split_project
from rlab.seeds import DEFAULT_EVAL_SEED, validate_eval_seed
from rlab.wandb_artifacts import artifact_download_dir, download_model_artifact
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def normalize_eval_config(job: dict[str, Any]) -> dict[str, Any]:
    config = dict(job.get("eval_config") or {})
    config.setdefault("episodes", 100)
    config.setdefault("seed", DEFAULT_EVAL_SEED)
    config["seed"] = validate_eval_seed(config["seed"], label="eval_config.seed")
    config.setdefault("n_envs", 20)
    config.setdefault("max_steps", 4500)
    config.setdefault("stochastic", True)
    config.setdefault("device", "auto")
    config.setdefault("capture_best_video", False)
    return config


def model_ref_for_config(config: dict[str, Any]) -> str:
    return str(
        config.get("artifact_ref")
        or config.get("model_artifact")
        or config.get("model_path")
        or ""
    )


def resolve_model_path(config: dict[str, Any], *, artifact_root: Path) -> Path:
    artifact_ref = config.get("artifact_ref") or config.get("model_artifact")
    if artifact_ref:
        ref = str(artifact_ref)
        return download_model_artifact(ref, artifact_download_dir(artifact_root, ref))
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("eval_config must define artifact_ref, model_artifact, or model_path")
    path = Path(str(model_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"model_path does not exist: {path}")
    return path


def read_model_metadata(model_path: Path) -> dict[str, Any]:
    path = model_metadata_path(model_path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step_for_job(job: dict[str, Any], config: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    for value in (job.get("checkpoint_step"), config.get("checkpoint_step"), metadata.get("checkpoint_step")):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def wandb_run_id_for_job(job: dict[str, Any], config: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    for value in (
        job.get("wandb_run_id"),
        config.get("wandb_run_id"),
        metadata.get("wandb_run_id"),
    ):
        if value:
            return str(value)
    run_path = metadata.get("wandb_run_path") or config.get("wandb_run_path")
    if run_path:
        return str(run_path).rstrip("/").rsplit("/", 1)[-1]
    return None


def wandb_project_path_for_metadata(config: dict[str, Any], metadata: dict[str, Any]) -> str:
    for value in (config.get("wandb_project_path"), metadata.get("wandb_project_path")):
        if value:
            return str(value)
    run_path = metadata.get("wandb_run_path") or config.get("wandb_run_path")
    if run_path:
        parts = str(run_path).strip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return DEFAULT_WANDB_PROJECT_PATH


def log_eval_to_wandb(
    *,
    job: dict[str, Any],
    config: dict[str, Any],
    model_path: Path,
    metrics: dict[str, Any],
    video_path: Path | None,
) -> dict[str, Any]:
    metadata = read_model_metadata(model_path)
    checkpoint_step = checkpoint_step_for_job(job, config, metadata)
    run_id = wandb_run_id_for_job(job, config, metadata)
    if checkpoint_step is None:
        return {"wandb_run_id": run_id, "wandb_logged": False, "wandb_log_error": "missing checkpoint_step"}
    if not run_id:
        return {"wandb_logged": False, "wandb_log_step": checkpoint_step, "wandb_log_error": "missing wandb_run_id"}

    load_wandb_env()
    import wandb

    entity, project = split_project(wandb_project_path_for_metadata(config, metadata))
    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow", mode=config.get("wandb_mode", "online"))
    try:
        payload: dict[str, Any] = {
            GLOBAL_STEP: checkpoint_step,
            EVAL_CHECKPOINT_STEP: checkpoint_step,
            EVAL_CHECKPOINT_ARTIFACT: model_ref_for_config(config),
            "eval/queue/job_id": int(job["id"]),
            "eval/protocol/hash": job.get("eval_protocol_hash") or config.get("eval_protocol_hash"),
            EVAL_REWARD_MEAN: metrics["reward_mean"],
            EVAL_REWARD_STD: metrics["reward_std"],
            EVAL_REWARD_MAX: metrics["reward_max"],
            EVAL_PROGRESS_X_MEAN: metrics["max_x_mean"],
            EVAL_PROGRESS_X_MAX: metrics["max_x_max"],
            EVAL_PROGRESS_LEVEL_X_MEAN: metrics["max_level_x_mean"],
            EVAL_PROGRESS_LEVEL_X_MAX: metrics["max_level_x_max"],
            EVAL_DEATH_COUNT: metrics["death_count"],
            EVAL_DEATH_RATE: metrics["death_rate"],
            EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
            EVAL_BEST_X: metrics["best_episode"]["max_x_pos"],
        }
        if "hud_crop_top" in metrics:
            payload[EVAL_CONFIG_HUD_CROP_TOP] = metrics["hud_crop_top"]
        for key, value in metrics.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                payload[key] = value
        payload.update(flat_numeric_metrics(metrics, "eval/done/"))
        payload.update(flat_numeric_metrics(metrics, "eval/info/"))
        if video_path is not None and video_path.is_file():
            payload[EVAL_BEST_VIDEO] = wandb.Video(str(video_path), format="mp4")
        run.log(payload, step=checkpoint_step)
    finally:
        run.finish()
    return {"wandb_run_id": run_id, "wandb_logged": True, "wandb_log_step": checkpoint_step}


def _env_config_overrides(config: dict[str, Any]) -> dict[str, Any]:
    field_names = set(EnvConfig.__dataclass_fields__)
    overrides = {key: value for key, value in config.items() if key in field_names}
    if isinstance(overrides.get("states"), list):
        overrides["states"] = tuple(overrides["states"])
    if isinstance(overrides.get("state_probs"), list):
        overrides["state_probs"] = tuple(overrides["state_probs"])
    return overrides


def eval_env_config(config: dict[str, Any], model_path: Path) -> EnvConfig:
    base = env_config_from_model_metadata(model_path, fallback=EnvConfig()) or EnvConfig()
    env_config_payload = config.get("env_config")
    if isinstance(env_config_payload, dict):
        base = env_config_from_config_dict(env_config_payload, fallback=base) or base
    overrides = _env_config_overrides(config)
    return resolve_env_config(replace(base, **overrides))


def write_eval_output(
    *,
    output_dir: Path,
    job: dict[str, Any],
    config: dict[str, Any],
    model_path: Path,
    env_config: EnvConfig,
    metrics: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_job_{job['id']}_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "job_id": int(job["id"]),
        "candidate_label": job.get("candidate_label"),
        "model_ref": model_ref_for_config(config),
        "model_path": str(model_path),
        "eval_config": config,
        "env_config": asdict(env_config),
        "metrics": json_safe(metrics),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_eval_job(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    lease_seconds: int,
    artifact_root: Path,
    output_dir: Path,
) -> str:
    config = normalize_eval_config(job)
    heartbeat = heartbeat_eval_job(
        conn,
        job_id=int(job["id"]),
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    if heartbeat is None or heartbeat.get("cancel_requested"):
        finish_eval_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="canceled",
            result={"candidate_label": job.get("candidate_label")},
            error="cancel requested or lease lost before eval",
        )
        return "canceled"

    try:
        model_path = resolve_model_path(config, artifact_root=artifact_root)
        env_config = eval_env_config(config, model_path)
        model = PPO.load(model_path, device=resolve_sb3_device(str(config["device"])))
        video_path = (
            output_dir / f"eval_job_{job['id']}_best_episode.mp4"
            if config.get("capture_best_video")
            else None
        )
        metrics, written_video = evaluate_model_episodes(
            model=model,
            config=env_config,
            episodes=int(config["episodes"]),
            seed=int(config["seed"]),
            max_steps=int(config["max_steps"]),
            deterministic=not bool(config["stochastic"]),
            completion_x_threshold=int(env_config.completion_x_threshold),
            n_envs=int(config["n_envs"]),
            capture_best_video=bool(config.get("capture_best_video")),
            video_path=video_path,
            extra={
                "queue_eval_job_id": int(job["id"]),
                "queue_candidate_label": job.get("candidate_label"),
                "checkpoint_step": job.get("checkpoint_step"),
                "checkpoint_artifact": model_ref_for_config(config),
                "eval_protocol_hash": job.get("eval_protocol_hash"),
            },
        )
        safe_metrics = json_safe(metrics)
        output_path = write_eval_output(
            output_dir=output_dir,
            job=job,
            config=config,
            model_path=model_path,
            env_config=env_config,
            metrics=safe_metrics,
        )
        finish_eval_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="succeeded",
            result={
                "candidate_label": job.get("candidate_label"),
                "model_ref": model_ref_for_config(config),
                "output_path": str(output_path),
                "video_path": str(written_video) if written_video else None,
                "artifact_ref": job.get("artifact_ref") or model_ref_for_config(config),
                "checkpoint_step": job.get("checkpoint_step"),
                "eval_protocol_hash": job.get("eval_protocol_hash"),
                "metrics_json": safe_metrics,
            },
        )
        try:
            wandb_result = log_eval_to_wandb(
                job=job,
                config=config,
                model_path=model_path,
                metrics=safe_metrics,
                video_path=written_video,
            )
        except Exception as exc:
            wandb_result = {
                "wandb_run_id": None,
                "wandb_logged": False,
                "wandb_log_step": job.get("checkpoint_step"),
                "wandb_log_error": repr(exc),
            }
        record_eval_wandb_status(
            conn,
            eval_job_id=int(job["id"]),
            wandb_run_id=wandb_result.get("wandb_run_id"),
            logged=bool(wandb_result.get("wandb_logged")),
            log_step=wandb_result.get("wandb_log_step"),
            error=wandb_result.get("wandb_log_error"),
        )
        print(
            "eval_job="
            f"{job['id']} completion_rate={metrics['completion_rate']:.3f} "
            f"episodes={metrics['episodes']} state={env_config.state}",
            flush=True,
        )
        return "succeeded"
    except Exception as exc:
        finish_eval_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="failed",
            result={
                "candidate_label": job.get("candidate_label"),
                "model_ref": model_ref_for_config(config),
            },
            error=repr(exc),
        )
        print(f"eval_job={job['id']} failed error={exc!r}", flush=True)
        return "failed"


def worker_loop(args: argparse.Namespace, *, worker_id: str) -> None:
    conn = connect(database_url(args.direct))
    completed = 0
    try:
        while args.max_jobs <= 0 or completed < args.max_jobs:
            job = claim_eval_job(
                conn,
                runtime_image_ref=args.runtime_image_ref,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
            )
            if job is None:
                if args.once:
                    return
                time.sleep(args.poll_seconds)
                continue
            status = run_eval_job(
                conn,
                job=job,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
                artifact_root=Path(args.artifact_root),
                output_dir=Path(args.output_dir),
            )
            completed += 1
            print(f"worker={worker_id} eval_job={job['id']} status={status}", flush=True)
            if job.get("drain_requested") or status.endswith("_drained"):
                return
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain Codex-authored PPO eval jobs.")
    parser.add_argument(
        "--runtime-image-ref",
        required=True,
        help="Exact immutable runtime image ref that claimed eval_jobs must require.",
    )
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit when no matching job is available.",
    )
    parser.add_argument("--artifact-root", default="runs/eval_artifacts")
    parser.add_argument("--output-dir", default="logs/eval_runner")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument(
        "--status-goal",
        help="Print compact queue status for this goal before starting workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.status_goal:
        conn = connect(database_url(args.direct))
        try:
            print_status(queue_status(conn, goal_slug=args.status_goal))
        finally:
            conn.close()
    worker_loop(args, worker_id=args.worker_id or new_worker_id("eval-runner"))


if __name__ == "__main__":
    main()
