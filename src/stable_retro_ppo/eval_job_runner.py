from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import (
    env_config_from_config_dict,
    env_config_from_model_metadata,
)
from stable_retro_ppo.campaign import (
    campaign_status,
    claim_eval_job,
    connect,
    database_url,
    finish_eval_job,
    heartbeat_eval_job,
    new_worker_id,
    print_status,
)
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import EnvConfig, resolve_env_config
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.json_utils import json_safe
from stable_retro_ppo.wandb_artifacts import artifact_download_dir, download_model_artifact


def normalize_eval_config(job: dict[str, Any]) -> dict[str, Any]:
    config = dict(job.get("eval_config") or {})
    config.setdefault("episodes", 100)
    config.setdefault("seed", 0)
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
    resolved = resolve_env_config(replace(base, **overrides))
    return replace(
        resolved,
        terminate_on_life_loss=False,
        terminate_on_level_change=False,
        terminate_on_completion=False,
    )


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
        "profile_id": job["profile_id"],
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
                "campaign_eval_job_id": int(job["id"]),
                "campaign_profile_id": job["profile_id"],
                "campaign_candidate_label": job.get("candidate_label"),
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
                "metrics_json": safe_metrics,
            },
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
                profile_id=args.profile,
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
    parser.add_argument("--profile", required=True, help="Exact eval_jobs.profile_id to claim.")
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
        help="Print compact campaign status for this goal before starting workers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.status_goal:
        conn = connect(database_url(args.direct))
        try:
            print_status(campaign_status(conn, goal_slug_or_id=args.status_goal))
        finally:
            conn.close()
    worker_loop(args, worker_id=args.worker_id or new_worker_id("eval-runner"))


if __name__ == "__main__":
    main()
