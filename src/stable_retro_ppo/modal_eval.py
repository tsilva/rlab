from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psycopg2
from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import (
    require_env_config_from_model_metadata,
    require_training_metadata,
    stable_json_hash,
)
from stable_retro_ppo.campaign import (
    claim_eval_job,
    connect as campaign_connect,
    database_url as campaign_database_url,
    finish_eval_job,
)
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import assert_rom_imported
from stable_retro_ppo.eval_job_runner import (
    eval_env_config,
    model_ref_for_config,
    normalize_eval_config,
)
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.json_utils import json_safe
from stable_retro_ppo.modal_core import (
    RUNS_DIR,
    VOLUME_ROOT,
    app,
    ensure_remote_roms,
    eval_queue_secret,
    image,
    volume,
)
from stable_retro_ppo.wandb_artifacts import artifact_download_dir, download_model_artifact


LOCAL_ARTIFACT_CACHE_DIR = Path("/tmp/stable-retro-ppo-eval-artifacts")


@dataclass(frozen=True)
class PrefetchedArtifact:
    path: Path
    download_elapsed_seconds: float


@dataclass(frozen=True)
class PrefetchedJob:
    job: dict[str, Any]
    future: Future[PrefetchedArtifact]


def connect():
    value = os.environ.get("DATABASE_URL") or os.environ.get("DIRECT_DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL or DIRECT_DATABASE_URL must be available in Modal")
    return campaign_connect(campaign_database_url(use_direct=False))


def close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def claim_job(
    conn,
    *,
    profile_id: str,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    return claim_eval_job(
        conn,
        profile_id=profile_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )


def mark_job_failed(conn, *, job: dict[str, Any], worker_id: str, error: str) -> bool:
    finish_eval_job(
        conn,
        job=job,
        worker_id=worker_id,
        status="failed",
        result={
            "candidate_label": job.get("candidate_label"),
            "model_ref": model_ref_for_config(normalize_eval_config(job)),
        },
        error=error[:4000],
    )
    return True


def commit_result(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    metrics: dict[str, Any],
) -> bool:
    config = normalize_eval_config(job)
    finish_eval_job(
        conn,
        job=job,
        worker_id=worker_id,
        status="succeeded",
        result={
            "candidate_label": job.get("candidate_label"),
            "model_ref": model_ref_for_config(config),
            "metrics_json": json_safe(metrics),
        },
    )
    return True


def commit_result_with_reconnect(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    metrics: dict[str, Any],
) -> tuple[bool, Any]:
    try:
        return commit_result(conn, job=job, worker_id=worker_id, metrics=metrics), conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        close_quietly(conn)
        conn = connect()
        return commit_result(conn, job=job, worker_id=worker_id, metrics=metrics), conn


def mark_job_failed_with_reconnect(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    error: str,
) -> tuple[bool, Any]:
    try:
        return mark_job_failed(conn, job=job, worker_id=worker_id, error=error), conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        close_quietly(conn)
        conn = connect()
        return mark_job_failed(conn, job=job, worker_id=worker_id, error=error), conn


def cached_model_artifact(ref: str, *, base_dir: Path = RUNS_DIR / "wandb_artifacts") -> Path:
    return download_model_artifact(ref, artifact_download_dir(base_dir, ref))


def prefetch_model_artifact(job: dict[str, Any]) -> PrefetchedArtifact:
    started_at = time.monotonic()
    config = normalize_eval_config(job)
    artifact_ref = config.get("artifact_ref") or config.get("model_artifact")
    if artifact_ref:
        model_path = cached_model_artifact(
            str(artifact_ref),
            base_dir=LOCAL_ARTIFACT_CACHE_DIR,
        )
    elif config.get("model_path"):
        model_path = Path(str(config["model_path"])).expanduser()
    else:
        raise ValueError("eval_config must define artifact_ref, model_artifact, or model_path")
    return PrefetchedArtifact(
        path=model_path,
        download_elapsed_seconds=round(time.monotonic() - started_at, 3),
    )


def evaluate_job(
    job: dict[str, Any],
    *,
    device: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    eval_config = normalize_eval_config(job)
    if model_path is None:
        model_path = prefetch_model_artifact(job).path
    config = eval_env_config(eval_config, model_path)
    try:
        training_hash = stable_json_hash(require_training_metadata(model_path))
    except ValueError:
        training_hash = ""
    assert_rom_imported(config.game)
    model = PPO.load(model_path, device=resolve_sb3_device(device))
    metrics, _ = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=int(eval_config["episodes"]),
        seed=int(eval_config["seed"]),
        max_steps=int(eval_config["max_steps"]),
        deterministic=not bool(eval_config["stochastic"]),
        completion_x_threshold=config.completion_x_threshold,
        n_envs=int(eval_config["n_envs"]),
        capture_best_video=False,
        extra={
            "campaign_eval_job_id": int(job["id"]),
            "campaign_profile_id": job["profile_id"],
            "campaign_candidate_label": job.get("candidate_label"),
            "model_ref": model_ref_for_config(eval_config),
            "training_metadata_hash": training_hash,
            "eval_seed": int(eval_config["seed"]),
        },
    )
    return metrics


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=6 * 60 * 60,
    cpu=4.0,
    memory=8192,
    secrets=[eval_queue_secret],
)
def eval_artifact_benchmark_remote(
    artifact_ref: str,
    episodes: int = 100,
    seed: int = 10007,
    n_envs: int = 0,
    env_threads: int = 0,
    device: str = "cpu",
    benchmark_cpu: float = 4.0,
    benchmark_memory_mib: int = 8192,
) -> dict[str, Any]:
    total_started_at = time.monotonic()
    ensure_remote_roms("evaluation benchmark")
    effective_n_envs = n_envs if n_envs > 0 else 1
    model_path = cached_model_artifact(artifact_ref)
    training = require_training_metadata(model_path)
    training_hash = stable_json_hash(training)
    config = require_env_config_from_model_metadata(model_path)
    if env_threads > 0:
        config = replace(config, env_threads=env_threads)
    assert_rom_imported(config.game)
    model = PPO.load(model_path, device=resolve_sb3_device(device))
    eval_started_at = time.monotonic()
    try:
        metrics, _ = evaluate_model_episodes(
            model=model,
            config=config,
            episodes=episodes,
            seed=seed,
            max_steps=config.max_episode_steps,
            deterministic=False,
            completion_x_threshold=config.completion_x_threshold,
            n_envs=effective_n_envs,
            capture_best_video=False,
            extra={
                "checkpoint_artifact": artifact_ref,
                "training_metadata_hash": training_hash,
                "eval_seed": seed,
                "benchmark": True,
                "benchmark_cpu": benchmark_cpu,
                "benchmark_memory_mib": benchmark_memory_mib,
                "benchmark_env_threads": env_threads,
            },
        )
    finally:
        eval_elapsed_seconds = time.monotonic() - eval_started_at
        volume.commit()

    total_elapsed_seconds = time.monotonic() - total_started_at
    return {
        "artifact_ref": artifact_ref,
        "training_metadata_hash": training_hash,
        "episodes": episodes,
        "n_envs": effective_n_envs,
        "env_threads": env_threads,
        "cpu": benchmark_cpu,
        "memory_mib": benchmark_memory_mib,
        "eval_elapsed_seconds": round(eval_elapsed_seconds, 3),
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "eval_episodes_per_second": round(episodes / eval_elapsed_seconds, 4),
        "total_episodes_per_second": round(episodes / total_elapsed_seconds, 4),
        "completion_rate": float(metrics["completion_rate"]),
        "death_rate": float(metrics["death_rate"]),
        "max_x_max": int(metrics["max_x_max"]),
        "reward_mean": float(metrics["reward_mean"]),
    }


@app.local_entrypoint()
def eval_artifact_benchmark(
    artifact_ref: str,
    episodes: int = 100,
    seed: int = 10007,
    n_envs: int = 0,
    env_threads: int = 0,
    cpu: float = 4.0,
    memory_mib: int = 8192,
    device: str = "cpu",
) -> None:
    remote = eval_artifact_benchmark_remote.with_options(cpu=cpu, memory=memory_mib)
    result = remote.remote(
        artifact_ref=artifact_ref,
        episodes=episodes,
        seed=seed,
        n_envs=n_envs,
        env_threads=env_threads,
        device=device,
        benchmark_cpu=cpu,
        benchmark_memory_mib=memory_mib,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=6 * 60 * 60,
    cpu=4.0,
    memory=8192,
    secrets=[eval_queue_secret],
)
def eval_worker_remote(
    profile: str = "mario-level1-quick",
    worker_name: str = "",
    max_jobs: int = 0,
    idle_polls: int = 2,
    idle_sleep_seconds: float = 5.0,
    lease_seconds: int = 1800,
    device: str = "cpu",
    prefetch_jobs: int = 2,
) -> dict[str, Any]:
    ensure_remote_roms("evaluation")
    worker_id = worker_name or f"modal-eval-{uuid.uuid4()}"
    started_at = time.monotonic()
    completed = 0
    failed = 0
    idle = 0
    processed_jobs: list[int] = []
    prefetch_target = max(prefetch_jobs, 1)
    pending_jobs: list[PrefetchedJob] = []

    conn = connect()
    prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="artifact-prefetch")
    try:
        while max_jobs <= 0 or completed + failed < max_jobs:
            while len(pending_jobs) < prefetch_target and (
                max_jobs <= 0 or completed + failed + len(pending_jobs) < max_jobs
            ):
                job = claim_job(
                    conn,
                    profile_id=profile,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
                if job is None:
                    break

                idle = 0
                future = prefetcher.submit(prefetch_model_artifact, job)
                pending_jobs.append(PrefetchedJob(job=job, future=future))
                print(
                    "claimed "
                    f"worker={worker_id} "
                    f"job_id={job['id']} "
                    f"profile={job['profile_id']} "
                    f"candidate={job.get('candidate_label') or ''} "
                    f"prefetch_queued={len(pending_jobs)}",
                    flush=True,
                )

            if not pending_jobs:
                idle += 1
                if idle >= idle_polls:
                    break
                time.sleep(idle_sleep_seconds)
                continue

            prefetched_job = pending_jobs.pop(0)
            job = prefetched_job.job
            try:
                prefetch_wait_started_at = time.monotonic()
                prefetched_artifact = prefetched_job.future.result()
                prefetch_wait_seconds = round(time.monotonic() - prefetch_wait_started_at, 3)
                print(
                    "prefetched "
                    f"worker={worker_id} "
                    f"job_id={job['id']} "
                    f"download_seconds={prefetched_artifact.download_elapsed_seconds:.3f} "
                    f"wait_seconds={prefetch_wait_seconds:.3f}",
                    flush=True,
                )
                metrics = evaluate_job(
                    job,
                    device=device,
                    model_path=prefetched_artifact.path,
                )
                metrics["artifact_download_seconds"] = prefetched_artifact.download_elapsed_seconds
                metrics["artifact_prefetch_wait_seconds"] = prefetch_wait_seconds
                committed, conn = commit_result_with_reconnect(
                    conn, job=job, worker_id=worker_id, metrics=metrics
                )
                if committed:
                    completed += 1
                    processed_jobs.append(int(job["id"]))
                    print(
                        "done "
                        f"worker={worker_id} "
                        f"job_id={job['id']} "
                        f"completion_rate={metrics['completion_rate']:.3f} "
                        f"max_x_max={metrics['max_x_max']} "
                        f"reward_mean={metrics['reward_mean']:.2f}",
                        flush=True,
                    )
                else:
                    failed += 1
                    print(f"lost lease worker={worker_id} job_id={job['id']}", flush=True)
            except Exception as exc:
                failed += 1
                try:
                    marked_failed, conn = mark_job_failed_with_reconnect(
                        conn, job=job, worker_id=worker_id, error=repr(exc)
                    )
                    suffix = "" if marked_failed else " mark_failed=false"
                except Exception as mark_exc:
                    suffix = f" mark_failed_error={mark_exc!r}"
                print(
                    f"failed worker={worker_id} job_id={job['id']} error={exc!r}{suffix}",
                    flush=True,
                )
    finally:
        prefetcher.shutdown(wait=False, cancel_futures=True)
        close_quietly(conn)
        volume.commit()

    return {
        "worker_id": worker_id,
        "completed": completed,
        "failed": failed,
        "processed_jobs": processed_jobs,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }


@app.local_entrypoint()
def eval_queue(
    profile: str = "mario-level1-quick",
    runners: int = 2,
    max_jobs_per_runner: int = 0,
    idle_polls: int = 2,
    idle_sleep_seconds: float = 5.0,
    lease_seconds: int = 1800,
    cpu: float = 4.0,
    memory_mib: int = 8192,
    device: str = "cpu",
    prefetch_jobs: int = 2,
) -> None:
    calls = []
    remote = eval_worker_remote.with_options(cpu=cpu, memory=memory_mib)
    for index in range(runners):
        worker_name = f"modal-eval-{uuid.uuid4()}-{index + 1}"
        call = remote.spawn(
            profile=profile,
            worker_name=worker_name,
            max_jobs=max_jobs_per_runner,
            idle_polls=idle_polls,
            idle_sleep_seconds=idle_sleep_seconds,
            lease_seconds=lease_seconds,
            device=device,
            prefetch_jobs=prefetch_jobs,
        )
        calls.append((worker_name, call))
        print(f"started {worker_name}", flush=True)

    results = []
    for worker_name, call in calls:
        result = call.get()
        print(f"result {worker_name}: {json.dumps(result, sort_keys=True)}", flush=True)
        results.append(result)

    print(json.dumps({"runners": runners, "results": results}, sort_keys=True), flush=True)
