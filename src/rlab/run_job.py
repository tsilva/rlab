from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.eval_job_runner import (
    eval_env_config,
    log_eval_to_wandb,
    model_ref_for_config,
    normalize_eval_config,
    resolve_model_path,
    write_eval_output,
)
from rlab.device import resolve_sb3_device
from rlab.eval_runner import evaluate_model_episodes
from rlab.json_utils import json_safe
from rlab.train_runner import (
    collect_result_metadata,
    normalize_train_config,
    train_command_for_job,
    write_train_config_file,
)

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - import errors surface only for eval payloads.
    PPO = None


RESULT_SCHEMA_VERSION = 1
DEFAULT_POST_TRAIN_EVAL_EPISODES = 100
DEFAULT_POST_TRAIN_EVAL_N_ENVS = 20


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job payload must be a JSON object")
    if payload.get("job_kind") not in {"train", "eval"}:
        raise ValueError("job payload must define job_kind=train or job_kind=eval")
    if not isinstance(payload.get("job"), dict):
        raise ValueError("job payload must define a job object")
    return payload


def write_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    path.write_text(json.dumps(json_safe(dict(result)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def base_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(payload["job"])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_kind": payload.get("job_kind"),
        "job_id": int(job["id"]),
        "launch_id": payload.get("launch_id"),
        "machine": payload.get("machine"),
        "runtime_image_ref": payload.get("runtime_image_ref") or job.get("runtime_image_ref"),
    }


def run_train_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    job = dict(payload["job"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_train_config_file(job, output_dir / "train_config.json")
    command = train_command_for_job(config_path)
    log_path = log_dir / f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    metadata = collect_result_metadata(job, log_path)
    status = "succeeded" if process.returncode == 0 else "failed"
    eval_section = (
        run_post_train_eval(job, metadata, output_dir)
        if status == "succeeded"
        else {"status": "skipped", "reason": "training did not succeed"}
    )
    result = {
        **base_result(payload),
        "status": status,
        "exit_code": process.returncode,
        "train": {
            "status": status,
            "result": metadata,
            "log_path": str(log_path),
        },
        "eval": eval_section,
    }
    if process.returncode != 0:
        result["error"] = f"train process exited {process.returncode}"
    return result


def _append_flag(command: list[str], name: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([name, str(value)])


def _comma_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def post_train_eval_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("post_train_eval", True))


def run_post_train_eval(
    job: Mapping[str, Any],
    train_metadata: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    config = normalize_train_config(dict(job), resolve_resume_artifact=False)
    if not post_train_eval_enabled(config):
        return {"status": "skipped", "reason": "post_train_eval disabled"}
    model_path = str(train_metadata.get("final_model_path") or "").strip()
    if not model_path or not Path(model_path).is_file():
        return {"status": "skipped", "reason": "final model not found"}
    eval_dir = output_dir / "post_train_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_output = eval_dir / "metrics.json"
    eval_log = eval_dir / "eval.log"
    command = [
        sys.executable,
        "-m",
        "rlab.eval",
        "--model",
        model_path,
        "--episodes",
        str(int(config.get("post_train_eval_episodes") or DEFAULT_POST_TRAIN_EVAL_EPISODES)),
        "--n-envs",
        str(int(config.get("post_train_eval_n_envs") or DEFAULT_POST_TRAIN_EVAL_N_ENVS)),
        "--output",
        str(eval_output),
        "--summary-only",
        "--no-progress",
        "--no-promote-best",
    ]
    _append_flag(command, "--game", config.get("game"))
    _append_flag(command, "--state", config.get("state"))
    _append_flag(command, "--states", _comma_value(config.get("states")))
    _append_flag(command, "--state-probs", _comma_value(config.get("state_probs")))
    _append_flag(command, "--max-steps", config.get("max_steps"))
    _append_flag(command, "--frame-skip", config.get("frame_skip"))
    _append_flag(command, "--hud-crop-top", config.get("hud_crop_top"))
    _append_flag(command, "--action-set", config.get("action_set"))
    _append_flag(command, "--device", config.get("device"))
    _append_flag(command, "--wandb-mode", config.get("wandb_mode"))
    if train_metadata.get("wandb_run_id"):
        _append_flag(command, "--wandb-run-id", train_metadata.get("wandb_run_id"))
    else:
        command.append("--no-wandb-log")
    with eval_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    if process.returncode != 0:
        return {
            "status": "failed",
            "exit_code": process.returncode,
            "error": f"post-train eval exited {process.returncode}",
            "log_path": str(eval_log),
        }
    metrics = json.loads(eval_output.read_text(encoding="utf-8")) if eval_output.is_file() else {}
    return {
        "status": "succeeded",
        "exit_code": process.returncode,
        "output_path": str(eval_output),
        "log_path": str(eval_log),
        "metrics_json": metrics,
    }


def run_eval_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if PPO is None:
        raise RuntimeError("stable_baselines3 is unavailable; cannot run eval payload")
    job = dict(payload["job"])
    config = normalize_eval_config(job)
    artifact_root = Path(str(payload.get("artifact_root") or output_dir / "artifacts"))
    model_path = resolve_model_path(config, artifact_root=artifact_root)
    env_config = eval_env_config(config, model_path)
    model = PPO.load(model_path, device=resolve_sb3_device(str(config["device"])))
    video_path = output_dir / f"eval_job_{job['id']}_best_episode.mp4" if config.get("capture_best_video") else None
    metrics, written_video = evaluate_model_episodes(
        model=model,
        config=env_config,
        episodes=int(config["episodes"]),
        seed=int(config["seed"]),
        max_steps=int(config["max_steps"]),
        deterministic=not bool(config["stochastic"]),
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
            "wandb_logged": False,
            "wandb_log_error": repr(exc),
            "wandb_log_step": job.get("checkpoint_step"),
        }
    return {
        **base_result(payload),
        "status": "succeeded",
        "exit_code": 0,
        "eval": {
            "status": "succeeded",
            "result": {
                "candidate_label": job.get("candidate_label"),
                "model_ref": model_ref_for_config(config),
                "output_path": str(output_path),
                "video_path": str(written_video) if written_video else None,
                "artifact_ref": job.get("artifact_ref") or model_ref_for_config(config),
                "checkpoint_step": job.get("checkpoint_step"),
                "eval_protocol_hash": job.get("eval_protocol_hash"),
                "metrics_json": safe_metrics,
                **wandb_result,
            },
        },
    }


def run_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if payload.get("job_kind") == "train":
        return run_train_payload(payload, output_dir)
    if payload.get("job_kind") == "eval":
        return run_eval_payload(payload, output_dir)
    raise ValueError(f"unsupported job_kind: {payload.get('job_kind')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one claimed rlab job payload and write result.json.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_payload(args.payload)
    try:
        result = run_payload(payload, args.output_dir)
    except Exception as exc:
        result = {
            **base_result(payload),
            "status": "failed",
            "exit_code": 1,
            "error": repr(exc),
        }
        write_result(args.output_dir, result)
        raise
    write_result(args.output_dir, result)
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
