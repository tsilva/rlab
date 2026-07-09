from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from rlab.artifact_worker import resume_wandb_run
from rlab.cli import parse_train_args
from rlab.device import resolve_sb3_device
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.eval_runner import evaluate_model_episodes
from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_CHECKPOINT_ARTIFACT,
    EVAL_CHECKPOINT_STEP,
    EVAL_CONFIG_HUD_CROP_TOP,
    EVAL_DURATION_SECONDS,
    EVAL_REWARD_MAX,
    EVAL_REWARD_MEAN,
    EVAL_REWARD_STD,
    GLOBAL_STEP,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.train import (
    eval_checkpoint_artifact_ref,
    eval_config_from_training_config,
    eval_score,
    log_checkpoint_eval_metrics,
)


def metric_payload(
    *,
    args,
    metrics: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_step: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        GLOBAL_STEP: checkpoint_step,
        EVAL_CHECKPOINT_STEP: checkpoint_step,
        EVAL_CHECKPOINT_ARTIFACT: eval_checkpoint_artifact_ref(args, checkpoint_path, checkpoint_step),
        EVAL_REWARD_MEAN: metrics["reward_mean"],
        EVAL_REWARD_STD: metrics["reward_std"],
        EVAL_REWARD_MAX: metrics["reward_max"],
        EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
        EVAL_CONFIG_HUD_CROP_TOP: args.hud_crop_top,
        "eval/episodes": metrics["episodes"],
    }
    if EVAL_DURATION_SECONDS in metrics:
        payload[EVAL_DURATION_SECONDS] = metrics[EVAL_DURATION_SECONDS]
    for key, value in metrics.items():
        if key.startswith(("eval/done/", "eval/info/")):
            payload[key] = value
    return payload


def update_summary_file(run_dir: Path, summary: dict[str, object]) -> None:
    path = run_dir / "checkpoint_eval_summary.json"
    existing: list[dict[str, object]] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [dict(item) for item in loaded if isinstance(item, dict)]
        except json.JSONDecodeError:
            existing = []
    step = summary.get("checkpoint_step")
    existing = [item for item in existing if item.get("checkpoint_step") != step]
    existing.append(summary)
    existing.sort(key=lambda item: int(item.get("checkpoint_step") or 0))
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def process_eval(
    *,
    store: MetricStore,
    args,
    config,
    run_dir: Path,
    row: dict[str, Any],
) -> None:
    checkpoint_id = int(row["id"])
    if not store.claim_eval(checkpoint_id):
        return
    checkpoint_path = Path(str(row["path"]))
    step = int(row["step"])
    try:
        eval_model = PPO.load(checkpoint_path, device=resolve_sb3_device(args.device))
        episodes = int(args.post_train_eval_episodes)
        max_steps = int(args.post_train_eval_max_steps or args.max_episode_steps)
        n_envs = int(args.post_train_eval_n_envs)
        metrics, _video_path = evaluate_model_episodes(
            model=eval_model,
            config=eval_config_from_training_config(config),
            episodes=episodes,
            seed=DEFAULT_EVAL_SEED,
            max_steps=max_steps,
            deterministic=not bool(args.post_train_eval_stochastic),
            n_envs=n_envs,
            progress=True,
            progress_description=f"eval checkpoint {step}",
            extra={
                "checkpoint_step": step,
                "checkpoint_artifact": str(checkpoint_path),
                "eval_source": "async_worker",
            },
        )
        payload = metric_payload(
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
        )
        store.append_metrics(payload, step=step, source="eval", checkpoint_step=step)
        store.mark_eval_succeeded(checkpoint_id, episodes=episodes, metrics=payload)
        artifact_ref = eval_checkpoint_artifact_ref(args, checkpoint_path, step)
        wandb_run = None
        try:
            wandb_run = resume_wandb_run(args, run_dir)
            log_checkpoint_eval_metrics(
                wandb_run,
                args=args,
                metrics=metrics,
                checkpoint_path=checkpoint_path,
                checkpoint_step_value=step,
                artifact_ref=artifact_ref,
                eval_source="async_worker",
            )
        except Exception as exc:
            print(f"checkpoint eval W&B logging failed step={step}: {exc}", flush=True)
        finally:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
        score = eval_score(metrics)
        summary = {
            "checkpoint_step": step,
            "checkpoint_path": str(checkpoint_path),
            "objective": score[0],
            "reward_mean": float(metrics["reward_mean"]),
            "eval_source": "async_worker",
        }
        update_summary_file(run_dir, summary)
        print(f"checkpoint eval ready: checkpoint_id={checkpoint_id} step={step}", flush=True)
    except Exception as exc:
        store.mark_eval_failed(checkpoint_id, repr(exc))
        print(f"checkpoint eval failed checkpoint_id={checkpoint_id}: {exc}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rlab checkpoints asynchronously.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_args = parse_train_args(["--train-config-json", str(args.train_config_json)])
    config = resolve_env_config(env_config_from_args(train_args, include_states=True))
    store = MetricStore(metric_store_path(args.run_dir))
    store.init()
    while True:
        rows = store.pending_evals(limit=max(args.limit, 1))
        if not rows and args.stop_file.exists():
            return 0
        for row in rows:
            process_eval(
                store=store,
                args=train_args,
                config=config,
                run_dir=args.run_dir,
                row=row,
            )
        if not rows:
            time.sleep(max(args.poll_seconds, 0.25))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
