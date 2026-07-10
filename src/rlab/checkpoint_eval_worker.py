from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from rlab.artifact_worker import resume_wandb_run
from rlab.artifacts import sanitize_artifact_name
from rlab.checkpoint_eval_config import (
    CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP,
    CHECKPOINT_EVAL_CANDIDATE_EPISODES,
    CHECKPOINT_EVAL_CANDIDATE_PASS,
    CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX,
    normalize_checkpoint_eval_stages,
    staged_metric_name,
)
from rlab.cli import parse_train_args
from rlab.device import resolve_sb3_device
from rlab.early_stop import evaluate_early_stop_config
from rlab.env import resolve_env_config, task_max_episode_steps, task_termination, with_task_termination
from rlab.env_config import env_config_from_args
from rlab.eval_runner import evaluate_model_episodes
from rlab.eval_metrics import flat_numeric_metrics
from rlab.eval_metrics import completion_score as eval_completion_score
from rlab.eval_metrics import eval_selection_objective_name, eval_selection_score
from rlab.metric_names import (
    EVAL_BEST_X,
    EVAL_BEST_REWARD,
    EVAL_CHECKPOINT_ARTIFACT,
    EVAL_CHECKPOINT_STEP,
    EVAL_CONFIG_HUD_CROP_TOP,
    EVAL_DURATION_SECONDS,
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
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_COMPLETION_RATE,
    LEADER_CHECKPOINT_COMPLETION_RATE_MEAN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_LOCAL_PATH,
    LEADER_CHECKPOINT_MAX_X_MAX,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_OBJECTIVE_NAME,
    LEADER_CHECKPOINT_REWARD_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL,
    LEADER_CHECKPOINT_UPDATED_AT,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.wandb_utils import DEFAULT_WANDB_ENTITY, resolve_wandb_project


def eval_checkpoint_artifact_ref(args, checkpoint_path: Path, step: int) -> str:
    if getattr(args, "no_wandb_artifacts", False):
        return str(checkpoint_path)
    entity = str(getattr(args, "wandb_entity", "") or DEFAULT_WANDB_ENTITY).strip()
    project = resolve_wandb_project(
        getattr(args, "wandb_project", None),
        str(getattr(args, "game", "") or ""),
    )
    if entity and project:
        name = f"{sanitize_artifact_name(args.run_name)}-checkpoint"
        return f"{entity}/{project}/{name}:step-{step}"
    return str(checkpoint_path)


def eval_score(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    return eval_selection_score(dict(metrics))


def update_best_checkpoint_summary(
    wandb_run,
    *,
    metrics: dict[str, object],
    checkpoint_path: Path,
    checkpoint_step_value: int,
    artifact_ref: str,
    eval_source: str = "async_worker",
) -> None:
    if wandb_run is None:
        return

    def remove_summary_key(key: str) -> None:
        try:
            del wandb_run.summary[key]
        except (AttributeError, KeyError):
            pass

    def summary_float(key: str) -> float:
        try:
            value = wandb_run.summary.get(key)
        except AttributeError:
            value = None
        if value is None:
            return float("-inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    score = eval_score(metrics)
    objective_name = eval_selection_objective_name(dict(metrics))
    try:
        previous_objective_name = str(
            wandb_run.summary.get(LEADER_CHECKPOINT_OBJECTIVE_NAME) or ""
        )
    except AttributeError:
        previous_objective_name = ""
    previous_objective = summary_float(LEADER_CHECKPOINT_OBJECTIVE)
    previous_completion = summary_float(LEADER_CHECKPOINT_COMPLETION_RATE)
    previous_completion_mean = summary_float(LEADER_CHECKPOINT_COMPLETION_RATE_MEAN)
    previous_steps_to_goal = summary_float(LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL)
    previous_step_score = (
        -previous_steps_to_goal
        if previous_steps_to_goal > float("-inf")
        else float("-inf")
    )
    if previous_objective_name == objective_name and previous_objective > float("-inf"):
        previous = (
            previous_objective,
            previous_completion_mean
            if objective_name != "eval/reward/mean"
            else summary_float(LEADER_CHECKPOINT_REWARD_MEAN),
            previous_step_score,
            summary_float(LEADER_CHECKPOINT_REWARD_MEAN),
        )
    else:
        previous = (
            previous_completion,
            previous_completion_mean,
            previous_step_score,
            summary_float(LEADER_CHECKPOINT_REWARD_MEAN),
        )
    if score < previous:
        return

    wandb_run.summary[LEADER_CHECKPOINT_OBJECTIVE] = score[0]
    wandb_run.summary[LEADER_CHECKPOINT_OBJECTIVE_NAME] = objective_name
    completion = eval_completion_score(dict(metrics))
    if completion is not None:
        wandb_run.summary[LEADER_CHECKPOINT_COMPLETION_RATE] = completion[0]
        wandb_run.summary[LEADER_CHECKPOINT_COMPLETION_RATE_MEAN] = completion[1]
    else:
        remove_summary_key(LEADER_CHECKPOINT_COMPLETION_RATE)
        remove_summary_key(LEADER_CHECKPOINT_COMPLETION_RATE_MEAN)
    wandb_run.summary[LEADER_CHECKPOINT_REWARD_MEAN] = score[3]
    wandb_run.summary[LEADER_CHECKPOINT_MAX_X_MAX] = metrics.get("max_x_max")
    wandb_run.summary[LEADER_CHECKPOINT_STEP] = checkpoint_step_value
    if score[2] > float("-inf"):
        wandb_run.summary[LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL] = checkpoint_step_value
    else:
        remove_summary_key(LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL)
    wandb_run.summary[LEADER_CHECKPOINT_ARTIFACT_REF] = artifact_ref
    wandb_run.summary[LEADER_CHECKPOINT_LOCAL_PATH] = str(checkpoint_path)
    wandb_run.summary[LEADER_CHECKPOINT_EVAL_SOURCE] = eval_source
    wandb_run.summary[LEADER_CHECKPOINT_UPDATED_AT] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )


def log_checkpoint_eval_metrics(
    wandb_run,
    *,
    args,
    metrics: dict[str, object],
    checkpoint_path: Path,
    checkpoint_step_value: int,
    artifact_ref: str,
    eval_source: str = "async_worker",
) -> None:
    if wandb_run is None:
        return
    payload: dict[str, object] = {
        GLOBAL_STEP: checkpoint_step_value,
        EVAL_CHECKPOINT_STEP: checkpoint_step_value,
        EVAL_CHECKPOINT_ARTIFACT: artifact_ref,
        EVAL_REWARD_MEAN: metrics["reward_mean"],
        EVAL_REWARD_STD: metrics["reward_std"],
        EVAL_REWARD_MAX: metrics["reward_max"],
        EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
        EVAL_CONFIG_HUD_CROP_TOP: args.hud_crop_top,
        "eval/source": eval_source,
    }
    optional_metric_pairs = {
        EVAL_DURATION_SECONDS: EVAL_DURATION_SECONDS,
        EVAL_PROGRESS_X_MEAN: "max_x_mean",
        EVAL_PROGRESS_X_MAX: "max_x_max",
        EVAL_PROGRESS_LEVEL_X_MEAN: "max_level_x_mean",
        EVAL_PROGRESS_LEVEL_X_MAX: "max_level_x_max",
        EVAL_DEATH_COUNT: "death_count",
        EVAL_DEATH_RATE: "death_rate",
    }
    for metric_name, source_key in optional_metric_pairs.items():
        if source_key in metrics:
            payload[metric_name] = metrics[source_key]
    if "max_x_pos" in metrics["best_episode"]:
        payload[EVAL_BEST_X] = metrics["best_episode"]["max_x_pos"]
    payload.update(flat_numeric_metrics(metrics, "eval/done/"))
    payload.update(flat_numeric_metrics(metrics, "eval/info/"))
    wandb_run.log(payload)
    update_best_checkpoint_summary(
        wandb_run,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_step_value=checkpoint_step_value,
        artifact_ref=artifact_ref,
        eval_source=eval_source,
    )


def eval_config_from_training_config(config):
    termination = task_termination(config)
    return with_task_termination(
        config,
        failure=[],
        success=[name for name in termination.get("success", ()) if name == "level_change"],
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


def staged_metric_payload(
    *,
    stage_name: str,
    stage_index: int,
    args,
    metrics: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_step: int,
    passed: bool,
) -> dict[str, object]:
    canonical_payload = metric_payload(
        args=args,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    payload: dict[str, object] = {GLOBAL_STEP: checkpoint_step}
    for key, value in canonical_payload.items():
        if key == GLOBAL_STEP:
            continue
        if key.startswith("eval/"):
            payload[staged_metric_name(stage_name, key)] = value
    payload[f"checkpoint_eval/{stage_name}/pass"] = 1.0 if passed else 0.0
    payload[f"checkpoint_eval/{stage_name}/stage_index"] = float(stage_index)
    payload[f"checkpoint_eval/{stage_name}/source"] = "async_worker"
    return payload


def log_staged_checkpoint_eval_metrics(wandb_run, payload: dict[str, object]) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload)


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
    stage = summary.get("stage_name")
    existing = [
        item
        for item in existing
        if not (item.get("checkpoint_step") == step and item.get("stage_name") == stage)
    ]
    existing.append(summary)
    existing.sort(
        key=lambda item: (
            int(item.get("checkpoint_step") or 0),
            int(item.get("stage_index") or -1),
        )
    )
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_eval_stages(args) -> list[dict[str, Any]]:
    return normalize_checkpoint_eval_stages(
        getattr(args, "checkpoint_eval_stages", None),
        label="checkpoint_eval_stages",
    )


def _stage_for_row(stages: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    stage_name = str(row["stage_name"])
    stage_index = int(row["stage_index"])
    for index, stage in enumerate(stages):
        if str(stage["name"]) == stage_name and index == stage_index:
            return stage
    raise ValueError(f"checkpoint eval stage not configured: {stage_name}")


def _metric_value(metrics: dict[str, object], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def process_staged_eval(
    *,
    store: MetricStore,
    args,
    config,
    run_dir: Path,
    row: dict[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    stage_id = int(row["eval_stage_id"])
    checkpoint_id = int(row["id"])
    if not store.claim_checkpoint_eval_stage(stage_id):
        return
    checkpoint_path = Path(str(row["path"]))
    step = int(row["step"])
    stage = _stage_for_row(stages, row)
    stage_index = int(row["stage_index"])
    stage_name = str(stage["name"])
    try:
        eval_model = PPO.load(checkpoint_path, device=resolve_sb3_device(args.device))
        episodes = int(stage["episodes"])
        max_steps = int(args.post_train_eval_max_steps or task_max_episode_steps(config))
        n_envs = int(stage.get("n_envs") or getattr(args, "checkpoint_eval_n_envs", 20))
        metrics, _video_path = evaluate_model_episodes(
            model=eval_model,
            config=eval_config_from_training_config(config),
            episodes=episodes,
            seed=DEFAULT_EVAL_SEED,
            max_steps=max_steps,
            deterministic=not bool(args.post_train_eval_stochastic),
            n_envs=n_envs,
            progress=True,
            progress_description=f"eval checkpoint {step} {stage_name}",
            extra={
                "checkpoint_step": step,
                "checkpoint_artifact": str(checkpoint_path),
                "eval_source": f"async_worker:{stage_name}",
            },
        )
        canonical_payload = metric_payload(
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
        )
        passed, observed = evaluate_early_stop_config(
            stage["pass"],
            lambda metric: _metric_value(canonical_payload, metric),
        )
        passed = bool(passed)
        payload = staged_metric_payload(
            stage_name=stage_name,
            stage_index=stage_index,
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
            passed=passed,
        )
        store.append_metrics(payload, step=step, source="checkpoint_eval", checkpoint_step=step)
        store.mark_checkpoint_eval_stage_succeeded(
            stage_id,
            episodes=episodes,
            n_envs=n_envs,
            metrics=payload,
        )
        next_stage_index = stage_index + 1
        candidate_payload: dict[str, object] = {}
        if passed and next_stage_index < len(stages):
            store.enqueue_checkpoint_eval_stage(
                checkpoint_id,
                stages[next_stage_index],
                stage_index=next_stage_index,
            )
        elif passed and bool(stage.get("candidate_stop")):
            candidate_payload = {
                GLOBAL_STEP: step,
                CHECKPOINT_EVAL_CANDIDATE_PASS: 1.0,
                CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX: float(stage_index),
                CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP: float(step),
                CHECKPOINT_EVAL_CANDIDATE_EPISODES: float(episodes),
            }
            store.append_metrics(
                candidate_payload,
                step=step,
                source="checkpoint_eval",
                checkpoint_step=step,
            )
            store.mark_checkpoint_eval_candidate(
                checkpoint_id,
                episodes=episodes,
                metrics={**payload, **candidate_payload},
            )
        elif not passed:
            store.mark_checkpoint_eval_non_candidate(
                checkpoint_id,
                episodes=episodes,
                metrics=payload,
            )
        else:
            store.mark_checkpoint_eval_non_candidate(
                checkpoint_id,
                episodes=episodes,
                metrics=payload,
            )

        wandb_run = None
        try:
            wandb_run = resume_wandb_run(args, run_dir)
            log_staged_checkpoint_eval_metrics(wandb_run, {**payload, **candidate_payload})
        except Exception as exc:
            print(f"checkpoint staged eval W&B logging failed step={step}: {exc}", flush=True)
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
            "stage_name": stage_name,
            "stage_index": stage_index,
            "episodes": episodes,
            "n_envs": n_envs,
            "passed": passed,
            "candidate_stop": bool(candidate_payload),
            "observed_rules": observed,
            "objective": score[0],
            "reward_mean": float(metrics["reward_mean"]),
            "eval_source": "async_worker",
        }
        update_summary_file(run_dir, summary)
        print(
            "checkpoint staged eval ready: "
            f"checkpoint_id={checkpoint_id} step={step} stage={stage_name} passed={passed}",
            flush=True,
        )
    except Exception as exc:
        store.mark_checkpoint_eval_stage_failed(stage_id, repr(exc))
        print(
            f"checkpoint staged eval failed checkpoint_id={checkpoint_id} "
            f"stage={stage_name}: {exc}",
            flush=True,
        )


def process_eval(
    *,
    store: MetricStore,
    args,
    config,
    run_dir: Path,
    row: dict[str, Any],
) -> None:
    stages = _checkpoint_eval_stages(args)
    if stages:
        if "eval_stage_id" not in row:
            raise ValueError("staged checkpoint eval requires a checkpoint_eval_stages row")
        process_staged_eval(
            store=store,
            args=args,
            config=config,
            run_dir=run_dir,
            row=row,
            stages=stages,
        )
        return
    checkpoint_id = int(row["id"])
    if not store.claim_eval(checkpoint_id):
        return
    checkpoint_path = Path(str(row["path"]))
    step = int(row["step"])
    try:
        eval_model = PPO.load(checkpoint_path, device=resolve_sb3_device(args.device))
        episodes = int(args.post_train_eval_episodes)
        max_steps = int(args.post_train_eval_max_steps or task_max_episode_steps(config))
        n_envs = int(args.checkpoint_eval_n_envs)
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
    stages = _checkpoint_eval_stages(train_args)
    while True:
        if stages:
            store.ensure_checkpoint_eval_stages(stages)
            rows = store.pending_checkpoint_eval_stages(limit=max(args.limit, 1))
            if rows and int(rows[0]["stage_index"]) == 0:
                skipped = store.skip_stale_initial_checkpoint_eval_stages(
                    keep_checkpoint_id=int(rows[0]["id"]),
                )
                if skipped:
                    print(f"checkpoint staged eval skipped stale checkpoints: {skipped}", flush=True)
                    rows = store.pending_checkpoint_eval_stages(limit=max(args.limit, 1))
        else:
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
