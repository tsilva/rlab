from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.wandb_artifacts import artifact_write_ref
from rlab.checkpoint_eval_config import (
    checkpoint_eval_max_steps,
    normalize_checkpoint_eval_stages,
)
from rlab.device import resolve_sb3_device
from rlab.early_stop import evaluate_early_stop_config
from rlab.env import EnvConfig, assert_provider_runtime_available, resolve_env_config
from rlab.env_metadata import env_config_from_config_dict
from rlab.eval_runner import evaluate_model_episodes
from rlab.rom_assets import manifest_from_train_config
from rlab.rom_runtime import RomRuntimeBinding, bind_cached_rom, runtime_cache_root
from rlab.eval_metrics import eval_by_start_rows
from rlab.eval_metrics import completion_score as eval_completion_score
from rlab.metric_names import (
    CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP,
    CHECKPOINT_EVAL_CANDIDATE_EPISODES,
    CHECKPOINT_EVAL_CANDIDATE_PASS,
    CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX,
    EVAL_FULL_BY_START,
    EVAL_FULL_DURATION_SECONDS,
    EVAL_FULL_EPISODE_RETURN_BEST,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    EVAL_FULL_PROGRESS_X_MAX,
    GLOBAL_STEP,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_BEST_RETURN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_LOCAL_PATH,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_OBJECTIVE_NAME,
    LEADER_CHECKPOINT_PROGRESS_MAX,
    LEADER_CHECKPOINT_RANK,
    LEADER_CHECKPOINT_RANK_VALUES,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_GOAL,
    LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_UPDATED_AT,
    checkpoint_eval_stage_metric,
    eval_metric,
    validate_metric_payload,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.policy_models import load_internal_policy_model as load_policy_model
from rlab.ranking import (
    objective_rank_strings,
    parse_objective_rank,
    rank_metric_values,
    rank_score,
    require_objective_rank,
)
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.train_config import materialized_train_args
from rlab.wandb_utils import resolve_wandb_namespace


def eval_checkpoint_artifact_ref(args, checkpoint_path: Path, step: int) -> str:
    if getattr(args, "no_wandb_artifacts", False):
        return str(checkpoint_path)
    entity, project = resolve_wandb_namespace(
        getattr(args, "wandb_entity", None),
        getattr(args, "wandb_project", None),
        str(getattr(args, "game", "") or ""),
        env_provider=getattr(args, "env_provider", None),
    )
    if entity and project:
        return artifact_write_ref(
            namespace=f"{entity}/{project}",
            kind="checkpoint",
            run_id=getattr(args, "wandb_run_id", None),
            alias=f"step-{step}",
        )
    return str(checkpoint_path)


def eval_score(metrics: dict[str, object], selection_rank: object) -> tuple[float, ...]:
    return rank_score(metrics, require_objective_rank(selection_rank))


def update_best_checkpoint_summary(
    wandb_run,
    *,
    metrics: dict[str, object],
    checkpoint_path: Path | str,
    checkpoint_step_value: int,
    artifact_ref: str,
    eval_source: str = "async_worker",
    selection_rank: object = (),
    force: bool = False,
) -> None:
    if wandb_run is None:
        return

    def remove_summary_key(key: str) -> None:
        try:
            del wandb_run.summary[key]
        except AttributeError, KeyError:
            pass

    criteria = require_objective_rank(selection_rank)
    score = rank_score(metrics, criteria)
    values = rank_metric_values(metrics, criteria)
    objective_name = criteria[0].metric
    try:
        previous_rank = parse_objective_rank(wandb_run.summary.get(LEADER_CHECKPOINT_RANK))
        previous_values = wandb_run.summary.get(LEADER_CHECKPOINT_RANK_VALUES)
    except AttributeError:
        previous_rank = ()
        previous_values = None
    if previous_rank == criteria and isinstance(previous_values, list | tuple):
        previous_metrics = {
            criterion.metric: value
            for criterion, value in zip(criteria, previous_values, strict=False)
        }
        previous = rank_score(previous_metrics, criteria)
    else:
        previous = tuple(float("-inf") for _criterion in criteria)
    if not force and score < previous:
        return

    wandb_run.summary[LEADER_CHECKPOINT_OBJECTIVE] = values[0]
    wandb_run.summary[LEADER_CHECKPOINT_OBJECTIVE_NAME] = objective_name
    wandb_run.summary[LEADER_CHECKPOINT_RANK] = list(objective_rank_strings(criteria))
    wandb_run.summary[LEADER_CHECKPOINT_RANK_VALUES] = list(values)
    completion = eval_completion_score(dict(metrics))
    if completion is not None:
        wandb_run.summary[LEADER_CHECKPOINT_SUCCESS_RATE_MIN] = completion[0]
        wandb_run.summary[LEADER_CHECKPOINT_SUCCESS_RATE_MEAN] = completion[1]
    else:
        remove_summary_key(LEADER_CHECKPOINT_SUCCESS_RATE_MIN)
        remove_summary_key(LEADER_CHECKPOINT_SUCCESS_RATE_MEAN)
    wandb_run.summary[LEADER_CHECKPOINT_RETURN_MEAN] = metrics.get(EVAL_FULL_EPISODE_RETURN_MEAN)
    wandb_run.summary[LEADER_CHECKPOINT_BEST_RETURN] = metrics.get(EVAL_FULL_EPISODE_RETURN_BEST)
    wandb_run.summary[LEADER_CHECKPOINT_PROGRESS_MAX] = metrics.get(EVAL_FULL_PROGRESS_X_MAX)
    wandb_run.summary[LEADER_CHECKPOINT_STEP] = checkpoint_step_value
    if any(
        criterion.metric == LEADER_CHECKPOINT_STEPS_TO_GOAL and value is not None
        for criterion, value in zip(criteria, values, strict=True)
    ):
        wandb_run.summary[LEADER_CHECKPOINT_STEPS_TO_GOAL] = checkpoint_step_value
    else:
        remove_summary_key(LEADER_CHECKPOINT_STEPS_TO_GOAL)
    wandb_run.summary[LEADER_CHECKPOINT_ARTIFACT_REF] = artifact_ref
    wandb_run.summary[LEADER_CHECKPOINT_LOCAL_PATH] = str(checkpoint_path)
    wandb_run.summary[LEADER_CHECKPOINT_EVAL_SOURCE] = eval_source
    wandb_run.summary[LEADER_CHECKPOINT_UPDATED_AT] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )


def evaluation_metric_payload(
    *,
    protocol: str,
    metrics: Mapping[str, Any],
    checkpoint_step: int,
    checkpoint_artifact: str,
    eval_source: str,
) -> dict[str, object]:
    payload: dict[str, object] = {GLOBAL_STEP: checkpoint_step}
    for name, value in metrics.items():
        if not name.startswith("eval/full/"):
            continue
        suffix = name.removeprefix("eval/full/")
        if suffix == "episode/return/best" and protocol != "full":
            continue
        if suffix.startswith(("episode/", "outcome/")) or (
            protocol == "full" and suffix.startswith("progress/")
        ):
            payload[eval_metric(protocol, suffix)] = value
    payload[eval_metric(protocol, "checkpoint/step")] = checkpoint_step
    payload[eval_metric(protocol, "checkpoint/artifact")] = checkpoint_artifact
    payload[eval_metric(protocol, "source")] = eval_source
    duration = metrics.get(EVAL_FULL_DURATION_SECONDS)
    if duration is not None:
        payload[eval_metric(protocol, "duration/seconds")] = duration
    validate_metric_payload(payload)
    return payload


def log_checkpoint_eval_metrics(
    wandb_run,
    *,
    args,
    metrics: dict[str, object],
    checkpoint_path: Path | str,
    checkpoint_step_value: int,
    artifact_ref: str,
    eval_source: str = "async_worker",
    config: EnvConfig,
    update_leader: bool = True,
    force_leader: bool = False,
) -> None:
    if wandb_run is None:
        return
    del config
    payload = evaluation_metric_payload(
        protocol="full",
        metrics=metrics,
        checkpoint_step=checkpoint_step_value,
        checkpoint_artifact=artifact_ref,
        eval_source=eval_source,
    )
    table_rows = metrics.get("_eval_by_start_rows")
    episode_results = metrics.get("episode_results")
    if not isinstance(table_rows, list) and isinstance(episode_results, list):
        table_rows = eval_by_start_rows(episode_results)
    if isinstance(table_rows, list):
        import wandb

        columns = [
            "checkpoint_step",
            "start_id",
            "episodes",
            "success_count",
            "success_rate",
            "return_mean",
            "return_std",
            "return_median",
            "reason",
            "reason_count",
            "reason_rate",
        ]
        payload[EVAL_FULL_BY_START] = wandb.Table(
            columns=columns,
            data=[[checkpoint_step_value, *row] for row in table_rows],
        )
    wandb_run.log(payload)
    if update_leader:
        update_best_checkpoint_summary(
            wandb_run,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step_value=checkpoint_step_value,
            artifact_ref=artifact_ref,
            eval_source=eval_source,
            selection_rank=getattr(args, "selection_rank", ()),
            force=force_leader,
        )


def checkpoint_eval_config_from_args(args) -> EnvConfig:
    raw_config = getattr(args, "checkpoint_eval_environment", None)
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            "checkpoint_eval_environment must be materialized from the goal eval contract"
        )
    config = env_config_from_config_dict(dict(raw_config))
    if config is None:
        raise ValueError("checkpoint_eval_environment did not define an environment")
    return resolve_env_config(config)


def metric_payload(
    *,
    args,
    metrics: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_step: int,
    config: EnvConfig,
) -> dict[str, object]:
    del config
    return evaluation_metric_payload(
        protocol="full",
        metrics=metrics,
        checkpoint_step=checkpoint_step,
        checkpoint_artifact=eval_checkpoint_artifact_ref(args, checkpoint_path, checkpoint_step),
        eval_source=str(metrics.get("eval_source") or "async_worker"),
    )


def staged_metric_payload(
    *,
    stage_name: str,
    stage_index: int,
    args,
    metrics: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_step: int,
    passed: bool,
    config: EnvConfig,
) -> dict[str, object]:
    del config
    payload = evaluation_metric_payload(
        protocol=stage_name,
        metrics=metrics,
        checkpoint_step=checkpoint_step,
        checkpoint_artifact=eval_checkpoint_artifact_ref(args, checkpoint_path, checkpoint_step),
        eval_source="async_worker",
    )
    payload[checkpoint_eval_stage_metric(stage_name, "candidate/pass")] = 1.0 if passed else 0.0
    payload[checkpoint_eval_stage_metric(stage_name, "candidate/stage_index")] = float(stage_index)
    validate_metric_payload(payload)
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
    rom_binding: RomRuntimeBinding | None = None,
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
        eval_model = load_policy_model(
            checkpoint_path,
            execution_id=f"checkpoint-eval:{getattr(args, 'wandb_run_id', '')}:{checkpoint_id}",
            device=resolve_sb3_device(args.device),
        )
        episodes = int(stage["episodes"])
        max_steps = checkpoint_eval_max_steps(vars(args))
        n_envs = int(stage.get("n_envs") or getattr(args, "checkpoint_eval_n_envs", 20))
        metrics, _video_path = evaluate_model_episodes(
            model=eval_model,
            config=config,
            episodes=episodes,
            seed=DEFAULT_EVAL_SEED,
            max_steps=max_steps,
            deterministic=False,
            n_envs=n_envs,
            progress=True,
            progress_description=f"eval checkpoint {step} {stage_name}",
            extra={
                "checkpoint_step": step,
                "checkpoint_artifact": str(checkpoint_path),
                "eval_source": f"async_worker:{stage_name}",
            },
            rom_binding=rom_binding,
        )
        canonical_payload = metric_payload(
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
            config=config,
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
            config=config,
        )
        store.append_metrics(
            payload,
            step=step,
            source="checkpoint_eval",
            publish=bool(getattr(args, "wandb", False)),
        )
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
                publish=bool(getattr(args, "wandb", False)),
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

        score = eval_score(metrics, getattr(args, "selection_rank", ()))
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
            "return_mean": float(metrics["return_mean"]),
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
    rom_binding: RomRuntimeBinding | None = None,
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
            rom_binding=rom_binding,
        )
        return
    checkpoint_id = int(row["id"])
    if not store.claim_eval(checkpoint_id):
        return
    checkpoint_path = Path(str(row["path"]))
    step = int(row["step"])
    try:
        eval_model = load_policy_model(
            checkpoint_path,
            execution_id=f"checkpoint-eval:{getattr(args, 'wandb_run_id', '')}:{checkpoint_id}",
            device=resolve_sb3_device(args.device),
        )
        episodes = int(args.post_train_eval_episodes)
        max_steps = checkpoint_eval_max_steps(vars(args))
        n_envs = int(args.checkpoint_eval_n_envs)
        metrics, _video_path = evaluate_model_episodes(
            model=eval_model,
            config=config,
            episodes=episodes,
            seed=DEFAULT_EVAL_SEED,
            max_steps=max_steps,
            deterministic=False,
            n_envs=n_envs,
            progress=True,
            progress_description=f"eval checkpoint {step}",
            extra={
                "checkpoint_step": step,
                "checkpoint_artifact": str(checkpoint_path),
                "eval_source": "async_worker",
            },
            rom_binding=rom_binding,
        )
        payload = metric_payload(
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step=step,
            config=config,
        )
        store.append_metrics(
            payload,
            step=step,
            source="eval",
            publish=False,
        )
        store.mark_eval_succeeded(checkpoint_id, episodes=episodes, metrics=payload)
        artifact_ref = eval_checkpoint_artifact_ref(args, checkpoint_path, step)
        store.enqueue_event(
            kind="checkpoint_eval",
            payload={
                "metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": step,
                "artifact_ref": artifact_ref,
                "eval_source": "async_worker",
            },
            step=step,
            source="checkpoint_eval",
            event_id=f"checkpoint-eval:{checkpoint_id}:{step}",
        )
        score = eval_score(metrics, getattr(args, "selection_rank", ()))
        summary = {
            "checkpoint_step": step,
            "checkpoint_path": str(checkpoint_path),
            "objective": score[0],
            "return_mean": float(metrics["return_mean"]),
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
    train_args = materialized_train_args(args.train_config_json)
    config = checkpoint_eval_config_from_args(train_args)
    manifest = manifest_from_train_config(vars(train_args), expected_game=config.game)
    rom_binding = (
        bind_cached_rom(manifest, cache_root=runtime_cache_root(container_default=True))
        if manifest is not None
        else None
    )
    assert_provider_runtime_available(config, rom_binding=rom_binding)
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
                    print(
                        f"checkpoint staged eval skipped stale checkpoints: {skipped}", flush=True
                    )
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
                rom_binding=rom_binding,
            )
        if not rows:
            time.sleep(max(args.poll_seconds, 0.25))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
