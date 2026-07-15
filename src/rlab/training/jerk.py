from __future__ import annotations

import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from rlab.artifacts import write_model_metadata
from rlab.batch_runtime import EpisodeRecord
from rlab.early_stop import evaluate_early_stop_config
from rlab.env import make_training_vec_env, task_action_set
from rlab.jerk import JerkSearch
from rlab.metric_names import (
    GLOBAL_STEP,
    TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN,
    TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH,
    TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY,
    TRAIN_ALGORITHM_JERK_RETAINED_COUNT,
    TRAIN_EPISODE_COUNT,
    TRAIN_EPISODE_LENGTH_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_THROUGHPUT_LOOP_FPS,
)
from rlab.targets import target_for_game
from rlab.training_backend import BackendContext


DEFAULT_CONFIG: dict[str, Any] = {
    "exploit_bias": 0.25,
    "forward_steps": 100,
    "backtrack_steps": 70,
    "jump_probability": 0.1,
    "jump_repeat": 4,
    "retained_limit": 256,
    "forward_action": "right_b",
    "jump_action": "right_a_b",
    "backtrack_action": "left",
    "fallback_action": "noop",
    "log_interval_steps": 10_000,
}

_POSITIVE_INTEGER_FIELDS = {
    "forward_steps",
    "backtrack_steps",
    "jump_repeat",
    "retained_limit",
    "log_interval_steps",
}
_PROBABILITY_FIELDS = {"exploit_bias", "jump_probability"}
_ACTION_FIELDS = {"forward_action", "jump_action", "backtrack_action", "fallback_action"}


def normalize_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in _POSITIVE_INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
    for key in _PROBABILITY_FIELDS:
        value = normalized[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
    for key in _ACTION_FIELDS:
        if not isinstance(normalized[key], str) or not normalized[key].strip():
            raise ValueError(f"{label}.{key} must be a non-empty string")
    return normalized


def _checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"jerk_{slug or 'retro'}"


def _save_policy_bundle(
    *,
    search: JerkSearch,
    context: BackendContext,
    model_path: Path,
    kind: str,
    step: int,
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.parent / f".{model_path.stem}.{uuid.uuid4().hex}.zip"
    search.policy().save(temporary)
    temporary.replace(model_path)
    metadata_path = write_model_metadata(
        model_path,
        context.args,
        context.environment,
        kind,
        checkpoint_step_value=step,
    )
    checkpoint_id = context.metric_store.record_checkpoint(
        run_name=str(context.args.run_name),
        kind=kind,
        step=step,
        path=model_path,
        metadata_path=metadata_path,
        sha256=None,
        eval_required=context.args.checkpoint_eval_backend != "none",
    )
    print(f"{kind} JERK policy ready: id={checkpoint_id} step={step} path={model_path}")
    return model_path


def _publish_metrics(
    context: BackendContext,
    search: JerkSearch,
    *,
    step: int,
    elapsed: float,
    returns: list[float],
    lengths: list[int],
) -> None:
    candidate = search.best_candidate()
    payload: dict[str, int | float] = {
        TRAIN_EPISODE_COUNT: search.completed_episodes,
        TRAIN_ALGORITHM_JERK_RETAINED_COUNT: search.retained_count,
        TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY: search.exploit_probability,
        TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH: (
            len(candidate.actions) if candidate is not None else 0
        ),
        TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN: (
            candidate.mean_return if candidate is not None else 0.0
        ),
        TRAIN_THROUGHPUT_LOOP_FPS: step / max(elapsed, 1e-9),
    }
    if returns:
        payload[TRAIN_EPISODE_RETURN_SHAPED_MEAN] = float(np.mean(returns[-100:]))
        payload[TRAIN_EPISODE_LENGTH_MEAN] = float(np.mean(lengths[-100:]))
    context.metric_store.append_metrics(
        payload,
        step=step,
        source="train",
        publish=context.wandb_run is None,
    )
    if context.wandb_run is not None:
        context.wandb_run.log({GLOBAL_STEP: step, **payload})


def _early_stop_requested(context: BackendContext, *, step: int) -> bool:
    detector = context.args.early_stop
    if not detector:
        return False
    matched, values = evaluate_early_stop_config(
        detector,
        lambda metric: context.metric_store.latest_metric(metric),
    )
    if matched is not True:
        return False
    path = context.run_dir / "early_stop.txt"
    path.write_text(
        "early_stop=metric_threshold\n"
        + f"timesteps={step}\n"
        + "".join(f"{name}={value:.12g}\n" for name, value in sorted(values.items())),
        encoding="utf-8",
    )
    return True


def run_jerk(context: BackendContext) -> None:
    args = context.args
    config = context.environment
    n_envs = int(args.resolved_n_envs)
    env = make_training_vec_env(config=config, n_envs=n_envs, seed=args.seed)
    try:
        if int(args.timesteps) % n_envs != 0:
            raise ValueError("JERK timesteps must be divisible by the environment count")
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("JERK requires a discrete task action space")
        action_names = target_for_game(config.game).action_names_for_set(task_action_set(config))
        search = JerkSearch(
            n_envs=n_envs,
            seed=args.seed,
            total_timesteps=args.timesteps,
            action_names=action_names,
            forward_action=args.forward_action,
            jump_action=args.jump_action,
            backtrack_action=args.backtrack_action,
            fallback_action=args.fallback_action,
            forward_steps=args.forward_steps,
            backtrack_steps=args.backtrack_steps,
            jump_probability=args.jump_probability,
            jump_repeat=args.jump_repeat,
            exploit_bias=args.exploit_bias,
            retained_limit=args.retained_limit,
        )
        env.reset()
        context.mark_ready()
        started_at = time.perf_counter()
        next_log = args.log_interval_steps
        next_checkpoint = args.checkpoint_freq if args.checkpoint_freq > 0 else None
        next_early_stop_poll = time.monotonic() + 30.0
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        early_stopped = False
        while search.global_step < args.timesteps and not context.stop_flag.requested:
            actions = search.next_actions()
            _observations, rewards, dones, _infos = env.step(actions)
            records = env.drain_records()
            records_by_lane: dict[int, EpisodeRecord] = {}
            for record in records:
                if isinstance(record, EpisodeRecord):
                    records_by_lane[int(record.lane)] = record
                    episode_returns.append(float(record.episode_return))
                    episode_lengths.append(int(record.episode_length))
            search.observe(rewards, dones, records_by_lane)
            step = search.global_step
            if step >= next_log:
                _publish_metrics(
                    context,
                    search,
                    step=step,
                    elapsed=time.perf_counter() - started_at,
                    returns=episode_returns,
                    lengths=episode_lengths,
                )
                next_log += args.log_interval_steps
            while next_checkpoint is not None and step >= next_checkpoint:
                checkpoint_path = context.checkpoint_dir / (
                    f"{_checkpoint_prefix(config.game)}_{step}_steps.zip"
                )
                _save_policy_bundle(
                    search=search,
                    context=context,
                    model_path=checkpoint_path,
                    kind="checkpoint",
                    step=step,
                )
                next_checkpoint += args.checkpoint_freq
            if time.monotonic() >= next_early_stop_poll:
                next_early_stop_poll = time.monotonic() + 30.0
                if _early_stop_requested(context, step=step):
                    print(f"early stop: checkpoint evaluation accepted at step={step}")
                    early_stopped = True
                    break

        step = search.global_step
        _publish_metrics(
            context,
            search,
            step=step,
            elapsed=time.perf_counter() - started_at,
            returns=episode_returns,
            lengths=episode_lengths,
        )
        if context.stop_flag.requested and args.checkpoint_freq > 0:
            interrupted = context.checkpoint_dir / (
                f"{_checkpoint_prefix(config.game)}_interrupted_{step}_steps.zip"
            )
            _save_policy_bundle(
                search=search,
                context=context,
                model_path=interrupted,
                kind="interrupted",
                step=step,
            )
        final_path = context.run_dir / "final_model.zip"
        _save_policy_bundle(
            search=search,
            context=context,
            model_path=final_path,
            kind="final",
            step=step,
        )
        print(
            f"saved {final_path} retained={search.retained_count} "
            f"episodes={search.completed_episodes} early_stopped={early_stopped}"
        )
    finally:
        env.close()


class JerkBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config
        normalize_config(backend_config, label="training_backend.config")

    def run(self, context: BackendContext) -> None:
        run_jerk(context)


_BACKEND = JerkBackend()


def backend_for_id(backend_id: str) -> JerkBackend:
    if backend_id != "rlab.jerk":
        raise ValueError(f"JERK backend module does not define {backend_id!r}")
    return _BACKEND


def contract_payload(backend_id: str) -> dict[str, Any]:
    if backend_id != "rlab.jerk":
        raise ValueError(f"JERK backend module does not define {backend_id!r}")
    return {"schema_version": 1, "status": "available", "defaults": DEFAULT_CONFIG}


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    del backend_config
    if backend_id != "rlab.jerk":
        raise ValueError(f"JERK backend module does not define {backend_id!r}")
    return {
        "training_backend_id": backend_id,
        "algorithm_id": "jerk",
        "model_class": "rlab.jerk.JerkPolicy",
    }
