from __future__ import annotations

import re
import time
import uuid
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from rlab.artifacts import write_model_metadata
from rlab.action_contract import configured_action_meanings
from rlab.batch_runtime import EpisodeRecord
from rlab.early_stop import evaluate_early_stop_config
from rlab.env import make_training_vec_env
from rlab.jerk import JerkSearch
from rlab.metric_names import (
    TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN,
    TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH,
    TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN,
    TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY,
    TRAIN_ALGORITHM_JERK_RETAINED_COUNT,
    TRAIN_EPISODE_LENGTH_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
    TRAIN_OUTCOME_TERMINAL_COUNT,
    TRAIN_THROUGHPUT_LOOP_FPS,
    metric_value_segment,
    train_success_attempts_metric,
    train_success_count_metric,
    train_success_window_rate_metric,
)
from rlab.task_kernels import Outcome
from rlab.training_backend import (
    CHECKPOINT_EVAL_ACCEPTANCE,
    FIRST_TRAINING_SUCCESS_ACCEPTANCE,
    BackendContext,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "acceptance_mode": CHECKPOINT_EVAL_ACCEPTANCE,
    "archive_replay_probability_initial": 0.25,
    "archive_replay_probability_max": 0.9,
    "protected_prefix_steps": 128,
    "max_prefix_shorten_steps": 128,
    "retained_limit": 256,
    "fallback_action": "noop",
    "log_interval_steps": 10_000,
}

_POSITIVE_INTEGER_FIELDS = {
    "max_prefix_shorten_steps",
    "retained_limit",
    "log_interval_steps",
}
_NON_NEGATIVE_INTEGER_FIELDS = {"protected_prefix_steps"}
_PROBABILITY_FIELDS = {
    "archive_replay_probability_initial",
    "archive_replay_probability_max",
}
_ACTION_FIELDS = {"fallback_action"}
_ACCEPTANCE_MODES = {CHECKPOINT_EVAL_ACCEPTANCE, FIRST_TRAINING_SUCCESS_ACCEPTANCE}


def normalize_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in _POSITIVE_INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
    for key in _NON_NEGATIVE_INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
    for key in _PROBABILITY_FIELDS:
        value = normalized[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
    for key in _ACTION_FIELDS:
        if not isinstance(normalized[key], str) or not normalized[key].strip():
            raise ValueError(f"{label}.{key} must be a non-empty string")
    if normalized["acceptance_mode"] not in _ACCEPTANCE_MODES:
        allowed = ", ".join(sorted(_ACCEPTANCE_MODES))
        raise ValueError(f"{label}.acceptance_mode must be one of: {allowed}")
    if (
        normalized["archive_replay_probability_initial"]
        > normalized["archive_replay_probability_max"]
    ):
        raise ValueError(
            f"{label}.archive_replay_probability_initial must not exceed "
            f"{label}.archive_replay_probability_max"
        )
    return normalized


def _is_success(record: EpisodeRecord) -> bool:
    return record.outcome == Outcome.SUCCESS or bool(record.metrics.get("level_complete"))


class _OutcomeMetrics:
    def __init__(self, *, configured_starts: tuple[str, ...]) -> None:
        self.configured_starts = tuple(dict.fromkeys(configured_starts))
        self.terminal_count = 0
        self.success_counts: dict[str, int] = {}
        self.attempt_counts: dict[str, int] = {}
        self.windows: dict[str, deque[bool]] = {}

    def consume(self, record: EpisodeRecord, *, fallback_start: str) -> None:
        self.terminal_count += 1
        start = metric_value_segment(record.start_id or fallback_start)
        window = self.windows.setdefault(start, deque(maxlen=100))
        completed = _is_success(record)
        window.append(completed)
        self.attempt_counts[start] = self.attempt_counts.get(start, 0) + 1
        if completed:
            self.success_counts[start] = self.success_counts.get(start, 0) + 1

    def payload(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {
            TRAIN_OUTCOME_TERMINAL_COUNT: self.terminal_count,
        }
        if not self.attempt_counts:
            return payload
        current_rates: dict[str, float] = {}
        for start, attempts in self.attempt_counts.items():
            successes = self.success_counts.get(start, 0)
            rate = successes / attempts
            current_rates[start] = rate
            payload[train_success_count_metric(start)] = successes
            payload[train_success_attempts_metric(start)] = attempts
            if len(self.windows[start]) >= 100:
                payload[train_success_window_rate_metric(start)] = sum(self.windows[start]) / len(
                    self.windows[start]
                )
        expected_starts = self.configured_starts or tuple(self.attempt_counts)
        expected_segments = tuple(metric_value_segment(start) for start in expected_starts)
        payload[TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN] = min(current_rates.values())
        payload[TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN] = float(
            np.mean(tuple(current_rates.values()))
        )
        payload[TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE] = sum(
            start in self.attempt_counts for start in expected_segments
        ) / len(expected_segments)
        if expected_segments and all(
            len(self.windows.get(start, ())) >= 100 for start in expected_segments
        ):
            window_rates = [
                sum(self.windows[start]) / len(self.windows[start]) for start in expected_segments
            ]
            payload[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN] = min(window_rates)
            payload[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN] = float(np.mean(window_rates))
        return payload


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
    outcome_metrics: _OutcomeMetrics,
) -> None:
    candidate = search.best_candidate()
    payload: dict[str, int | float] = {
        TRAIN_ALGORITHM_JERK_RETAINED_COUNT: search.retained_count,
        TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY: search.archive_replay_probability,
        TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN: (
            search.archive_selected_prefix_return_mean
        ),
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
    payload.update(outcome_metrics.payload())
    context.metric_store.append_metrics(
        payload,
        step=step,
        source="train",
        publish=context.wandb_enabled,
    )


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
    env = make_training_vec_env(
        config=config,
        n_envs=n_envs,
        seed=args.seed,
        rom_binding=getattr(context, "rom_binding", None),
    )
    try:
        if int(args.timesteps) % n_envs != 0:
            raise ValueError("JERK timesteps must be divisible by the environment count")
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("JERK requires a discrete task action space")
        action_names = configured_action_meanings(config)
        search = JerkSearch(
            n_envs=n_envs,
            seed=args.seed,
            total_timesteps=args.timesteps,
            action_names=action_names,
            fallback_action=args.fallback_action,
            archive_replay_probability_initial=args.archive_replay_probability_initial,
            archive_replay_probability_max=args.archive_replay_probability_max,
            protected_prefix_steps=args.protected_prefix_steps,
            max_prefix_shorten_steps=args.max_prefix_shorten_steps,
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
        configured_starts = tuple(
            str(start)
            for start in (
                tuple(config.states)
                if getattr(config, "states", ())
                else (getattr(config, "state", None),)
            )
            if start
        )
        fallback_start = configured_starts[0] if configured_starts else "default"
        outcome_metrics = _OutcomeMetrics(configured_starts=configured_starts)
        acceptance_mode = str(args.acceptance_mode)
        accepted = False
        early_stopped = False
        while search.global_step < args.timesteps and not context.stop_flag.requested:
            actions = search.next_actions()
            _observations, rewards, dones, _infos = env.step(actions)
            records = env.drain_records()
            records_by_lane: dict[int, EpisodeRecord] = {}
            success_records: list[EpisodeRecord] = []
            for record in records:
                if isinstance(record, EpisodeRecord):
                    records_by_lane[int(record.lane)] = record
                    episode_returns.append(float(record.episode_return))
                    episode_lengths.append(int(record.episode_length))
                    outcome_metrics.consume(record, fallback_start=fallback_start)
                    if _is_success(record):
                        success_records.append(record)
            search.observe(rewards, dones, records_by_lane)
            step = search.global_step
            if acceptance_mode == FIRST_TRAINING_SUCCESS_ACCEPTANCE and success_records:
                accepted = True
                accepted_path = context.checkpoint_dir / (
                    f"{_checkpoint_prefix(config.game)}_{step}_steps.zip"
                )
                _save_policy_bundle(
                    search=search,
                    context=context,
                    model_path=accepted_path,
                    kind="checkpoint",
                    step=step,
                )
                print(
                    f"accepted JERK policy at first training success: step={step} "
                    f"start={success_records[0].start_id or fallback_start}"
                )
                break
            if step >= next_log:
                _publish_metrics(
                    context,
                    search,
                    step=step,
                    elapsed=time.perf_counter() - started_at,
                    returns=episode_returns,
                    lengths=episode_lengths,
                    outcome_metrics=outcome_metrics,
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
            outcome_metrics=outcome_metrics,
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
            f"episodes={search.completed_episodes} accepted={accepted} "
            f"early_stopped={early_stopped}"
        )
        if (
            acceptance_mode == FIRST_TRAINING_SUCCESS_ACCEPTANCE
            and not accepted
            and not context.stop_flag.requested
            and step >= args.timesteps
        ):
            raise RuntimeError(
                f"JERK exhausted {args.timesteps} transitions without a goal success event"
            )
    finally:
        env.close()


class JerkBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        normalized = normalize_config(backend_config, label="training_backend.config")
        if (
            normalized["acceptance_mode"] == FIRST_TRAINING_SUCCESS_ACCEPTANCE
            and common_config.get("checkpoint_eval_backend") != "none"
        ):
            raise ValueError(
                "training_backend.config.acceptance_mode=first_training_success requires "
                "checkpoint_eval_backend=none"
            )

    def run(self, context: BackendContext) -> None:
        run_jerk(context)


_BACKEND = JerkBackend()


def acceptance_mode(backend_id: str, backend_config: Mapping[str, Any]) -> str:
    if backend_id != "rlab.jerk":
        raise ValueError(f"JERK backend module does not define {backend_id!r}")
    return str(normalize_config(backend_config, label="training_backend.config")["acceptance_mode"])


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
