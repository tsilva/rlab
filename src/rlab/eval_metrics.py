from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from rlab.metric_names import (
    EVAL_FULL_SUCCESS_RATE_MEAN,
    EVAL_FULL_SUCCESS_RATE_MIN,
    eval_metric,
    eval_progress_metric,
    eval_reason_rate_metric,
    eval_success_from_rate_metric,
    eval_success_rate_metric,
)
from rlab.targets import EvalSemantics, target_for_game
from rlab.task_kernels import Outcome

DEFAULT_EVAL_SEMANTICS = target_for_game("SuperMarioBros-Nes-v0").eval_semantics


def default_eval_semantics() -> EvalSemantics:
    return DEFAULT_EVAL_SEMANTICS


def is_completion_event(
    info: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> bool:
    semantics = semantics or default_eval_semantics()
    for key in semantics.completion_info_keys:
        if key in info:
            return bool(info.get(key))
    return False


def is_level_complete(info: dict[str, Any]) -> bool:
    return is_completion_event(info, default_eval_semantics())


def drain_runtime_records(env: Any) -> list[Any]:
    """Drain all records from the native vector runtime."""
    drain = getattr(env, "drain_records", None)
    if not callable(drain):
        raise TypeError("this workflow requires RlabVecEnv.drain_records()")
    return list(drain())


def episode_records(records: list[Any]) -> list[Any]:
    return [record for record in records if hasattr(record, "episode_return")]


def batch_metrics_for_lane(records: list[Any], lane: int) -> dict[str, Any]:
    """Materialize the latest task metric batch for an interactive consumer."""
    for record in reversed(records):
        if hasattr(record, "lane") or not hasattr(record, "num_envs"):
            continue
        metrics = getattr(record, "metrics", {}) or {}
        result: dict[str, Any] = {}
        for name, values in metrics.items():
            value = np.asarray(values)[lane]
            result[str(name)] = value.item() if isinstance(value, np.generic) else value
        return result
    return {}


def drain_episode_records(env: Any) -> list[Any]:
    """Drain canonical episode records from the native vector runtime."""
    return episode_records(drain_runtime_records(env))


def outcome_name(value: Any) -> str:
    if isinstance(value, Outcome):
        return value.name.lower()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    if isinstance(value, str):
        return value.lower()
    try:
        return Outcome(int(value)).name.lower()
    except TypeError, ValueError:
        return "neutral"


def episode_is_complete(episode: Mapping[str, Any]) -> bool:
    if "level_complete" in episode:
        return bool(episode.get("level_complete"))
    return str(episode.get("outcome", "")).lower() == "success"


def episode_result_from_record(
    record: Any,
    *,
    semantics: EvalSemantics | None = None,
    terminal_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate the runtime's provider-neutral episode record to eval output."""
    semantics = semantics or default_eval_semantics()
    metrics = dict(getattr(record, "metrics", {}) or {})
    events = tuple(str(event) for event in (getattr(record, "events", ()) or ()))
    outcome = outcome_name(getattr(record, "outcome", Outcome.NEUTRAL))
    info = serializable_info(dict(terminal_info or {}))
    # Canonical task metrics are authoritative for overlapping provider fields.
    info.update(metrics)

    start_id = getattr(record, "start_id", None)
    result: dict[str, Any] = {
        "env_index": int(getattr(record, "lane", 0)),
        "episode_index": int(getattr(record, "episode_index", 0)),
        "start_state": start_id or info.get("start_state") or info.get("state"),
        "return": float(getattr(record, "episode_return", 0.0)),
        "score": int(info.get("score", 0) or 0),
        "lives": int(info.get("lives", 0) or 0),
        "time": int(info.get("time", 0) or 0),
        "steps": int(getattr(record, "episode_length", 0)),
        "terminated": bool(getattr(record, "terminated", False)),
        "truncated": bool(getattr(record, "truncated", False)),
        "outcome": outcome,
        "events": list(events),
        "final_info": info,
    }

    died = bool(metrics.get("died", False)) or "life_loss" in events
    if semantics.completion_reason:
        explicit_success = outcome == "success"
        explicit_failure = outcome == "failure"
        completion_signal = semantics.completion_reason in events or is_completion_event(
            info, semantics
        )
        result["level_complete"] = bool(
            (explicit_success and not died)
            or (completion_signal and not died and not explicit_failure)
        )

    for field in semantics.progress_fields:
        value = metrics.get(field.result_key, info.get(field.info_key))
        if value is None and field.result_key == "max_level_x_pos":
            value = metrics.get("max_x_pos", 0)
        result[field.result_key] = int(value or 0)

    if semantics.death_flag_key:
        death_x_pos = metrics.get("death_x_pos", info.get(semantics.death_position_key or ""))
        if died and death_x_pos is None:
            death_x_pos = result.get("max_x_pos", 0)
        result["died"] = died
        result["death_x_pos"] = int(death_x_pos) if death_x_pos is not None else None
    return result


def death_location_histogram(death_x_positions: list[int], bin_size: int = 100) -> dict[str, int]:
    bins: dict[str, int] = {}
    for x_pos in death_x_positions:
        start = (int(x_pos) // bin_size) * bin_size
        key = f"{start}-{start + bin_size - 1}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def episode_start_state(episode: dict[str, Any]) -> str | None:
    state = episode.get("start_state") or episode.get("start_id") or episode.get("state")
    final_info = episode.get("final_info")
    if not state and isinstance(final_info, dict):
        state = final_info.get("start_state") or final_info.get("state")
    return str(state) if state else None


def serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    result = dict(info)
    result.pop("terminal_observation", None)
    return result


def episode_reasons(episode: Mapping[str, Any]) -> set[str]:
    if episode_is_complete(episode):
        return set()
    return episode_reason_names(
        episode.get("events", ()) or (),
        terminated=bool(episode.get("terminated")),
        truncated=bool(episode.get("truncated")),
    )


def episode_reason_names(
    events: Sequence[object],
    *,
    terminated: bool,
    truncated: bool,
) -> set[str]:
    """Return the shared train/eval terminal-reason taxonomy."""
    reasons = {str(event) for event in events if str(event) != "timeout"}
    if truncated:
        reasons.add("max_steps")
    if not reasons:
        reasons.add("terminated" if terminated else "unclassified")
    return reasons


def eval_outcome_metrics(
    episode_results: list[dict[str, Any]],
    *,
    protocol: str = "full",
    event_names: Sequence[str] = (),
    track_success: bool = False,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    success_rates: list[float] = []
    configured_reasons = set(str(name) for name in event_names)
    all_reasons = sorted(
        configured_reasons
        | {reason for episode in episode_results for reason in episode_reasons(episode)}
    )
    episode_count = len(episode_results)
    for reason in all_reasons:
        count = sum(reason in episode_reasons(episode) for episode in episode_results)
        metrics[eval_reason_rate_metric(protocol, reason)] = count / episode_count

    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        state_episodes = [
            episode for episode in episode_results if episode_start_state(episode) == state
        ]
        denominator = len(state_episodes)
        if track_success:
            success_count = sum(episode_is_complete(episode) for episode in state_episodes)
            success_rate = success_count / denominator
            success_rates.append(success_rate)
            metrics[eval_success_from_rate_metric(protocol, state)] = success_rate
    if success_rates:
        metrics[eval_success_rate_metric(protocol, "min")] = min(success_rates)
        metrics[eval_success_rate_metric(protocol, "mean")] = float(np.mean(success_rates))
    return metrics


def eval_by_start_rows(episode_results: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        episodes = [episode for episode in episode_results if episode_start_state(episode) == state]
        returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
        success_count = sum(episode_is_complete(episode) for episode in episodes)
        reason_counts: dict[str, int] = {}
        for episode in episodes:
            for reason in episode_reasons(episode):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reasons = sorted(reason_counts) or [""]
        for reason in reasons:
            reason_count = reason_counts.get(reason, 0)
            rows.append(
                [
                    state,
                    len(episodes),
                    success_count,
                    success_count / len(episodes),
                    float(np.mean(returns)),
                    float(np.std(returns)),
                    float(np.median(returns)),
                    reason,
                    reason_count,
                    reason_count / len(episodes),
                ]
            )
    return rows


def primary_progress_value(
    result: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> float:
    semantics = semantics or default_eval_semantics()
    for field in semantics.progress_fields:
        if field.rank:
            return float(result.get(field.result_key, 0.0) or 0.0)
    return 0.0


def episode_rank(
    result: dict[str, Any],
    semantics: EvalSemantics | None = None,
) -> tuple[float, ...]:
    semantics = semantics or default_eval_semantics()
    values: list[float] = []
    for item in semantics.best_episode_rank:
        if item == "completion":
            values.append(float(episode_is_complete(result)))
        elif item == "progress":
            values.append(primary_progress_value(result, semantics))
        elif item == "reward":
            values.append(float(result.get("return", 0.0) or 0.0))
        else:
            values.append(float(result.get(item, 0.0) or 0.0))
    return tuple(values or [float(result.get("return", 0.0) or 0.0)])


def progress_summary_fields(result_key: str) -> tuple[str, str]:
    if result_key == "max_x_pos":
        return ("max_x_mean", "max_x_max")
    if result_key == "max_level_x_pos":
        return ("max_level_x_mean", "max_level_x_max")
    return (f"{result_key}_mean", f"{result_key}_max")


def single_env_action(action) -> int | np.ndarray:
    action_array = np.asarray(action)
    if action_array.shape == ():
        return int(action_array)
    first = np.asarray(action_array[0])
    if first.shape == ():
        return int(first)
    return first.astype(np.int8, copy=True)


def summarize_episode_results(
    episode_results: list[dict[str, Any]],
    *,
    deterministic: bool,
    extra: dict[str, Any] | None = None,
    semantics: EvalSemantics | None = None,
    event_names: Sequence[str] = (),
    track_success: bool = False,
) -> dict[str, Any]:
    if not episode_results:
        raise ValueError("episode_results must not be empty")
    semantics = semantics or default_eval_semantics()

    returns = np.array([episode["return"] for episode in episode_results], dtype=np.float64)
    lengths = np.array([episode["steps"] for episode in episode_results], dtype=np.float64)
    progress_metrics: dict[str, int | float] = {}
    for field in semantics.progress_fields:
        values = np.array(
            [episode.get(field.result_key, 0) for episode in episode_results],
            dtype=np.float64,
        )
        mean_key, max_key = progress_summary_fields(field.result_key)
        progress_metrics[mean_key] = float(values.mean())
        progress_metrics[max_key] = int(values.max())
        progress_name = (
            "x"
            if field.result_key == "max_x_pos"
            else "level_x"
            if field.result_key == "max_level_x_pos"
            else field.result_key.removeprefix("max_").removesuffix("_pos")
        )
        progress_metrics[eval_progress_metric("full", progress_name, "mean")] = float(values.mean())
        progress_metrics[eval_progress_metric("full", progress_name, "max")] = int(values.max())
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode_is_complete(episode))
    death_count = sum(1 for episode in episode_results if episode.get("died"))
    episode_count = len(episode_results)
    metrics: dict[str, Any] = {
        "episodes": episode_count,
        "deterministic": deterministic,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_median": float(np.median(returns)),
        "episode_length_mean": float(lengths.mean()),
        eval_metric("full", "episode/return/mean"): float(returns.mean()),
        eval_metric("full", "episode/return/std"): float(returns.std()),
        eval_metric("full", "episode/return/median"): float(np.median(returns)),
        eval_metric("full", "episode/return/best"): float(returns.max()),
        eval_metric("full", "episode/length/mean"): float(lengths.mean()),
        eval_metric("full", "episode/count"): episode_count,
        "episode_results": episode_results,
    }
    metrics.update(progress_metrics)
    if track_success:
        metrics["success_count"] = completion_count
        metrics["success_rate"] = completion_count / episode_count
    if semantics.death_flag_key:
        metrics.update(
            {
                "death_count": death_count,
                "death_rate": death_count / episode_count,
                "death_x_histogram": death_location_histogram(death_x_positions),
            }
        )
    metrics.update(
        eval_outcome_metrics(
            episode_results,
            event_names=event_names,
            track_success=track_success,
        )
    )
    if extra:
        metrics = {**extra, **metrics}
    return metrics


def metric_float(metrics: dict[str, Any] | Any, key: str, default: float = float("-inf")) -> float:
    value = metrics.get(key) if hasattr(metrics, "get") else None
    if value is None:
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def completion_score(metrics: dict[str, Any]) -> tuple[float, float] | None:
    completion_min = metric_float(metrics, EVAL_FULL_SUCCESS_RATE_MIN)
    completion_mean = metric_float(metrics, EVAL_FULL_SUCCESS_RATE_MEAN)
    if completion_min == float("-inf"):
        return None
    return (float(completion_min), float(completion_mean))


def run_eval_episode(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    seed: int,
    capture_actions: bool = False,
    default_start_state: str | None = None,
    semantics: EvalSemantics | None = None,
    observation_callback: Callable[[object], object] | None = None,
) -> dict[str, Any]:
    semantics = semantics or default_eval_semantics()
    reset_episode = getattr(model, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()
    env.seed(seed)
    obs = env.reset()
    actions: list[Any] = []

    for _step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_value = single_env_action(action)
        if capture_actions:
            actions.append(action_value)
        obs, _rewards, dones, infos = env.step(action)
        if observation_callback is not None:
            observation_callback(obs)
        info = dict(infos[0])
        records = drain_episode_records(env)
        if records:
            result = episode_result_from_record(
                records[0],
                semantics=semantics,
                terminal_info=info,
            )
            if result.get("start_state") is None:
                result["start_state"] = default_start_state
            result["actions"] = actions
            return result
        if bool(dones[0]):
            raise RuntimeError("RlabVecEnv returned done without an episode record")
    raise RuntimeError("task runtime reached max_steps without a timeout episode record")
