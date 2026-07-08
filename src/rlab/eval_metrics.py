from __future__ import annotations

from typing import Any

import numpy as np

from rlab.metric_names import (
    EVAL_DONE_ALL,
    EVAL_DONE_LEVEL_CHANGE,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    EVAL_DONE_LEVEL_CHANGE_RATE,
    EVAL_DONE_MAX_STEPS,
    EVAL_DONE_MAX_STEPS_RATE,
    EVAL_DONE_TERMINATED,
    EVAL_DONE_TERMINATED_RATE,
    EVAL_DONE_UNCLASSIFIED,
    EVAL_DONE_UNCLASSIFIED_RATE,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
    eval_done_value_metric,
)
from rlab.targets import EvalSemantics, target_for_game

COMPLETION_GOAL_RATE = 0.99
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
    fallback_key = semantics.completion_fallback_info_key
    if fallback_key is None:
        return False
    if not bool(info.get(fallback_key, False)):
        return False
    return not any(bool(info.get(key, False)) for key in semantics.completion_blocking_info_keys)


def is_level_complete(info: dict[str, Any]) -> bool:
    return is_completion_event(info, default_eval_semantics())


def death_location_histogram(death_x_positions: list[int], bin_size: int = 100) -> dict[str, int]:
    bins: dict[str, int] = {}
    for x_pos in death_x_positions:
        start = (int(x_pos) // bin_size) * bin_size
        key = f"{start}-{start + bin_size - 1}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def episode_start_state(episode: dict[str, Any]) -> str | None:
    state = episode.get("start_state") or episode.get("state")
    final_info = episode.get("final_info")
    if not state and isinstance(final_info, dict):
        state = final_info.get("start_state") or final_info.get("state")
    return str(state) if state else None


def serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    result = dict(info)
    result.pop("terminal_observation", None)
    return result


def eval_done_from_metrics(
    episode_results: list[dict[str, Any]],
    *,
    semantics: EvalSemantics | None = None,
) -> dict[str, int | float]:
    semantics = semantics or default_eval_semantics()
    metrics: dict[str, int | float] = {}
    completion_rates: list[float] = []
    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        state_episodes = [
            episode for episode in episode_results if episode_start_state(episode) == state
        ]
        denominator = len(state_episodes)
        completion_count = sum(1 for episode in state_episodes if episode.get("level_complete"))
        terminated_count = sum(
            1
            for episode in state_episodes
            if episode.get("terminated") and not episode.get("truncated")
        )
        max_steps_count = sum(1 for episode in state_episodes if episode.get("truncated"))
        unclassified_count = sum(
            1
            for episode in state_episodes
            if not episode.get("level_complete") and not episode.get("truncated")
        )

        all_metric = eval_done_value_metric("all", "from", state)
        max_steps_metric = eval_done_value_metric("max_steps", "from", state)
        terminated_metric = eval_done_value_metric("terminated", "from", state)
        unclassified_metric = eval_done_value_metric("unclassified", "from", state)
        metrics.update(
            {
                all_metric: denominator,
                max_steps_metric: max_steps_count,
                f"{max_steps_metric}/rate": max_steps_count / denominator,
                terminated_metric: terminated_count,
                f"{terminated_metric}/rate": terminated_count / denominator,
                unclassified_metric: unclassified_count,
                f"{unclassified_metric}/rate": unclassified_count / denominator,
            },
        )
        if semantics.completion_reason:
            completion_rate = completion_count / denominator
            completion_rates.append(completion_rate)
            completion_metric = eval_done_value_metric(
                semantics.completion_reason,
                "from",
                state,
            )
            metrics.update(
                {
                    completion_metric: completion_count,
                    f"{completion_metric}/rate": completion_rate,
                }
            )
    if completion_rates and semantics.completion_reason == "level_change":
        completion_rate_min = min(completion_rates)
        completion_rate_mean = float(np.mean(completion_rates))
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN] = completion_rate_min
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN] = completion_rate_mean
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = completion_rate_min
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST] = completion_rate_mean
    return metrics


def flat_numeric_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, int | float]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(prefix) and isinstance(value, int | float) and not isinstance(value, bool)
    }


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
            values.append(float(bool(result.get("level_complete"))))
        elif item == "progress":
            values.append(primary_progress_value(result, semantics))
        elif item == "reward":
            values.append(float(result.get("reward", 0.0) or 0.0))
        else:
            values.append(float(result.get(item, 0.0) or 0.0))
    return tuple(values or [float(result.get("reward", 0.0) or 0.0)])


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
) -> dict[str, Any]:
    if not episode_results:
        raise ValueError("episode_results must not be empty")
    semantics = semantics or default_eval_semantics()

    rewards = np.array([episode["reward"] for episode in episode_results], dtype=np.float64)
    progress_metrics: dict[str, int | float] = {}
    for field in semantics.progress_fields:
        values = np.array(
            [episode.get(field.result_key, 0) for episode in episode_results],
            dtype=np.float64,
        )
        mean_key, max_key = progress_summary_fields(field.result_key)
        progress_metrics[mean_key] = float(values.mean())
        progress_metrics[max_key] = int(values.max())
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode.get("level_complete"))
    death_count = sum(1 for episode in episode_results if episode.get("died"))
    terminated_count = sum(
        1
        for episode in episode_results
        if episode.get("terminated") and not episode.get("truncated")
    )
    truncated_count = sum(1 for episode in episode_results if episode.get("truncated"))
    unclassified_count = sum(
        1
        for episode in episode_results
        if not episode.get("level_complete") and not episode.get("truncated")
    )
    episode_count = len(episode_results)
    metrics: dict[str, Any] = {
        "episodes": episode_count,
        "deterministic": deterministic,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_max": float(rewards.max()),
        "terminated_count": terminated_count,
        "terminated_rate": terminated_count / episode_count,
        "truncated_count": truncated_count,
        "truncated_rate": truncated_count / episode_count,
        "unclassified_count": unclassified_count,
        "unclassified_rate": unclassified_count / episode_count,
        EVAL_DONE_ALL: episode_count,
        EVAL_DONE_MAX_STEPS: truncated_count,
        EVAL_DONE_MAX_STEPS_RATE: truncated_count / episode_count,
        EVAL_DONE_TERMINATED: terminated_count,
        EVAL_DONE_TERMINATED_RATE: terminated_count / episode_count,
        EVAL_DONE_UNCLASSIFIED: unclassified_count,
        EVAL_DONE_UNCLASSIFIED_RATE: unclassified_count / episode_count,
        "episode_results": episode_results,
    }
    metrics.update(progress_metrics)
    if semantics.completion_reason == "level_change":
        metrics.update(
            {
                "completion_count": completion_count,
                "completion_rate": completion_count / episode_count,
                EVAL_DONE_LEVEL_CHANGE: completion_count,
                EVAL_DONE_LEVEL_CHANGE_RATE: completion_count / episode_count,
            }
        )
    if semantics.death_flag_key:
        metrics.update(
            {
                "death_count": death_count,
                "death_rate": death_count / episode_count,
                "death_x_histogram": death_location_histogram(death_x_positions),
            }
        )
    metrics.update(eval_done_from_metrics(episode_results, semantics=semantics))
    if extra:
        metrics = {**extra, **metrics}
    return metrics


def metric_float(metrics: dict[str, Any] | Any, key: str, default: float = float("-inf")) -> float:
    value = metrics.get(key) if hasattr(metrics, "get") else None
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def completion_score(metrics: dict[str, Any]) -> tuple[float, float] | None:
    completion_min = metric_float(metrics, EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN)
    if completion_min == float("-inf"):
        completion_min = metric_float(metrics, EVAL_DONE_LEVEL_CHANGE_RATE)
    if completion_min == float("-inf"):
        completion_min = metric_float(metrics, "completion_rate")
    completion_mean = metric_float(
        metrics,
        EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
        metric_float(
            metrics,
            EVAL_DONE_LEVEL_CHANGE_RATE,
            metric_float(metrics, "completion_rate"),
        ),
    )
    if completion_min == float("-inf"):
        return None
    return (float(completion_min), float(completion_mean))


def eval_selection_objective_name(metrics: dict[str, Any]) -> str:
    if completion_score(metrics) is not None:
        return EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN
    return "eval/reward/mean"


def eval_selection_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    completion = completion_score(metrics)
    reward_mean = metric_float(metrics, "reward_mean")
    reward_max = metric_float(metrics, "reward_max")
    checkpoint_step = metric_float(metrics, "checkpoint_step")
    if completion is None:
        return (reward_mean, reward_max, -checkpoint_step, reward_mean)
    completion_min, completion_mean = completion
    steps_to_goal = (
        checkpoint_step
        if completion_min >= COMPLETION_GOAL_RATE and checkpoint_step > float("-inf")
        else float("inf")
    )
    return (completion_min, completion_mean, -steps_to_goal, reward_mean)


def run_eval_episode(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    seed: int,
    capture_actions: bool = False,
    default_start_state: str | None = None,
    semantics: EvalSemantics | None = None,
) -> dict[str, Any]:
    semantics = semantics or default_eval_semantics()
    env.seed(seed)
    obs = env.reset()
    actions: list[Any] = []
    total_reward = 0.0
    progress_values = {field.result_key: 0 for field in semantics.progress_fields}
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    completed = False
    died = False
    death_x_pos: Any | None = None
    start_state = default_start_state
    steps_taken = 0

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_value = single_env_action(action)
        if capture_actions:
            actions.append(action_value)
        obs, rewards, dones, infos = env.step(action)
        info = dict(infos[0])
        steps_taken = step_idx + 1
        terminated = bool(dones[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        total_reward += float(rewards[0])
        if start_state is None:
            start_state = info.get("start_state") or info.get("state")
        for field in semantics.progress_fields:
            progress_values[field.result_key] = max(
                progress_values[field.result_key],
                int(info.get(field.info_key, 0)),
            )
        final_info = info
        completed = completed or is_completion_event(final_info, semantics)
        if semantics.death_flag_key and bool(info.get(semantics.death_flag_key, False)):
            died = True
            if death_x_pos is None:
                death_x_pos = (
                    info.get(semantics.death_position_key)
                    if semantics.death_position_key
                    else None
                )
                if death_x_pos is None:
                    death_x_pos = progress_values.get("max_x_pos", 0)
        if terminated:
            break
    else:
        truncated = True

    result = {
        "start_state": start_state
        or final_info.get("start_state")
        or final_info.get("state")
        or default_start_state,
        "reward": total_reward,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "time": int(final_info.get("time", 0)),
        "steps": steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "final_info": serializable_info(final_info),
        "actions": actions,
    }
    if semantics.completion_reason:
        result["level_complete"] = completed
    for field in semantics.progress_fields:
        result[field.result_key] = progress_values[field.result_key]
    if semantics.death_flag_key:
        result["died"] = died
        result["death_x_pos"] = int(death_x_pos) if death_x_pos is not None else None
    return result
