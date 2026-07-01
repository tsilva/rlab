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
    EVAL_DONE_UNCLASSIFIED,
    EVAL_DONE_UNCLASSIFIED_RATE,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
    eval_done_value_metric,
)


def is_level_complete(info: dict[str, Any]) -> bool:
    if "completion_event" in info or "level_complete" in info:
        return bool(info.get("completion_event", info.get("level_complete", False)))
    return bool(info.get("level_changed", False)) and not bool(
        info.get("died", False) or info.get("life_loss", False),
    )


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


def eval_done_from_metrics(episode_results: list[dict[str, Any]]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    level_change_rates: list[float] = []
    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        state_episodes = [
            episode for episode in episode_results if episode_start_state(episode) == state
        ]
        denominator = len(state_episodes)
        level_change_count = sum(1 for episode in state_episodes if episode["level_complete"])
        max_steps_count = sum(1 for episode in state_episodes if episode.get("truncated"))
        unclassified_count = sum(
            1
            for episode in state_episodes
            if not episode["level_complete"] and not episode.get("truncated")
        )
        level_change_rate = level_change_count / denominator
        level_change_rates.append(level_change_rate)

        all_metric = eval_done_value_metric("all", "from", state)
        level_change_metric = eval_done_value_metric("level_change", "from", state)
        max_steps_metric = eval_done_value_metric("max_steps", "from", state)
        unclassified_metric = eval_done_value_metric("unclassified", "from", state)
        metrics.update(
            {
                all_metric: denominator,
                level_change_metric: level_change_count,
                f"{level_change_metric}/rate": level_change_rate,
                max_steps_metric: max_steps_count,
                f"{max_steps_metric}/rate": max_steps_count / denominator,
                unclassified_metric: unclassified_count,
                f"{unclassified_metric}/rate": unclassified_count / denominator,
            },
        )
    if level_change_rates:
        level_change_rate_min = min(level_change_rates)
        level_change_rate_mean = float(np.mean(level_change_rates))
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN] = level_change_rate_min
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN] = level_change_rate_mean
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = level_change_rate_min
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST] = level_change_rate_mean
    return metrics


def flat_numeric_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, int | float]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(prefix) and isinstance(value, int | float) and not isinstance(value, bool)
    }


def episode_rank(result: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(bool(result["level_complete"])),
        float(result["max_x_pos"]),
        float(result["reward"]),
    )


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
) -> dict[str, Any]:
    if not episode_results:
        raise ValueError("episode_results must not be empty")

    rewards = np.array([episode["reward"] for episode in episode_results], dtype=np.float64)
    max_x_positions = np.array(
        [episode["max_x_pos"] for episode in episode_results],
        dtype=np.float64,
    )
    max_level_x_positions = np.array(
        [episode["max_level_x_pos"] for episode in episode_results],
        dtype=np.float64,
    )
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode["level_complete"])
    death_count = sum(1 for episode in episode_results if episode["died"])
    terminated_count = sum(
        1
        for episode in episode_results
        if episode.get("terminated") and not episode.get("truncated")
    )
    truncated_count = sum(1 for episode in episode_results if episode.get("truncated"))
    unclassified_count = sum(
        1
        for episode in episode_results
        if not episode["level_complete"] and not episode.get("truncated")
    )
    episode_count = len(episode_results)
    metrics: dict[str, Any] = {
        "episodes": episode_count,
        "deterministic": deterministic,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_max": float(rewards.max()),
        "max_x_mean": float(max_x_positions.mean()),
        "max_x_max": int(max_x_positions.max()),
        "max_level_x_mean": float(max_level_x_positions.mean()),
        "max_level_x_max": int(max_level_x_positions.max()),
        "completion_count": completion_count,
        "completion_rate": completion_count / episode_count,
        "death_count": death_count,
        "death_rate": death_count / episode_count,
        "terminated_count": terminated_count,
        "terminated_rate": terminated_count / episode_count,
        "truncated_count": truncated_count,
        "truncated_rate": truncated_count / episode_count,
        "unclassified_count": unclassified_count,
        "unclassified_rate": unclassified_count / episode_count,
        EVAL_DONE_ALL: episode_count,
        EVAL_DONE_LEVEL_CHANGE: completion_count,
        EVAL_DONE_LEVEL_CHANGE_RATE: completion_count / episode_count,
        EVAL_DONE_MAX_STEPS: truncated_count,
        EVAL_DONE_MAX_STEPS_RATE: truncated_count / episode_count,
        EVAL_DONE_UNCLASSIFIED: unclassified_count,
        EVAL_DONE_UNCLASSIFIED_RATE: unclassified_count / episode_count,
        "death_x_histogram": death_location_histogram(death_x_positions),
        "episode_results": episode_results,
    }
    metrics.update(eval_done_from_metrics(episode_results))
    if extra:
        metrics = {**extra, **metrics}
    return metrics


def run_eval_episode(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    seed: int,
    capture_actions: bool = False,
    default_start_state: str | None = None,
) -> dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    actions: list[Any] = []
    total_reward = 0.0
    max_x_pos = 0
    max_level_x_pos = 0
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
        max_x_pos = max(max_x_pos, int(info.get("max_x_pos", 0)))
        max_level_x_pos = max(max_level_x_pos, int(info.get("level_max_x_pos", 0)))
        final_info = info
        completed = completed or is_level_complete(final_info)
        if bool(info.get("died", False)):
            died = True
            if death_x_pos is None:
                death_x_pos = info.get("death_x_pos")
                if death_x_pos is None:
                    death_x_pos = max_x_pos
        if terminated:
            break
    else:
        truncated = True

    return {
        "start_state": start_state
        or final_info.get("start_state")
        or final_info.get("state")
        or default_start_state,
        "reward": total_reward,
        "max_x_pos": max_x_pos,
        "max_level_x_pos": max_level_x_pos,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "time": int(final_info.get("time", 0)),
        "steps": steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "level_complete": completed,
        "died": died,
        "death_x_pos": int(death_x_pos) if death_x_pos is not None else None,
        "final_info": serializable_info(final_info),
        "actions": actions,
    }
