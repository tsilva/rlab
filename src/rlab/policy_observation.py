from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rlab.env import info_value_from_state_name, task_conditioning, task_conditioning_info_values


def task_info_vars(config: Any) -> tuple[str, ...]:
    conditioning = task_conditioning(config)
    signal_name = conditioning.get("signal")
    signals = config.task.get("signals", {})
    source = signals.get(signal_name) if isinstance(signals, Mapping) else None
    return (source,) if isinstance(source, str) else tuple(source or ())


def task_state_names(config: Any) -> tuple[str, ...]:
    if task_info_vars(config):
        return tuple(
            ",".join(str(part) for part in value) for value in task_conditioning_info_values(config)
        )
    states = tuple(dict.fromkeys(config.states or ((config.state,) if config.state else ())))
    if not states:
        raise ValueError("task-conditioned playback requires at least one configured state")
    return states


def task_vector_for_state(
    config: Any,
    task_dim: int,
    active_state: str | None = None,
    active_info_value: tuple[int | str, ...] | None = None,
) -> np.ndarray:
    info_vars = task_info_vars(config)
    if info_vars:
        values = task_conditioning_info_values(config)
        if not values:
            raise ValueError("info-var task-conditioned playback has no configured task values")
        if task_dim != len(values):
            raise ValueError(
                "task-conditioned model expects "
                f"{task_dim} task values, but playback metadata has {len(values)} "
                f"info-value rows: {values}"
            )
        active_info_value = active_info_value or info_value_from_state_name(
            active_state or config.state,
            info_vars,
        )
        if active_info_value not in values:
            raise ValueError(
                f"playback info value {active_info_value!r} is not in "
                f"task-conditioned info values {values!r}",
            )
        task = np.zeros((1, task_dim), dtype=np.float32)
        task[0, values.index(active_info_value)] = 1.0
        return task

    states = task_state_names(config)
    if task_dim != len(states):
        raise ValueError(
            "task-conditioned model expects "
            f"{task_dim} task values, but playback metadata has {len(states)} state(s): {states}",
        )
    active_state = active_state or config.state or states[0]
    if active_state not in states:
        raise ValueError(
            f"playback state {active_state!r} is not in task-conditioned state list {states!r}",
        )
    task = np.zeros((1, task_dim), dtype=np.float32)
    task[0, states.index(active_state)] = 1.0
    return task


def model_observation(
    model: Any,
    image_obs: np.ndarray,
    config: Any,
    *,
    active_task_state: str | None = None,
    active_info_value: tuple[int | str, ...] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    observation_space = model.observation_space
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict):
        return image_obs
    if "image" not in spaces or "task" not in spaces:
        raise ValueError(
            "dict-observation model must have 'image' and 'task' observation keys, "
            f"got {tuple(spaces)}",
        )
    task_shape = getattr(spaces["task"], "shape", None)
    if task_shape is None or len(task_shape) != 1:
        raise ValueError(f"expected one-dimensional task observation, got {spaces['task']!r}")
    return {
        "image": image_obs,
        "task": task_vector_for_state(
            config,
            task_dim=int(task_shape[0]),
            active_state=active_task_state,
            active_info_value=active_info_value,
        ),
    }


def task_info_value_from_info(
    info: Mapping[str, Any], config: Any
) -> tuple[int | str, ...] | None:
    info_vars = task_info_vars(config)
    if not info_vars:
        return None
    try:
        value = tuple(info[var] for var in info_vars)
    except KeyError:
        return None
    return value if value in task_conditioning_info_values(config) else None
