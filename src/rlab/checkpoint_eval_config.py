from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from rlab.early_stop import normalize_early_stop_config


def checkpoint_eval_max_steps(config: Mapping[str, Any]) -> int:
    explicit = int(config.get("post_train_eval_max_steps") or 0)
    if explicit > 0:
        return explicit
    environment = config.get("checkpoint_eval_environment")
    task = environment.get("task") if isinstance(environment, Mapping) else None
    termination = task.get("termination") if isinstance(task, Mapping) else None
    fallback = int(
        termination.get("max_episode_steps")
        if isinstance(termination, Mapping) and termination.get("max_episode_steps") is not None
        else 0
    )
    if fallback > 0:
        return fallback
    raise ValueError("checkpoint eval max steps are not materialized")


def normalize_checkpoint_eval_stages(
    value: Any,
    *,
    label: str = "checkpoint_eval_stages",
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a non-empty list")
    if not value:
        raise ValueError(f"{label} must be a non-empty list")

    stages: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_stage in enumerate(value):
        stage_label = f"{label}[{index}]"
        if not isinstance(raw_stage, Mapping):
            raise ValueError(f"{stage_label} must be an object")
        extra = sorted(set(raw_stage) - {"name", "episodes", "n_envs", "pass", "candidate_stop"})
        if extra:
            raise ValueError(f"{stage_label} has unexpected keys: {extra}")
        name = raw_stage.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{stage_label}.name must be a non-empty string")
        name = name.strip()
        if name not in {"screen", "confirm"}:
            raise ValueError(f"{stage_label}.name must be 'screen' or 'confirm'")
        if name in names:
            raise ValueError(f"{stage_label}.name must be unique")
        names.add(name)

        episodes = raw_stage.get("episodes")
        if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
            raise ValueError(f"{stage_label}.episodes must be an integer >= 1")
        n_envs = raw_stage.get("n_envs")
        if n_envs is not None and (
            not isinstance(n_envs, int) or isinstance(n_envs, bool) or n_envs < 1
        ):
            raise ValueError(f"{stage_label}.n_envs must be an integer >= 1")
        pass_rules = normalize_early_stop_config(raw_stage.get("pass"), label=f"{stage_label}.pass")
        candidate_stop = raw_stage.get("candidate_stop", False)
        if not isinstance(candidate_stop, bool):
            raise ValueError(f"{stage_label}.candidate_stop must be a boolean")
        stages.append(
            {
                "name": name,
                "episodes": int(episodes),
                "n_envs": None if n_envs is None else int(n_envs),
                "pass": deepcopy(pass_rules),
                "candidate_stop": candidate_stop,
            }
        )
    return stages
