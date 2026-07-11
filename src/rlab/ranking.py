from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    EVAL_DONE_LEVEL_CHANGE_RATE,
    EVAL_REWARD_MEAN,
    LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL,
)


DEFAULT_COMPLETION_RANK = (
    f"max({EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN})",
    f"max({EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN})",
    f"min({LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL})",
    f"max({EVAL_REWARD_MEAN})",
)
DEFAULT_REWARD_RANK = (
    f"max({EVAL_REWARD_MEAN})",
    f"max({EVAL_BEST_REWARD})",
    f"min({LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL})",
)
_RANK_RE = re.compile(r"^(max|min)\(([^()]+)\)$")


@dataclass(frozen=True)
class RankCriterion:
    direction: str
    metric: str


def parse_objective_rank(value: Any) -> tuple[RankCriterion, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    criteria: list[RankCriterion] = []
    for item in value:
        match = _RANK_RE.fullmatch(str(item).strip())
        if match is None:
            return ()
        criteria.append(RankCriterion(match.group(1), match.group(2).strip()))
    return tuple(criteria)


def objective_rank_strings(criteria: Sequence[RankCriterion]) -> tuple[str, ...]:
    return tuple(f"{criterion.direction}({criterion.metric})" for criterion in criteria)


def default_objective_rank(metrics: Mapping[str, Any]) -> tuple[RankCriterion, ...]:
    has_completion = any(
        _metric_value(metrics, metric) is not None
        for metric in (
            EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
            EVAL_DONE_LEVEL_CHANGE_RATE,
            "completion_rate",
        )
    )
    return parse_objective_rank(DEFAULT_COMPLETION_RANK if has_completion else DEFAULT_REWARD_RANK)


def _nested_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for segment in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(segment)
    return value


def _metric_value(metrics: Mapping[str, Any], metric: str) -> Any:
    value = metrics.get(metric)
    if value is not None:
        return value
    aliases = {
        EVAL_REWARD_MEAN: "reward_mean",
        EVAL_BEST_REWARD: "best_episode.reward",
        EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN: "completion_rate",
        EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN: "completion_rate",
        LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL: "checkpoint_step",
    }
    alias = aliases.get(metric)
    value = _nested_value(metrics, alias) if alias and "." in alias else metrics.get(alias or "")
    if value is None and metric == EVAL_BEST_REWARD:
        return metrics.get("reward_max")
    return value


def rank_metric_values(
    metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]
) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for criterion in criteria:
        value = _metric_value(metrics, criterion.metric)
        if criterion.metric == LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL:
            completion = _metric_value(metrics, EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN)
            if completion is not None and float(completion) < 0.99:
                value = None
        try:
            values.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            values.append(None)
    return tuple(values)


def rank_score(
    metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]
) -> tuple[float, ...]:
    score: list[float] = []
    for criterion, value in zip(criteria, rank_metric_values(metrics, criteria), strict=True):
        if value is None:
            score.append(float("-inf"))
        else:
            score.append(value if criterion.direction == "max" else -value)
    return tuple(score)
