from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlab.metric_names import (
    EVAL_FULL_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_GOAL,
    METRICS_SCHEMA_VERSION,
    validate_metric_name,
)


_RANK_RE = re.compile(r"^(max|min)\(([^()]+)\)$")

# Persisted queue records predate the current metrics registry. These aliases are
# accepted only by the historical read path; new goal and train configuration
# validation continues to use parse_objective_rank() and rejects them.
_HISTORICAL_RANK_METRICS = frozenset(
    {
        "eval/done/serve_stall/rate",
        "eval/done/level_change/from_rate/min",
        "eval/done/level_change/from_rate/mean",
        "eval/info/level_complete/rate/min",
        "eval/info/level_complete/rate/mean",
        "eval/full/info/level_complete/rate/min",
        "eval/full/info/level_complete/rate/mean",
        "eval/reward/mean",
        "eval/best/reward",
        "eval/full/reward/mean",
        "leader/checkpoint/steps_to_completion_goal",
    }
)


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
        metric = match.group(2).strip()
        try:
            validate_metric_name(metric)
        except ValueError:
            return ()
        criteria.append(RankCriterion(match.group(1), metric))
    return tuple(criteria)


def parse_persisted_objective_rank(value: Any) -> tuple[RankCriterion, ...]:
    """Parse current or explicitly known historical queue ranking criteria."""
    current = parse_objective_rank(value)
    if current:
        return current
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    criteria: list[RankCriterion] = []
    for item in value:
        match = _RANK_RE.fullmatch(str(item).strip())
        if match is None:
            return ()
        metric = match.group(2).strip()
        if metric not in _HISTORICAL_RANK_METRICS:
            return ()
        criteria.append(RankCriterion(match.group(1), metric))
    return tuple(criteria)


def require_objective_rank(value: Any) -> tuple[RankCriterion, ...]:
    criteria = parse_objective_rank(value)
    if not criteria:
        raise ValueError(
            f"objective.rank must contain valid schema-v{METRICS_SCHEMA_VERSION} metric criteria"
        )
    return criteria


def objective_rank_strings(criteria: Sequence[RankCriterion]) -> tuple[str, ...]:
    return tuple(f"{criterion.direction}({criterion.metric})" for criterion in criteria)


def _metric_value(metrics: Mapping[str, Any], metric: str) -> Any:
    if metric == LEADER_CHECKPOINT_STEP:
        return metrics.get(metric, metrics.get("checkpoint_step"))
    if metric != LEADER_CHECKPOINT_STEPS_TO_GOAL:
        return metrics.get(metric)
    success = metrics.get(EVAL_FULL_SUCCESS_RATE_MIN)
    if success is None or float(success) < 0.99:
        return None
    return metrics.get(metric, metrics.get("checkpoint_step"))


def rank_metric_values(
    metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]
) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for criterion in criteria:
        value = _metric_value(metrics, criterion.metric)
        try:
            values.append(float(value) if value is not None else None)
        except TypeError, ValueError:
            values.append(None)
    return tuple(values)


def rank_score(metrics: Mapping[str, Any], criteria: Sequence[RankCriterion]) -> tuple[float, ...]:
    score: list[float] = []
    for criterion, value in zip(criteria, rank_metric_values(metrics, criteria), strict=True):
        score.append(
            float("-inf") if value is None else value if criterion.direction == "max" else -value
        )
    return tuple(score)
