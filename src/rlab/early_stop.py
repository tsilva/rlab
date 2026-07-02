from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any


EARLY_STOP_OPERATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
}
EARLY_STOP_RULE_KEYS = frozenset({"metric", "operator", "threshold"})


def _label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_non_empty_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    return value.strip()


def _require_finite_number(document: Mapping[str, Any], key: str, *, label: str) -> float:
    value = document.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{_label_path(label, key)} must be finite")
    return value


def normalize_early_stop_rule(value: Any, *, label: str) -> dict[str, Any]:
    node = _require_mapping(value, label=label)
    extra_keys = sorted(set(node) - EARLY_STOP_RULE_KEYS)
    if extra_keys:
        raise ValueError(f"{label} has unexpected keys: {extra_keys}")
    metric = _require_non_empty_string(node, "metric", label=label)
    operator = _require_non_empty_string(node, "operator", label=label)
    if operator not in EARLY_STOP_OPERATORS:
        allowed = ", ".join(sorted(EARLY_STOP_OPERATORS))
        raise ValueError(f"{_label_path(label, 'operator')} must be one of {allowed}")
    threshold = _require_finite_number(node, "threshold", label=label)
    return {"metric": metric, "operator": operator, "threshold": threshold}


def normalize_early_stop_config(value: Any, *, label: str = "early_stop") -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a non-empty list")
    if not value:
        raise ValueError(f"{label} must be a non-empty list")
    return [
        normalize_early_stop_rule(rule, label=f"{label}[{index}]")
        for index, rule in enumerate(value)
    ]


def flat_metric_rule_from_early_stop(value: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if len(value) != 1:
        return None
    rule = value[0]
    return deepcopy(dict(rule)) if isinstance(rule, Mapping) else None


def evaluate_early_stop_config(
    config: Sequence[Mapping[str, Any]],
    value_lookup: Callable[[str], float | None],
) -> tuple[bool | None, dict[str, float]]:
    values: dict[str, float] = {}

    pending = False
    for rule in config:
        metric = str(rule["metric"])
        value = value_lookup(metric)
        if value is None:
            pending = True
            continue
        values[metric] = value
        operator = str(rule["operator"])
        threshold = float(rule["threshold"])
        if not EARLY_STOP_OPERATORS[operator](value, threshold):
            return False, values
    return (None if pending else True), values
