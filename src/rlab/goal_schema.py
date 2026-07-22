from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GOAL_FIELDS = frozenset(
    {"eval", "goal_id", "objective", "release", "reward_shapes", "tags", "title", "train"}
)
GOAL_OBJECTIVE_FIELDS = frozenset({"rank", "states"})
GOAL_TRAIN_FIELDS = frozenset(
    {
        "checkpoint_eval_backend",
        "checkpoint_freq",
        "environment",
        "policy",
        "stop_on_acceptance",
    }
)
GOAL_EVAL_FIELDS = frozenset({"acceptance", "env_config", "environment", "episodes", "policy"})


def _reject_unknown_fields(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        return
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def validate_goal_document_shape(document: Mapping[str, Any], *, label: str) -> None:
    """Reject misspelled goal-owned fields without importing runtime orchestration."""

    _reject_unknown_fields(document, allowed=GOAL_FIELDS, label=label)
    _reject_unknown_fields(
        document.get("objective"),
        allowed=GOAL_OBJECTIVE_FIELDS,
        label=f"{label}.objective",
    )
    _reject_unknown_fields(
        document.get("train"),
        allowed=GOAL_TRAIN_FIELDS,
        label=f"{label}.train",
    )
    _reject_unknown_fields(
        document.get("eval"),
        allowed=GOAL_EVAL_FIELDS,
        label=f"{label}.eval",
    )
