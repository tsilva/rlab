from __future__ import annotations

from typing import Any


TRAIN_SEED_START = 0
TRAIN_SEED_MIN = TRAIN_SEED_START
EVAL_SEED_START = 10_000
TRAIN_SEED_MAX = EVAL_SEED_START - 1
DEFAULT_TRAIN_SEED = 123
DEFAULT_EVAL_SEED = EVAL_SEED_START


def _require_seed_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer seed")
    return value


def _seed_span(value: Any) -> int:
    try:
        span = int(value)
    except (TypeError, ValueError):
        return 1
    return max(span, 1)


def validate_training_seed(value: Any, *, label: str = "seed", seed_span: Any = 1) -> int:
    seed = _require_seed_int(value, label=label)
    span = _seed_span(seed_span)
    last_seed = seed + span - 1
    if seed < TRAIN_SEED_MIN or last_seed > TRAIN_SEED_MAX:
        if span == 1:
            raise ValueError(
                f"{label} must be in the training seed range "
                f"{TRAIN_SEED_MIN}..{TRAIN_SEED_MAX}; seeds >= {EVAL_SEED_START} "
                "are reserved for eval"
            )
        raise ValueError(
            f"{label} plus {span} training env slot(s) must stay in the "
            f"training seed range {TRAIN_SEED_MIN}..{TRAIN_SEED_MAX}; "
            f"seeds >= {EVAL_SEED_START} are reserved for eval"
        )
    return seed


def validate_eval_seed(value: Any, *, label: str = "seed") -> int:
    seed = _require_seed_int(value, label=label)
    if seed < EVAL_SEED_START:
        raise ValueError(
            f"{label} must be in the eval/test seed range >= {EVAL_SEED_START}; "
            f"seeds {TRAIN_SEED_MIN}..{TRAIN_SEED_MAX} are reserved for training"
        )
    return seed
