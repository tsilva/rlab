from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


VEC_WRAPPER_SPEC_ID_KEYS = ("id", "wrapper", "class", "name", "type")
VEC_WRAPPER_CONTROL_KEYS = frozenset(VEC_WRAPPER_SPEC_ID_KEYS)

GYM_VECTOR_TO_SB3 = "gym_vector_to_sb3"
DISCRETE_ACTIONS = "discrete_actions"
ALE_MASKED_PREPROCESS = "ale_masked_preprocess"
RETRO_PROGRESS_INFO = "retro_progress_info"
VEC_MONITOR = "vec_monitor"
TASK_CONDITIONING = "task_conditioning"
TRANSPOSE_IMAGE = "transpose_image"
SB3_FUSED = "sb3_fused"

KNOWN_VEC_WRAPPER_IDS = frozenset(
    {
        GYM_VECTOR_TO_SB3,
        DISCRETE_ACTIONS,
        ALE_MASKED_PREPROCESS,
        RETRO_PROGRESS_INFO,
        VEC_MONITOR,
        TASK_CONDITIONING,
        TRANSPOSE_IMAGE,
        SB3_FUSED,
    }
)

DEFAULT_VEC_WRAPPER_SPECS: tuple[dict[str, Any], ...] = (
    {"id": GYM_VECTOR_TO_SB3},
    {"id": DISCRETE_ACTIONS},
    {"id": ALE_MASKED_PREPROCESS},
    {"id": RETRO_PROGRESS_INFO},
    {"id": VEC_MONITOR},
    {"id": TASK_CONDITIONING},
    {"id": TRANSPOSE_IMAGE},
)


def _vec_wrapper_id(spec: Mapping[str, Any], *, label: str) -> str:
    for key in VEC_WRAPPER_SPEC_ID_KEYS:
        value = spec.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
            return value.strip()
    raise ValueError(f"{label} must include one of: {', '.join(VEC_WRAPPER_SPEC_ID_KEYS)}")


def normalize_vec_wrapper_specs(
    value: Any,
    *,
    label: str = "vec_wrappers",
) -> tuple[dict[str, Any], ...]:
    if value in (None, "", ()):
        return ()
    if isinstance(value, str):
        value = [{"id": value}]
    elif isinstance(value, Mapping):
        value = [value]
    elif not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list of vector wrapper specs")

    specs: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if isinstance(item, str):
            spec = {"id": item}
        elif isinstance(item, Mapping):
            spec = deepcopy(dict(item))
        else:
            raise ValueError(f"{item_label} must be an object or vector wrapper id string")
        wrapper_id = _vec_wrapper_id(spec, label=item_label)
        if wrapper_id not in KNOWN_VEC_WRAPPER_IDS:
            known = ", ".join(sorted(KNOWN_VEC_WRAPPER_IDS))
            raise ValueError(
                f"{item_label}.id is unknown: {wrapper_id!r}; known vector wrappers: {known}"
            )
        normalized = {
            key: deepcopy(value)
            for key, value in spec.items()
            if key not in VEC_WRAPPER_CONTROL_KEYS
        }
        normalized["id"] = wrapper_id
        specs.append(normalized)
    return tuple(specs)
