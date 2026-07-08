from __future__ import annotations

from copy import deepcopy
from typing import Any

from rlab.wrapper_specs import normalize_wrapper_spec_sequence


VEC_WRAPPER_SPEC_ID_KEYS = ("id", "wrapper", "class", "name", "type")
VEC_WRAPPER_CONTROL_KEYS = frozenset(VEC_WRAPPER_SPEC_ID_KEYS)

GYM_VECTOR_TO_SB3 = "gym_vector_to_sb3"
DISCRETE_ACTIONS = "discrete_actions"
OBSERVATION_MASK = "observation_mask"
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
        OBSERVATION_MASK,
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
    {"id": OBSERVATION_MASK},
    {"id": RETRO_PROGRESS_INFO},
    {"id": VEC_MONITOR},
    {"id": TASK_CONDITIONING},
    {"id": TRANSPOSE_IMAGE},
)


def normalize_vec_wrapper_specs(
    value: Any,
    *,
    label: str = "vec_wrappers",
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for index, spec in enumerate(
        normalize_wrapper_spec_sequence(
            value,
            label=label,
            id_keys=VEC_WRAPPER_SPEC_ID_KEYS,
            item_kind="vector wrapper",
        )
    ):
        item_label = f"{label}[{index}]"
        wrapper_id = spec["id"]
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
