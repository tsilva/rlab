from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


REWARD_PROGRAM_KIND_MARIO_V1 = "mario-v1"
MARIO_REWARD_KERNEL_REVISION = "mario-kernel-v1"
REWARD_SHAPE_KEY_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")

MARIO_REWARD_FIELDS = (
    "reward_mode",
    "use_native_reward",
    "clip_rewards",
    "progress_reward_cap",
    "progress_reward_scale",
    "terminal_reward",
    "reward_scale",
    "time_penalty",
    "death_penalty",
    "completion_reward",
    "score_progress_clipped",
)
MARIO_REWARD_FIELD_SET = frozenset(MARIO_REWARD_FIELDS)
MARIO_REWARD_MODES = frozenset({"native", "bounded", "baseline", "score", "additive"})
MARIO_BOOL_FIELDS = frozenset({"use_native_reward", "clip_rewards", "score_progress_clipped"})
MARIO_NUMBER_FIELDS = frozenset(MARIO_REWARD_FIELD_SET - MARIO_BOOL_FIELDS - {"reward_mode"})

# Mario signal arithmetic is currently int64-backed and reward outputs are float32.
# Reserve headroom for several simultaneously active components before the output cast.
_FLOAT32_MAX = 3.4028234663852886e38
_MARIO_SAFE_COEFFICIENT_ABS_MAX = _FLOAT32_MAX / (8.0 * float(2**32))


@dataclass(frozen=True)
class RewardShapeSelection:
    goal: dict[str, Any]
    key: str
    program_kind: str
    program_revision: str
    semantic_sha256: str
    is_default: bool
    reward: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def normalize_mario_reward(
    value: Mapping[str, Any],
    *,
    label: str,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(str(key) for key in value if key not in MARIO_REWARD_FIELD_SET)
    if unknown:
        raise ValueError(f"{label} has unexpected field(s): {', '.join(unknown)}")
    if require_complete:
        missing = [key for key in MARIO_REWARD_FIELDS if key not in value]
        if missing:
            raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")

    normalized: dict[str, Any] = {}
    if "reward_mode" in value:
        mode = value["reward_mode"]
        if not isinstance(mode, str) or mode not in MARIO_REWARD_MODES:
            raise ValueError(
                f"{label}.reward_mode must be one of {sorted(MARIO_REWARD_MODES)}, got {mode!r}"
            )
        normalized["reward_mode"] = mode
    for key in MARIO_BOOL_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if type(item) is not bool:
            raise ValueError(f"{label}.{key} must be a boolean")
        normalized[key] = item
    for key in MARIO_NUMBER_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}.{key} must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} must be finite")
        if abs(number) > _MARIO_SAFE_COEFFICIENT_ABS_MAX:
            raise ValueError(f"{label}.{key} exceeds the Mario float32 reward safety bound")
        normalized[key] = _normalize_zero(number)
    return {key: normalized[key] for key in MARIO_REWARD_FIELDS if key in normalized}


def _mario_compiled_semantics(reward: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(reward["reward_mode"])
    semantics: dict[str, Any] = {
        "reward_mode": mode,
        "clip_rewards": bool(reward["clip_rewards"]),
        "time_penalty": float(reward["time_penalty"]),
    }
    if mode == "bounded":
        semantics.update(
            progress_reward_cap=float(reward["progress_reward_cap"]),
            terminal_reward=float(reward["terminal_reward"]),
            reward_scale=float(reward["reward_scale"]) or 1.0,
        )
    elif mode == "baseline":
        semantics.update(
            terminal_reward=float(reward["terminal_reward"]),
            reward_scale=float(reward["reward_scale"]) or 1.0,
        )
    elif mode in {"score", "additive"}:
        semantics.update(
            use_native_reward=bool(reward["use_native_reward"]),
            progress_reward_scale=float(reward["progress_reward_scale"]),
            completion_reward=float(reward["completion_reward"]),
            death_penalty=float(reward["death_penalty"]),
        )
        if mode == "score":
            clipped = bool(reward["score_progress_clipped"])
            semantics["score_progress_clipped"] = clipped
            if clipped:
                semantics["progress_reward_cap"] = float(reward["progress_reward_cap"])
    return semantics


def mario_reward_semantic_sha256(reward: Mapping[str, Any]) -> str:
    normalized = normalize_mario_reward(
        reward,
        label="Mario reward definition",
        require_complete=True,
    )
    return _sha256(
        {
            "task_id": "mario",
            "program_kind": REWARD_PROGRAM_KIND_MARIO_V1,
            "program_revision": MARIO_REWARD_KERNEL_REVISION,
            "compiled_semantics": _mario_compiled_semantics(normalized),
        }
    )


def _catalog(document: Mapping[str, Any], *, label: str) -> Mapping[str, Any] | None:
    value = document.get("reward_shapes")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.reward_shapes must be an object")
    unknown = sorted(set(value) - {"program_kind", "default", "definitions"})
    if unknown:
        raise ValueError(
            f"{label}.reward_shapes has unknown field(s): {', '.join(str(x) for x in unknown)}"
        )
    return value


def validate_reward_shape_catalog(
    document: Mapping[str, Any],
    *,
    label: str = "goal",
) -> None:
    catalog = _catalog(document, label=label)
    if catalog is None:
        return
    kind = catalog.get("program_kind")
    if kind != REWARD_PROGRAM_KIND_MARIO_V1:
        raise ValueError(f"{label}.reward_shapes.program_kind has no registered compiler: {kind!r}")
    default = catalog.get("default")
    if not isinstance(default, str) or not REWARD_SHAPE_KEY_PATTERN.fullmatch(default):
        raise ValueError(f"{label}.reward_shapes.default must be a lowercase kebab key")
    definitions = catalog.get("definitions")
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError(f"{label}.reward_shapes.definitions must be a non-empty object")
    if default not in definitions:
        raise ValueError(f"{label}.reward_shapes.default references unknown key {default!r}")

    seen_hashes: dict[str, str] = {}
    for raw_key, raw_reward in definitions.items():
        key = str(raw_key)
        if not REWARD_SHAPE_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"{label}.reward_shapes.definitions key {key!r} must be 1-64 lowercase kebab characters"
            )
        reward = normalize_mario_reward(
            raw_reward,
            label=f"{label}.reward_shapes.definitions.{key}",
            require_complete=True,
        )
        semantic_hash = mario_reward_semantic_sha256(reward)
        previous = seen_hashes.get(semantic_hash)
        if previous is not None:
            raise ValueError(
                f"{label}.reward_shapes definitions {previous!r} and {key!r} have identical executable semantics"
            )
        seen_hashes[semantic_hash] = key

    for phase in ("train", "eval"):
        section = document.get(phase)
        environment = section.get("environment") if isinstance(section, Mapping) else None
        task = environment.get("task") if isinstance(environment, Mapping) else None
        if not isinstance(task, Mapping) or task.get("id") != "mario":
            raise ValueError(
                f"{label}.reward_shapes program {kind!r} requires {phase}.environment.task.id='mario'"
            )
        if "reward" in task:
            raise ValueError(
                f"{label}.{phase}.environment.task.reward must be omitted when reward_shapes is declared"
            )
    _validate_catalog_phase_semantics(document, label=label)


def _phase_semantic_projection(environment: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(dict(environment))
    env_config = projection.get("env_config")
    if isinstance(env_config, dict):
        for key in ("n_envs", "seed", "max_steps"):
            env_config.pop(key, None)
    task = projection.get("task")
    if isinstance(task, dict):
        task.pop("reward", None)
        termination = task.get("termination")
        if isinstance(termination, dict):
            for key in ("failure", "success", "timeout", "neutral", "max_episode_steps"):
                termination.pop(key, None)
    return projection


def _validate_catalog_phase_semantics(
    document: Mapping[str, Any],
    *,
    label: str,
) -> None:
    train = document.get("train")
    evaluation = document.get("eval")
    train_environment = train.get("environment") if isinstance(train, Mapping) else None
    eval_environment = evaluation.get("environment") if isinstance(evaluation, Mapping) else None
    if not isinstance(train_environment, Mapping) or not isinstance(eval_environment, Mapping):
        return
    if _phase_semantic_projection(train_environment) != _phase_semantic_projection(
        eval_environment
    ):
        raise ValueError(
            f"{label} catalog-backed train/eval environments may differ only in declared "
            "termination policy and execution-only evaluation settings"
        )


def select_goal_reward_shape(
    document: Mapping[str, Any],
    selector: str | None,
    *,
    label: str = "goal",
) -> RewardShapeSelection | None:
    catalog = _catalog(document, label=label)
    if catalog is None:
        if selector is not None:
            raise ValueError(f"{label} does not define reward_shapes; reward_shape is unsupported")
        return None
    validate_reward_shape_catalog(document, label=label)
    default = str(catalog["default"])
    key = str(selector).strip() if selector is not None else default
    if not REWARD_SHAPE_KEY_PATTERN.fullmatch(key):
        raise ValueError("reward_shape must be a 1-64 character lowercase kebab key")
    definitions = catalog["definitions"]
    if key not in definitions:
        available = ", ".join(sorted(str(item) for item in definitions))
        raise ValueError(f"unknown reward_shape {key!r}; available: {available}")
    reward = normalize_mario_reward(
        definitions[key],
        label=f"{label}.reward_shapes.definitions.{key}",
        require_complete=True,
    )
    effective = copy.deepcopy(dict(document))
    effective.pop("reward_shapes", None)
    for phase in ("train", "eval"):
        task = effective[phase]["environment"]["task"]
        task["reward"] = copy.deepcopy(reward)
    return RewardShapeSelection(
        goal=effective,
        key=key,
        program_kind=str(catalog["program_kind"]),
        program_revision=MARIO_REWARD_KERNEL_REVISION,
        semantic_sha256=mario_reward_semantic_sha256(reward),
        is_default=key == default,
        reward=reward,
    )


def goal_for_contract_validation(
    document: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    selected = select_goal_reward_shape(document, None, label=label)
    return selected.goal if selected is not None else copy.deepcopy(dict(document))
