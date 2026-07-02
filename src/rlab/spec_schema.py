from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any

from rlab.seeds import TRAIN_SEED_MAX, TRAIN_SEED_MIN, validate_training_seed


TRAIN_SPEC_SCHEMA_VERSION = 1
TRAIN_SPEC_REQUIRED_FIELDS = (
    "goal",
    "slug",
    "hypothesis",
    "wandb_group",
    "wandb_tags",
    "run_description_template",
    "train_config",
)
TRAIN_SPEC_REQUIRED_TRAIN_CONFIG_FIELDS = (
    "game",
    "timesteps",
    "wandb",
    "wandb_mode",
)
TRAIN_SPEC_ALLOWED_TEMPLATE_FIELDS = frozenset({"seed", "slug", "utc"})


TRAIN_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://tsilva.dev/rlab/train-spec.schema.json",
    "title": "rlab queue-backed train spec",
    "type": "object",
    "additionalProperties": True,
    "required": list(TRAIN_SPEC_REQUIRED_FIELDS),
    "properties": {
        "schema_version": {"const": TRAIN_SPEC_SCHEMA_VERSION},
        "goal": {
            "type": "object",
            "additionalProperties": True,
            "required": ["goal_id"],
            "properties": {
                "goal_id": {"type": "string", "minLength": 1},
            },
        },
        "goal_slug": {"type": "string", "minLength": 1},
        "slug": {"type": "string", "minLength": 1},
        "hypothesis": {"type": "string", "minLength": 1},
        "parent_spec_slug": {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "null"},
            ],
        },
        "max_attempts": {"type": "integer", "minimum": 1},
        "seeds": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "integer", "minimum": TRAIN_SEED_MIN, "maximum": TRAIN_SEED_MAX},
        },
        "wandb_group": {"type": "string", "minLength": 1},
        "run_name_label": {"type": "string", "minLength": 1},
        "wandb_tags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "run_description_template": {"type": "string", "minLength": 1},
        "selection_metrics": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "selection_policy": {
            "type": "object",
            "additionalProperties": True,
        },
        "train_config": {
            "type": "object",
            "additionalProperties": True,
            "required": list(TRAIN_SPEC_REQUIRED_TRAIN_CONFIG_FIELDS),
            "properties": {
                "game": {"type": "string", "minLength": 1},
                "state": {"type": "string", "minLength": 1},
                "states": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "timesteps": {"type": "integer", "minimum": 1},
                "wandb": {"type": "boolean"},
                "wandb_mode": {"enum": ["online", "offline", "disabled"]},
            },
        },
    },
}


def _label_path(label: str, key: str) -> str:
    if not label:
        return key
    return f"{label}.{key}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{_label_path(label, key)} is required by train spec schema")
    return document[key]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_non_empty_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _require_key(document, key, label=label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    return value


def _require_nullable_non_empty_string(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str | None:
    value = _require_key(document, key, label=label)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be null or a non-empty string")
    return value


def _require_int(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    value = _require_key(document, key, label=label)
    if not _is_int(value):
        raise ValueError(f"{_label_path(label, key)} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum}")
    return value


def _require_bool(document: Mapping[str, Any], key: str, *, label: str) -> bool:
    value = _require_key(document, key, label=label)
    if not isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a boolean")
    return value


def _require_string_list(document: Mapping[str, Any], key: str, *, label: str) -> list[str]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-empty string")
        values.append(item)
    return values


def _require_int_list(document: Mapping[str, Any], key: str, *, label: str) -> list[int]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value:
        raise ValueError(f"{_label_path(label, key)} must contain at least one seed")
    values: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item):
            raise ValueError(f"{_label_path(label, key)}[{index}] must be an integer")
        values.append(item)
    return values


def _format_field_names(template: str) -> set[str]:
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        root_name = field_name.split(".", 1)[0].split("[", 1)[0]
        names.add(root_name)
    return names


def _require_template(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    required_fields: set[str],
) -> str:
    template = _require_non_empty_string(document, key, label=label)
    field_names = _format_field_names(template)
    unknown = sorted(field_names - TRAIN_SPEC_ALLOWED_TEMPLATE_FIELDS)
    if unknown:
        raise ValueError(
            f"{_label_path(label, key)} uses unsupported template field(s): "
            f"{', '.join(unknown)}"
        )
    missing = sorted(required_fields - field_names)
    if missing:
        raise ValueError(
            f"{_label_path(label, key)} must include template field(s): {', '.join(missing)}"
        )
    try:
        template.format(seed=123, slug="candidate", utc="20260626T120000Z")
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{_label_path(label, key)} is not a valid format template: {exc}") from exc
    return template


def validate_train_spec_schema(document: Mapping[str, Any], *, label: str = "spec") -> None:
    """Validate the non-negotiable queue-backed train spec contract.

    Unknown top-level and train_config fields are intentionally allowed so older
    research metadata can keep flowing into spec_payload_json.
    """

    _require_mapping(document, label=label)
    if "schema_version" in document and (
        schema_version := _require_int(document, "schema_version", label=label, minimum=1)
    ) != TRAIN_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"{_label_path(label, 'schema_version')} must be "
            f"{TRAIN_SPEC_SCHEMA_VERSION}, got {schema_version}"
        )

    goal = _require_mapping(_require_key(document, "goal", label=label), label=_label_path(label, "goal"))
    _require_non_empty_string(goal, "goal_id", label=_label_path(label, "goal"))
    _require_non_empty_string(document, "slug", label=label)
    _require_non_empty_string(document, "hypothesis", label=label)
    if "parent_spec_slug" in document:
        _require_nullable_non_empty_string(document, "parent_spec_slug", label=label)
    if "max_attempts" in document:
        _require_int(document, "max_attempts", label=label, minimum=1)
    seed_values = _require_int_list(document, "seeds", label=label) if "seeds" in document else []
    _require_non_empty_string(document, "wandb_group", label=label)
    if "run_name_label" in document:
        _require_non_empty_string(document, "run_name_label", label=label)
    _require_string_list(document, "wandb_tags", label=label)
    _require_template(
        document,
        "run_description_template",
        label=label,
        required_fields={"seed"},
    )

    if "selection_metrics" in document:
        metrics = _require_string_list(document, "selection_metrics", label=label)
        if not metrics:
            raise ValueError(f"{_label_path(label, 'selection_metrics')} must not be empty")
    elif "selection_policy" in document:
        _require_mapping(
            _require_key(document, "selection_policy", label=label),
            label=_label_path(label, "selection_policy"),
        )
    else:
        raise ValueError(
            f"{label} must define selection_metrics or inherit goal-owned selection_policy"
        )

    train_config = _require_mapping(
        _require_key(document, "train_config", label=label),
        label=_label_path(label, "train_config"),
    )
    seed_span = train_config.get("n_envs", 1)
    for index, seed in enumerate(seed_values):
        validate_training_seed(
            seed,
            label=f"{_label_path(label, 'seeds')}[{index}]",
            seed_span=seed_span,
        )
    _require_non_empty_string(train_config, "game", label=_label_path(label, "train_config"))
    has_state = isinstance(train_config.get("state"), str) and bool(train_config["state"].strip())
    states = train_config.get("states")
    has_states = (
        isinstance(states, Sequence)
        and not isinstance(states, str | bytes)
        and bool(states)
        and all(isinstance(state, str) and bool(state.strip()) for state in states)
    )
    if not has_state and not has_states:
        raise ValueError(
            f"{_label_path(label, 'train_config')} must define non-empty state or states"
        )
    _require_int(train_config, "timesteps", label=_label_path(label, "train_config"), minimum=1)
    if "seed" in train_config and train_config["seed"] is not None:
        validate_training_seed(
            train_config["seed"],
            label=_label_path(label, "train_config.seed"),
            seed_span=train_config.get("n_envs", 1),
        )
    _require_bool(train_config, "wandb", label=_label_path(label, "train_config"))
    wandb_mode = _require_non_empty_string(
        train_config,
        "wandb_mode",
        label=_label_path(label, "train_config"),
    )
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            f"{_label_path(label, 'train_config.wandb_mode')} must be one of "
            "online, offline, disabled"
        )
