from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any

from rlab.seeds import validate_training_seed
from rlab.train_config import queue_required_train_config_fields


TRAIN_RECIPE_SCHEMA_VERSION = 1
TRAIN_RECIPE_REQUIRED_FIELDS = (
    "goal",
    "recipe_id",
    "description",
    "group_id",
    "tags",
    "train_config",
)
TRAIN_RECIPE_REQUIRED_TRAIN_CONFIG_FIELDS = queue_required_train_config_fields()
EXPLICIT_QUEUE_TRAIN_CONFIG_FIELDS = TRAIN_RECIPE_REQUIRED_TRAIN_CONFIG_FIELDS
TRAIN_RECIPE_ALLOWED_TEMPLATE_FIELDS = frozenset(
    {"group_id", "seed", "recipe_id", "timestamp", "utc"}
)
TRAIN_RECIPE_REMOVED_FIELDS = frozenset(
    {
        "hypothesis",
        "run_description_template",
        "slug",
        "wandb_tags",
        "wandb_group",
    }
)


def _label_path(label: str, key: str) -> str:
    if not label:
        return key
    return f"{label}.{key}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{_label_path(label, key)} is required by train recipe schema")
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


def require_explicit_queue_train_config(
    train_config: Mapping[str, Any],
    *,
    label: str = "train_config",
) -> None:
    missing = [key for key in EXPLICIT_QUEUE_TRAIN_CONFIG_FIELDS if key not in train_config]
    if missing:
        raise ValueError(
            f"{label} missing required recipe-defined field(s): "
            f"{', '.join(missing)}; queue-backed train jobs must define train values in recipes"
        )


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
    unknown = sorted(field_names - TRAIN_RECIPE_ALLOWED_TEMPLATE_FIELDS)
    if unknown:
        raise ValueError(
            f"{_label_path(label, key)} uses unsupported template field(s): {', '.join(unknown)}"
        )
    missing = sorted(required_fields - field_names)
    if missing:
        raise ValueError(
            f"{_label_path(label, key)} must include template field(s): {', '.join(missing)}"
        )
    try:
        template.format(
            seed=123,
            recipe_id="candidate",
            timestamp="20260626T120000Z",
            utc="20260626T120000Z",
            group_id="b-test",
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            f"{_label_path(label, key)} is not a valid format template: {exc}"
        ) from exc
    return template


def train_recipe_id(document: Mapping[str, Any]) -> str:
    return str(document.get("recipe_id") or "").strip()


def validate_train_recipe_schema(document: Mapping[str, Any], *, label: str = "recipe") -> None:
    """Validate the non-negotiable queue-backed train recipe contract.

    Unknown top-level and train_config fields are intentionally allowed so older
    research metadata can keep flowing into the persisted queue payload.
    """

    _require_mapping(document, label=label)
    removed_fields = sorted(field for field in TRAIN_RECIPE_REMOVED_FIELDS if field in document)
    if removed_fields:
        raise ValueError(f"{label} uses removed train recipe field(s): {', '.join(removed_fields)}")
    if (
        "schema_version" in document
        and (schema_version := _require_int(document, "schema_version", label=label, minimum=1))
        != TRAIN_RECIPE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{_label_path(label, 'schema_version')} must be "
            f"{TRAIN_RECIPE_SCHEMA_VERSION}, got {schema_version}"
        )

    goal = _require_mapping(
        _require_key(document, "goal", label=label), label=_label_path(label, "goal")
    )
    _require_non_empty_string(goal, "goal_id", label=_label_path(label, "goal"))
    if not train_recipe_id(document):
        raise ValueError(f"{label}.recipe_id is required by train recipe schema")
    _require_template(
        document,
        "description",
        label=label,
        required_fields=set(),
    )
    if "max_attempts" in document:
        _require_int(document, "max_attempts", label=label, minimum=1)
    seed_values = _require_int_list(document, "seeds", label=label) if "seeds" in document else []
    _require_non_empty_string(document, "group_id", label=label)
    if "run_name_label" in document:
        _require_non_empty_string(document, "run_name_label", label=label)
    if "run_name_template" in document:
        _require_template(
            document,
            "run_name_template",
            label=label,
            required_fields=set(),
        )
    _require_string_list(document, "tags", label=label)
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
    artifact_uri = _require_key(
        train_config,
        "wandb_artifact_storage_uri",
        label=_label_path(label, "train_config"),
    )
    if not isinstance(artifact_uri, str):
        raise ValueError(
            f"{_label_path(label, 'train_config.wandb_artifact_storage_uri')} must be a string"
        )
