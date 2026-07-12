from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rlab.config_loader import QUEUE_TEMPLATE_VALUES, validate_template_string
from rlab.env_registry import env_supports_states
from rlab.seeds import validate_training_seed
from rlab.train_config import (
    queue_required_train_config_fields,
    validate_and_normalize_train_config,
)
from rlab.provider_config import provider_num_envs
from rlab.validation import (
    int_list,
    label_path,
    require_int,
    require_key,
    require_mapping,
    require_non_empty_string,
    string_list,
)


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
TRAIN_RECIPE_OPTIONAL_FIELDS = frozenset(
    {
        "schema_version",
        "seeds",
        "batch_id",
        "max_attempts",
        "metadata",
        "notes",
        "recipe_overrides",
        "_composition",
        "environment",
        "environment_hash",
        "eval",
        "goal_id",
        "logging",
        "objective",
        "release",
        "title",
        "train",
    }
)
TRAIN_RECIPE_ALLOWED_FIELDS = frozenset(TRAIN_RECIPE_REQUIRED_FIELDS) | TRAIN_RECIPE_OPTIONAL_FIELDS
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


def _require_template(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    required_fields: set[str],
) -> str:
    template = require_non_empty_string(document, key, label=label, strip=False)
    validate_template_string(
        template,
        allowed_values=QUEUE_TEMPLATE_VALUES,
        required_fields=frozenset(required_fields),
        label=label_path(label, key),
    )
    return template


def train_recipe_id(document: Mapping[str, Any]) -> str:
    return str(document.get("recipe_id") or "").strip()


def _reject_unknown_fields(document: Mapping[str, Any], *, label: str) -> None:
    unknown = sorted(str(field) for field in document if field not in TRAIN_RECIPE_ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"{label} uses unknown train recipe field(s): {', '.join(unknown)}")


def validate_materialized_train_recipe(
    document: Mapping[str, Any], *, label: str = "recipe"
) -> None:
    """Validate a goal-composed recipe immediately before queue persistence."""

    require_mapping(document, label=label)
    _reject_unknown_fields(document, label=label)
    if (
        "schema_version" in document
        and (schema_version := require_int(document, "schema_version", label=label, minimum=1))
        != TRAIN_RECIPE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{label_path(label, 'schema_version')} must be "
            f"{TRAIN_RECIPE_SCHEMA_VERSION}, got {schema_version}"
        )

    goal = require_mapping(
        require_key(document, "goal", label=label), label=label_path(label, "goal")
    )
    require_non_empty_string(goal, "goal_id", label=label_path(label, "goal"))
    if not train_recipe_id(document):
        raise ValueError(f"{label}.recipe_id is required by train recipe schema")
    _require_template(
        document,
        "description",
        label=label,
        required_fields=set(),
    )
    if "max_attempts" in document:
        require_int(document, "max_attempts", label=label, minimum=1)
    seed_values = (
        int_list(require_key(document, "seeds", label=label), label=label_path(label, "seeds"))
        if "seeds" in document
        else []
    )
    require_non_empty_string(document, "group_id", label=label)
    if "batch_id" in document:
        require_non_empty_string(document, "batch_id", label=label)
    string_list(require_key(document, "tags", label=label), label=label_path(label, "tags"))

    train_config = require_mapping(
        require_key(document, "train_config", label=label),
        label=label_path(label, "train_config"),
    )
    validate_and_normalize_train_config(
        train_config,
        label=label_path(label, "train_config"),
        required_keys=("game", "timesteps", "wandb", "wandb_mode", "wandb_artifact_storage_uri"),
    )
    seed_span = provider_num_envs(train_config, explicit_n_envs=train_config.get("n_envs"))
    for index, seed in enumerate(seed_values):
        validate_training_seed(
            seed,
            label=f"{label_path(label, 'seeds')}[{index}]",
            seed_span=seed_span,
        )
    has_state = isinstance(train_config.get("state"), str) and bool(train_config["state"].strip())
    states = train_config.get("states")
    has_states = (
        isinstance(states, Sequence)
        and not isinstance(states, str | bytes)
        and bool(states)
        and all(isinstance(state, str) and bool(state.strip()) for state in states)
    )
    provider_id = str(train_config.get("env_provider") or "").strip()
    game = str(train_config.get("game") or "").strip()
    supports_states = env_supports_states(provider_id, game) if provider_id else True
    if supports_states and not has_state and not has_states:
        raise ValueError(
            f"{label_path(label, 'train_config')} must define non-empty state or states"
        )
    if "seed" in train_config and train_config["seed"] is not None:
        validate_training_seed(
            train_config["seed"],
            label=label_path(label, "train_config.seed"),
            seed_span=seed_span,
        )
