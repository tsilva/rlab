from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlab.config_loader import (
    YAML_EXTENSIONS,
    ComposedDocument,
    QUEUE_TEMPLATE_FIELDS,
    TEMPLATE_VARS_KEY,
    apply_dotlist_overrides,
    deep_merge,
    load_composed_mapping,
    load_mapping_document,
    render_template_vars,
    template_context_from_path,
)
from rlab.env_identity import (
    attach_environment_identity,
    train_config_from_source_environment,
    validate_task_config,
)
from rlab.file_utils import file_sha256
from rlab.provider_config import provider_num_envs
from rlab.recipe_schema import (
    TRAIN_RECIPE_SCHEMA_VERSION,
    train_recipe_id,
    validate_materialized_train_recipe,
)
from rlab.seeds import validate_training_seed
from rlab.train_config import train_config_keys_in_source_section, train_config_keys_owned_by


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)
TRAIN_CONFIG_SECTION_KEYS = ("train", "logging")
TRAIN_NESTED_SECTION_KEYS = frozenset({"backend"})
COMMON_TRAIN_CONFIG_KEYS = train_config_keys_in_source_section("train")
GOAL_TRAIN_CONFIG_KEYS = train_config_keys_in_source_section("goal_train")
SOURCE_RECIPE_FIELDS = frozenset(
    {
        "campaign_id",
        "defaults",
        "description",
        "logging",
        "max_attempts",
        "metadata",
        "notes",
        "recipe_id",
        "schema_version",
        "seeds",
        TEMPLATE_VARS_KEY,
        "train",
    }
)
SOURCE_PRESET_FIELDS = frozenset({"defaults", "logging", TEMPLATE_VARS_KEY, "train"})
RECIPE_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("description",): QUEUE_TEMPLATE_FIELDS,
    ("goal", "description"): QUEUE_TEMPLATE_FIELDS,
    ("goal", "tags", "2"): frozenset({"env_id"}),
}
GOAL_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    **RECIPE_DEFERRED_TEMPLATE_FIELDS,
    ("tags", "1"): frozenset({"slug", "recipe_id", "recipe_slug"}),
    ("tags", "2"): frozenset({"env_id"}),
}
GOAL_OWNED_ENV_CONFIG_KEYS = train_config_keys_owned_by("goal_environment") | {
    "provider",
    "env_id",
}
GOAL_OWNED_OBJECTIVE_CONFIG_KEYS = train_config_keys_owned_by("goal_objective")


def goal_contract_sha256(document: Mapping[str, Any]) -> str:
    """Hash the fully composed semantic goal contract, excluding source formatting."""

    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contains_secret_key(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            nested_path = f"{path}.{key}" if path else str(key)
            if any(fragment in key_text for fragment in SECRET_KEY_FRAGMENTS):
                return nested_path
            found = _contains_secret_key(nested, nested_path)
            if found:
                return found
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            found = _contains_secret_key(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def assert_no_secrets(value: Any, *, label: str) -> None:
    found = _contains_secret_key(value)
    if found:
        raise ValueError(f"{label} appears to contain a secret-like key: {found}")


def _document_train_environment(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    train_section = document.get("train")
    if isinstance(train_section, Mapping):
        train_environment = train_section.get("environment")
        if isinstance(train_environment, Mapping):
            return train_environment
    return None


def _without_keys(value: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {
        nested_key: copy.deepcopy(nested_value)
        for nested_key, nested_value in value.items()
        if nested_key not in keys
    }


def _goal_train_defaults(document: Mapping[str, Any]) -> dict[str, Any]:
    environment = _document_train_environment(document)
    config = (
        _train_environment_section_config(environment) if isinstance(environment, Mapping) else {}
    )
    train = document.get("train")
    if isinstance(train, Mapping):
        config = deep_merge(config, _train_config_from_train_section(train))
    config = deep_merge(config, _eval_train_defaults(document))
    objective = document.get("objective")
    if isinstance(objective, Mapping) and isinstance(objective.get("rank"), Sequence):
        config["selection_rank"] = copy.deepcopy(objective["rank"])
    return config


def _eval_train_defaults(document: Mapping[str, Any]) -> dict[str, Any]:
    eval_section = document.get("eval")
    if not isinstance(eval_section, Mapping):
        return {}
    episodes = eval_section.get("episodes")
    if episodes is None:
        return {}
    defaults: dict[str, Any] = {"post_train_eval_episodes": copy.deepcopy(episodes)}
    if "acceptance" in eval_section:
        defaults["checkpoint_eval_acceptance"] = copy.deepcopy(
            eval_section["acceptance"]
        )
    environment = eval_section.get("environment")
    if not isinstance(environment, Mapping):
        return defaults
    eval_config = _train_environment_section_config(environment)
    if "n_envs" in eval_config:
        defaults["checkpoint_eval_n_envs"] = eval_config.pop("n_envs")
    if "max_steps" in eval_config:
        defaults["post_train_eval_max_steps"] = eval_config.pop("max_steps")
    defaults["checkpoint_eval_environment"] = eval_config
    return defaults


def _train_config_section_value(
    document: Mapping[str, Any],
    key: str,
    *,
    strip_goal_owned: bool = False,
) -> Mapping[str, Any] | None:
    value = document.get(key)
    if not isinstance(value, Mapping):
        return None
    if key != "train":
        section = dict(value)
    else:
        section = _train_config_from_train_section(value)
    if not strip_goal_owned:
        return section
    if key == "logging":
        return _without_keys(section, GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    if key == "train":
        return _without_keys(section, GOAL_OWNED_ENV_CONFIG_KEYS | GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    return section


def _train_environment_section_config(environment: Mapping[str, Any]) -> dict[str, Any]:
    return train_config_from_source_environment(environment)


def _normalized_train_section(section: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(section, Mapping):
        return {}
    nested_environment = section.get("environment")
    environment = (
        copy.deepcopy(dict(nested_environment)) if isinstance(nested_environment, Mapping) else {}
    )
    common = {
        key: copy.deepcopy(value)
        for key, value in section.items()
        if key in GOAL_TRAIN_CONFIG_KEYS | COMMON_TRAIN_CONFIG_KEYS
    }
    nested_backend = section.get("backend")
    backend = copy.deepcopy(dict(nested_backend)) if isinstance(nested_backend, Mapping) else {}
    normalized: dict[str, Any] = {}
    if environment:
        normalized["environment"] = environment
    normalized.update(common)
    if backend:
        normalized["backend"] = backend
    return normalized


def _train_config_from_train_section(section: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_train_section(section)
    config: dict[str, Any] = {}
    environment = normalized.get("environment")
    if isinstance(environment, Mapping):
        config = deep_merge(config, _train_environment_section_config(environment))
    common = {
        key: copy.deepcopy(value)
        for key, value in normalized.items()
        if key in GOAL_TRAIN_CONFIG_KEYS | COMMON_TRAIN_CONFIG_KEYS
    }
    config = deep_merge(config, common)
    backend = normalized.get("backend")
    if isinstance(backend, Mapping):
        config["training_backend"] = copy.deepcopy(dict(backend))
    return config


def _explicit_train_environment_config(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    train = document.get("train")
    if not isinstance(train, Mapping):
        return None
    environment = train.get("environment")
    if not isinstance(environment, Mapping):
        return None
    return _train_environment_section_config(environment)


def _merge_train_config_sections(
    document: Mapping[str, Any],
    *,
    goal_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strip_goal_owned = goal_document is not None
    train_config: dict[str, Any] = _goal_train_defaults(goal_document or {})
    for key in TRAIN_CONFIG_SECTION_KEYS:
        value = _train_config_section_value(document, key, strip_goal_owned=strip_goal_owned)
        if isinstance(value, Mapping):
            train_config = deep_merge(train_config, value)
    if strip_goal_owned:
        explicit_environment = _explicit_train_environment_config(document)
        if isinstance(explicit_environment, Mapping):
            train_config = deep_merge(train_config, explicit_environment)

    return train_config


def _goal_slug_from_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("goal_id") or "").strip()
    return ""


def _load_rendered_goal_composition(path: Path, *, label: str | None = None) -> ComposedDocument:
    composition = load_composed_mapping(path, cycle_label="goal")
    return ComposedDocument(
        document=render_template_vars(
            composition.document,
            path=path,
            label=label or f"goal file {path}",
            deferred_fields_by_path=GOAL_DEFERRED_TEMPLATE_FIELDS,
        ),
        sources=composition.sources,
    )


def _reject_active_specs_path(path: Path) -> None:
    if "specs" in path.parts and ".deprecated" not in path.parts:
        raise ValueError(f"{path} is under removed active specs/ layout; use recipes/ instead")


def _materialize_goal_owned_fields(
    materialized: dict[str, Any],
    *,
    path: Path | None = None,
    goal_composition: ComposedDocument | None = None,
) -> Mapping[str, Any] | None:
    if goal_composition is None:
        return None
    goal_document = goal_composition.document
    materialized["goal"] = copy.deepcopy(dict(goal_document))
    return goal_document


def _materialize_goal_train_environment(
    materialized: dict[str, Any],
    goal_document: Mapping[str, Any] | None,
) -> None:
    if goal_document is None:
        return
    goal_environment = _document_train_environment(goal_document)
    if not isinstance(goal_environment, Mapping):
        return
    train = _normalized_train_section(materialized.get("train"))
    train["environment"] = deep_merge(
        copy.deepcopy(dict(goal_environment)),
        train.get("environment") if isinstance(train.get("environment"), Mapping) else {},
    )
    materialized["train"] = train


def _materialize_goal_queue_defaults(
    materialized: dict[str, Any],
    goal_document: Mapping[str, Any] | None,
    *,
    path: Path | None = None,
) -> None:
    if goal_document is None:
        return
    for key in ("campaign_id",):
        if key in materialized:
            continue
        value = goal_document.get(key)
        if isinstance(value, str) and value.strip():
            materialized[key] = value
    if "tags" not in materialized:
        tags = goal_document.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, str | bytes):
            materialized["tags"] = list(tags)
            if path is not None:
                tag_document = {
                    "tags": materialized["tags"],
                    "recipe_id": materialized.get("recipe_id"),
                    "train": goal_document.get("train"),
                }
                materialized["tags"] = render_template_vars(
                    tag_document,
                    path=path,
                    label=f"goal tags for recipe file {path}",
                )["tags"]


def materialize_train_recipe_document(
    document: Mapping[str, Any],
    *,
    path: Path | None = None,
    goal_composition: ComposedDocument | None = None,
) -> dict[str, Any]:
    materialized = copy.deepcopy(dict(document))
    source_sections = [key for key in TRAIN_CONFIG_SECTION_KEYS if key in materialized]
    if isinstance(materialized.get("train_config"), Mapping):
        if source_sections:
            raise ValueError(
                "recipe cannot mix compiled train_config with source section(s): "
                + ", ".join(source_sections)
            )
        return materialized
    normalized_train = _normalized_train_section(materialized.get("train"))
    if normalized_train:
        materialized["train"] = normalized_train
    goal_document = _materialize_goal_owned_fields(
        materialized,
        path=path,
        goal_composition=goal_composition,
    )
    _materialize_goal_queue_defaults(materialized, goal_document, path=path)
    _materialize_goal_train_environment(materialized, goal_document)
    train_config = _merge_train_config_sections(materialized, goal_document=goal_document)
    if train_config:
        from rlab.training_backend import accepts_first_training_success

        if accepts_first_training_success(train_config):
            train_config["checkpoint_eval_backend"] = "none"
            train_config["early_stop"] = None
            train_config["checkpoint_eval_stages"] = []
            train_config.pop("checkpoint_eval_asset_manifest", None)
    if train_config:
        materialized["train_config"] = train_config
    return materialized


def _recipe_source_metadata(sources: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {
            "path": str(source),
            "sha256": file_sha256(source),
        }
        for source in sources
    ]


def assert_no_template_vars(value: Any, *, label: str = "document") -> None:
    if isinstance(value, Mapping):
        if TEMPLATE_VARS_KEY in value:
            raise ValueError(f"{label} still contains {TEMPLATE_VARS_KEY}; render templates first")
        for key, nested in value.items():
            assert_no_template_vars(nested, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            assert_no_template_vars(nested, label=f"{label}[{index}]")


def validate_source_recipe_shape(
    document: Mapping[str, Any],
    *,
    label: str,
    preset: bool = False,
) -> None:
    allowed_fields = SOURCE_PRESET_FIELDS if preset else SOURCE_RECIPE_FIELDS
    retired = sorted(
        set(document) & {"environment", "reward", "train_config", "group_id", "batch_id"}
    )
    if retired:
        raise ValueError(
            f"{label} uses compiled or retired source field(s): {', '.join(retired)}; "
            "author recipes with train.backend and logging"
        )
    unknown = sorted(str(key) for key in set(document) - allowed_fields)
    if unknown:
        kind = "recipe preset" if preset else "recipe"
        raise ValueError(
            f"{label} uses goal-owned or unsupported {kind} field(s): {', '.join(unknown)}"
        )
    train = document.get("train")
    if train is not None:
        if not isinstance(train, Mapping):
            raise ValueError(f"{label}.train must be an object")
        if "policy" in train:
            raise ValueError(
                f"{label}.train.policy is retired; use train.backend with an explicit id and config"
            )
        allowed = TRAIN_NESTED_SECTION_KEYS | COMMON_TRAIN_CONFIG_KEYS
        unexpected = sorted(set(train) - allowed)
        if unexpected:
            raise ValueError(
                f"{label}.train uses unsupported flat field(s): {', '.join(unexpected)}; "
                "put common fields directly under train and backend options under "
                "train.backend.config"
            )
    if not preset:
        recipe_id = train_recipe_id(document)
        if not recipe_id:
            raise ValueError(f"{label}.recipe_id is required")
        if recipe_id == "base":
            raise ValueError(f"{label}.recipe_id=base is unsupported; use an explicit recipe id")
        description = document.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}.description is required")


def load_recipe_source_document(path: Path) -> ComposedDocument:
    _reject_active_specs_path(path)
    validate_source_recipe_shape(
        load_mapping_document(path, label=f"recipe file {path}"),
        label=f"recipe file {path}",
    )
    composed = load_composed_mapping(
        path,
        cycle_label="recipe",
    )
    validate_source_recipe_shape(
        composed.document,
        label=f"composed recipe file {path}",
    )
    active_root = (Path("experiments") / "recipes").resolve()
    resolved_path = path.resolve()
    if resolved_path.is_relative_to(active_root):
        for source in composed.sources[:-1]:
            relative = source.relative_to(active_root)
            if not relative.parts or relative.parts[0] != "_presets":
                raise ValueError(
                    f"recipe leaf {path} may compose only presets under "
                    f"experiments/recipes/_presets, got {source}"
                )
    for source in composed.sources[:-1]:
        validate_source_recipe_shape(
            load_mapping_document(source, label=f"recipe preset {source}"),
            label=f"recipe preset {source}",
            preset=True,
        )
    return composed


def compose_train_document(
    goal_path: Path,
    recipe_path: Path,
    *,
    recipe_overrides: Sequence[str] = (),
) -> dict[str, Any]:
    goal_composition = _load_rendered_goal_composition(goal_path)
    if goal_composition.sources:
        from rlab.config_validation import validate_goal_contract_document

        validate_goal_contract_document(
            goal_composition.document,
            goal_composition.sources[-1],
            Path(".").resolve(),
        )
    recipe_composition = load_recipe_source_document(recipe_path)
    recipe_override_list = [str(item).strip() for item in recipe_overrides if str(item).strip()]
    source_document = apply_dotlist_overrides(
        recipe_composition.document,
        recipe_override_list,
        label=f"recipe overrides for {recipe_path}",
    )
    recipe_id = train_recipe_id(source_document)
    goal_context = template_context_from_path(goal_path, goal_composition.document)
    document = render_template_vars(
        source_document,
        path=goal_path,
        label=f"recipe file {recipe_path} for goal file {goal_path}",
        extra_context={
            **goal_context,
            "recipe_id": recipe_id,
            "recipe_slug": recipe_id,
            "slug": recipe_id,
        },
        deferred_fields_by_path=RECIPE_DEFERRED_TEMPLATE_FIELDS,
    )
    sources = [*goal_composition.sources, *recipe_composition.sources]
    sources = list(dict.fromkeys(sources))
    document = materialize_train_recipe_document(
        document,
        path=goal_path,
        goal_composition=goal_composition,
    )
    document["train_config"]["goal_contract_sha256"] = goal_contract_sha256(
        goal_composition.document
    )
    document = attach_environment_identity(document)
    if recipe_override_list:
        document["recipe_overrides"] = recipe_override_list
    if recipe_path.suffix.lower() in YAML_EXTENSIONS or len(sources) > 1:
        document["_composition"] = {
            "goal_root_path": str(goal_path.resolve()),
            "recipe_root_path": str(recipe_path.resolve()),
            "source_files": _recipe_source_metadata(sources),
        }
    label = f"goal file {goal_path} with recipe file {recipe_path}"
    validate_materialized_train_recipe(document, label=label)
    assert_no_template_vars(document, label=label)
    assert_no_secrets(document, label=label)
    validate_launch_event_config(
        document["train_config"],
        label=f"{label} train_config",
    )
    return document


def _git_text(args: Sequence[str], *, cwd: Path = Path(".")) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


def repo_git_commit(cwd: Path = Path(".")) -> str | None:
    return _git_text(("rev-parse", "HEAD"), cwd=cwd)


def repo_is_dirty(cwd: Path = Path(".")) -> bool:
    text = _git_text(("status", "--porcelain"), cwd=cwd)
    return bool(text)


def recipe_slug(document: Mapping[str, Any]) -> str:
    return train_recipe_id(document)


def compiled_recipe_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact, traceable recipe identity persisted with a queue row.

    The queue row separately owns the resolved train config and execution metadata;
    source file hashes in ``_composition`` preserve the exact goal/recipe inputs.
    """

    payload: dict[str, Any] = {
        "schema_version": TRAIN_RECIPE_SCHEMA_VERSION,
        "goal_id": recipe_goal_slug(document),
        "recipe_id": recipe_slug(document),
        "description": document.get("description"),
        "tags": recipe_tags(document),
    }
    for key in ("campaign_id", "recipe_overrides", "_composition"):
        value = document.get(key)
        if value not in (None, "", (), [], {}):
            payload[key] = copy.deepcopy(value)
    return payload


def recipe_metadata(
    goal_path: Path,
    recipe_path: Path,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    slug = recipe_slug(document)
    return {
        "goal_slug": recipe_goal_slug(document),
        "goal_path": str(goal_path),
        "goal_sha256": file_sha256(goal_path),
        "goal_contract_sha256": str(
            document.get("train_config", {}).get("goal_contract_sha256") or ""
        ),
        "recipe_slug": slug,
        "recipe_path": str(recipe_path),
        "recipe_sha256": file_sha256(recipe_path),
        "recipe_payload": compiled_recipe_payload(document),
        "repo_git_commit": repo_git_commit(),
        "repo_dirty": repo_is_dirty(),
    }


def _non_empty_config_value(value: Any) -> bool:
    return value not in (None, "", (), [], {})


def validate_launch_event_config(
    train_config: Mapping[str, Any], *, label: str = "train_config"
) -> None:
    task = train_config.get("task")
    if isinstance(task, Mapping):
        validate_task_config(task, label=f"{label}.task")


def validate_launch_seed_config(
    train_config: Mapping[str, Any],
    *,
    seed: int | None = None,
    label: str = "train_config",
) -> None:
    config_seed = train_config.get("seed")
    seed_span = provider_num_envs(train_config, explicit_n_envs=train_config.get("n_envs"))
    if _non_empty_config_value(config_seed):
        validate_training_seed(config_seed, label=f"{label}.seed", seed_span=seed_span)
    if seed is not None:
        validate_training_seed(seed, label="seed", seed_span=seed_span)


def recipe_goal_slug(document: Mapping[str, Any]) -> str:
    return _goal_slug_from_value(document.get("goal"))


def recipe_tags(document: Mapping[str, Any]) -> list[str]:
    tags = []
    seen: set[str] = set()
    for raw_tag in document.get("tags") or []:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def load_goal_contract_document(path: Path, *, label: str | None = None) -> dict[str, Any]:
    return _load_rendered_goal_composition(path, label=label).document
