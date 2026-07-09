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
    TEMPLATE_VARS_KEY,
    deep_merge,
    load_composed_mapping,
    render_template_vars,
)
from rlab.env_identity import attach_environment_identity, train_config_from_environment
from rlab.seeds import validate_training_seed
from rlab.recipe_schema import train_recipe_id, validate_train_recipe_schema


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)
LEGACY_EVENT_TRAIN_CONFIG_KEYS = ("done_on_info_json", "done_on_info")
TRAIN_CONFIG_SECTION_KEYS = ("env", "train", "reward", "logging")
TRAIN_CONFIG_TOP_LEVEL_KEYS = ("state", "states", "state_probs", "resume")
TRAIN_NESTED_SECTION_KEYS = frozenset({"environment", "policy"})
PROVIDER_OWNED_INFO_EVENTS = {
    "stable-retro-turbo": frozenset({"life_loss", "level_change"}),
    "supermariobrosnes-turbo": frozenset({"life_loss", "level_change"}),
}
GOAL_GAME_DIR_NAMES = frozenset({"SuperMarioBros-Nes-v0", "super-mario-bros-nes-v0"})
QUEUE_TEMPLATE_FIELDS = frozenset({"group_id", "seed", "recipe_id", "timestamp", "utc"})
RECIPE_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("description",): QUEUE_TEMPLATE_FIELDS,
    ("goal", "description"): QUEUE_TEMPLATE_FIELDS,
    ("goal", "tags", "2"): frozenset({"env_id"}),
    ("run_name_template",): QUEUE_TEMPLATE_FIELDS,
    ("goal", "run_name_template"): QUEUE_TEMPLATE_FIELDS,
    (
        "goal",
        "release",
        "huggingface",
        "checkpoint_filename",
    ): frozenset({"checkpoint_step"}),
    (
        "release",
        "huggingface",
        "checkpoint_filename",
    ): frozenset({"checkpoint_step"}),
}
GOAL_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    **RECIPE_DEFERRED_TEMPLATE_FIELDS,
    ("tags", "1"): frozenset({"slug", "recipe_id", "recipe_slug"}),
    ("tags", "2"): frozenset({"env_id"}),
}
GOAL_OWNED_ENV_CONFIG_KEYS = frozenset(
    {
        "env_provider",
        "provider",
        "env_id",
        "env_args",
        "game",
        "state",
        "states",
        "state_probs",
        "task_conditioning",
        "task_conditioning_info_vars",
        "task_conditioning_info_values",
        "action_set",
        "frame_skip",
        "max_pool_frames",
        "sticky_action_prob",
        "obs_crop",
        "obs_resize_algorithm",
        "observation_size",
        "hud_crop_top",
        "max_episode_steps",
        "episodic_life",
        "info_events",
        "info_events_json",
        "done_on_events",
        "env_wrappers",
        "n_envs",
        "env_threads",
    }
)
GOAL_OWNED_OBJECTIVE_CONFIG_KEYS = frozenset(
    {
        "early_stop",
    }
)


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
    environment = document.get("environment")
    return environment if isinstance(environment, Mapping) else None


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
    return config


def _selection_policy_from_goal(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selection_policy = document.get("selection_policy")
    if isinstance(selection_policy, Mapping):
        return selection_policy
    objective = document.get("objective")
    if not isinstance(objective, Mapping):
        return None
    rank = objective.get("rank")
    if isinstance(rank, Sequence) and not isinstance(rank, str | bytes):
        return {"rank_order": copy.deepcopy(rank)}
    return None


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
    if key == "env":
        return _without_keys(section, GOAL_OWNED_ENV_CONFIG_KEYS)
    if key == "logging":
        return _without_keys(section, GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    if key == "train":
        return _without_keys(section, GOAL_OWNED_ENV_CONFIG_KEYS | GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    return section


def _train_environment_section_config(environment: Mapping[str, Any]) -> dict[str, Any]:
    config = train_config_from_environment(environment)
    direct_items = {
        key: copy.deepcopy(value)
        for key, value in environment.items()
        if key
        not in {
            "env_config",
            "env_id",
            "state",
            "states",
            "state_probs",
            "action",
            "preprocessing",
            "task_conditioning",
            "termination",
            "reward",
        }
    }
    return deep_merge(config, direct_items)


def _normalized_train_section(section: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(section, Mapping):
        return {}
    nested_environment = section.get("environment")
    environment = copy.deepcopy(dict(nested_environment)) if isinstance(nested_environment, Mapping) else {}
    policy = {
        key: copy.deepcopy(value)
        for key, value in section.items()
        if key not in TRAIN_NESTED_SECTION_KEYS
    }
    nested_policy = section.get("policy")
    if isinstance(nested_policy, Mapping):
        policy = deep_merge(policy, nested_policy)
    normalized: dict[str, Any] = {}
    if environment:
        normalized["environment"] = environment
    if policy:
        normalized["policy"] = policy
    return normalized


def _train_config_from_train_section(section: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_train_section(section)
    config: dict[str, Any] = {}
    environment = normalized.get("environment")
    if isinstance(environment, Mapping):
        config = deep_merge(config, _train_environment_section_config(environment))
    policy = normalized.get("policy")
    if isinstance(policy, Mapping):
        config = deep_merge(config, policy)
    return config


def _explicit_train_environment_config(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    train = document.get("train")
    if not isinstance(train, Mapping):
        return None
    environment = train.get("environment")
    if not isinstance(environment, Mapping):
        return None
    return _train_environment_section_config(environment)


def _top_level_train_config_items(
    document: Mapping[str, Any],
    *,
    strip_goal_owned: bool = False,
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    blocked = GOAL_OWNED_ENV_CONFIG_KEYS if strip_goal_owned else frozenset()
    for key in TRAIN_CONFIG_TOP_LEVEL_KEYS:
        if key in blocked:
            continue
        value = document.get(key)
        if _non_empty_config_value(value):
            items[key] = copy.deepcopy(value)
    return items


def _train_config_mapping_value(
    document: Mapping[str, Any],
    key: str,
    *,
    strip_goal_owned: bool = False,
) -> Mapping[str, Any] | None:
    value = document.get(key)
    if not isinstance(value, Mapping):
        return None
    if not strip_goal_owned:
        return value
    return _without_keys(
        value,
        GOAL_OWNED_ENV_CONFIG_KEYS | GOAL_OWNED_OBJECTIVE_CONFIG_KEYS,
    )


def _merge_train_config_sections(
    document: Mapping[str, Any],
    *,
    goal_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strip_goal_owned = goal_document is not None
    train_config: dict[str, Any] = _goal_train_defaults(goal_document or {})
    if not train_config:
        train_config = train_config_from_environment(_document_train_environment(document))
    for key in TRAIN_CONFIG_SECTION_KEYS:
        value = _train_config_section_value(document, key, strip_goal_owned=strip_goal_owned)
        if isinstance(value, Mapping):
            train_config = deep_merge(train_config, value)
    if strip_goal_owned:
        explicit_environment = _explicit_train_environment_config(document)
        if isinstance(explicit_environment, Mapping):
            train_config = deep_merge(train_config, explicit_environment)

    existing_train_config = _train_config_mapping_value(
        document,
        "train_config",
        strip_goal_owned=strip_goal_owned,
    )
    if isinstance(existing_train_config, Mapping):
        train_config = deep_merge(train_config, existing_train_config)

    train_config = deep_merge(
        train_config,
        _top_level_train_config_items(document, strip_goal_owned=strip_goal_owned),
    )

    overrides = document.get("overrides")
    if isinstance(overrides, Mapping):
        override_train_config = _train_config_mapping_value(
            overrides,
            "train_config",
            strip_goal_owned=strip_goal_owned,
        )
        if isinstance(override_train_config, Mapping):
            train_config = deep_merge(train_config, override_train_config)
        for key in TRAIN_CONFIG_SECTION_KEYS:
            value = _train_config_section_value(
                overrides,
                key,
                strip_goal_owned=strip_goal_owned,
            )
            if isinstance(value, Mapping):
                train_config = deep_merge(train_config, value)
        if strip_goal_owned:
            explicit_environment = _explicit_train_environment_config(overrides)
            if isinstance(explicit_environment, Mapping):
                train_config = deep_merge(train_config, explicit_environment)
        train_config = deep_merge(
            train_config,
            _top_level_train_config_items(overrides, strip_goal_owned=strip_goal_owned),
        )

    return train_config


def _infer_goal_slug_from_path(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "recipes" and index > 0:
            return parts[index - 1]
    for index, part in enumerate(parts):
        if part == "goals" and index + 1 < len(parts):
            next_part = parts[index + 1]
            if index + 2 < len(parts) and next_part in GOAL_GAME_DIR_NAMES:
                return parts[index + 2]
            return next_part
    return ""


def _goal_slug_from_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("goal_id") or value.get("goal") or value.get("goal_slug") or ""
        ).strip()
    return str(value or "").strip()


def _goal_slug_for_recipe(path: Path, document: Mapping[str, Any]) -> str:
    explicit = _goal_slug_from_value(document.get("goal")) or _goal_slug_from_value(
        document.get("goal_slug")
    )
    return explicit or _infer_goal_slug_from_path(path)


def _goal_composition_for_recipe(path: Path, document: Mapping[str, Any]) -> ComposedDocument | None:
    goal_slug = _goal_slug_for_recipe(path, document)
    if not goal_slug:
        return None
    inferred_path = path.resolve()
    if inferred_path.parent.name == "recipes":
        goal_dir = inferred_path.parent.parent
        for filename in ("_goal.yaml", "goal.yaml"):
            candidate = goal_dir / filename
            if candidate.is_file():
                return _load_rendered_goal_composition(candidate)
    for parent in inferred_path.parents:
        if parent.name == goal_slug:
            for filename in ("_goal.yaml", "goal.yaml"):
                candidate = parent / filename
                if candidate.is_file():
                    return _load_rendered_goal_composition(candidate)
    return None


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
    if goal_composition is None and path is not None:
        goal_composition = _goal_composition_for_recipe(path, materialized)
    if goal_composition is None:
        return None
    goal_document = goal_composition.document
    materialized["goal"] = copy.deepcopy(dict(goal_document))
    if "selection_policy" not in materialized:
        selection_policy = _selection_policy_from_goal(goal_document)
        if selection_policy is not None:
            materialized["selection_policy"] = copy.deepcopy(selection_policy)
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
    for key in ("group_id", "run_name_template"):
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


def load_recipe_document(path: Path, *, recipe_overrides: Sequence[str] = ()) -> dict[str, Any]:
    _reject_active_specs_path(path)
    recipe_override_list = [str(item).strip() for item in recipe_overrides if str(item).strip()]
    composed = load_composed_mapping(
        path,
        cycle_label="recipe",
        overrides=recipe_override_list,
    )
    document = render_template_vars(
        composed.document,
        path=path,
        label=f"recipe file {path}",
        deferred_fields_by_path=RECIPE_DEFERRED_TEMPLATE_FIELDS,
    )
    sources = list(composed.sources)
    goal_composition = _goal_composition_for_recipe(path, document)
    if goal_composition is not None:
        sources = [*goal_composition.sources, *sources]
    document = materialize_train_recipe_document(
        document,
        path=path,
        goal_composition=goal_composition,
    )
    document = attach_environment_identity(document)
    if recipe_override_list:
        document["recipe_overrides"] = recipe_override_list
    if path.suffix.lower() in YAML_EXTENSIONS or len(sources) > 1:
        document["_composition"] = {
            "root_path": str(path.resolve()),
            "source_files": _recipe_source_metadata(sources),
        }
    validate_train_recipe_schema(document, label=f"recipe file {path}")
    assert_no_template_vars(document, label=f"recipe file {path}")
    assert_no_secrets(document, label=f"recipe file {path}")
    validate_launch_event_config(
        document["train_config"],
        label=f"recipe file {path} train_config",
    )
    return document


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(args: Sequence[str], *, cwd: Path = Path(".")) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def repo_git_commit(cwd: Path = Path(".")) -> str | None:
    return _git_text(("rev-parse", "HEAD"), cwd=cwd)


def repo_is_dirty(cwd: Path = Path(".")) -> bool:
    text = _git_text(("status", "--porcelain"), cwd=cwd)
    return bool(text)


def recipe_slug(document: Mapping[str, Any]) -> str:
    return train_recipe_id(document)


def recipe_metadata(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    slug = recipe_slug(document)
    payload = dict(document)
    if slug and "recipe_id" not in payload:
        payload["recipe_id"] = slug
    digest = file_sha256(path)
    return {
        "goal_slug": recipe_goal_slug(document),
        "recipe_slug": slug,
        "recipe_path": str(path),
        "recipe_sha256": digest,
        "recipe_payload": payload,
        "repo_git_commit": repo_git_commit(),
        "repo_dirty": repo_is_dirty(),
    }


def _non_empty_config_value(value: Any) -> bool:
    return value not in (None, "", (), [], {})


def _configured_event_names(value: Any, *, label: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    elif isinstance(value, Sequence):
        names = tuple(str(name).strip() for name in value if str(name).strip())
    else:
        raise ValueError(f"{label} must be a comma-separated string or list")
    return tuple(dict.fromkeys(names))


def _configured_info_event_map(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError(f"{label} must define info_events as an object")


def _provider_owned_info_event_names(train_config: Mapping[str, Any]) -> frozenset[str]:
    provider_id = str(train_config.get("env_provider") or "").strip()
    return PROVIDER_OWNED_INFO_EVENTS.get(provider_id, frozenset())


def validate_launch_event_config(
    train_config: Mapping[str, Any], *, label: str = "train_config"
) -> None:
    legacy_keys = [
        key
        for key in LEGACY_EVENT_TRAIN_CONFIG_KEYS
        if _non_empty_config_value(train_config.get(key))
    ]
    if legacy_keys:
        raise ValueError(
            f"{label} uses legacy event key(s) {', '.join(legacy_keys)}; "
            "use info_events plus done_on_events for new launches"
        )
    done_event_names = _configured_event_names(
        train_config.get("done_on_events"),
        label=f"{label}.done_on_events",
    )
    if not done_event_names:
        return
    provider_owned_events = _provider_owned_info_event_names(train_config)
    info_events_value = train_config.get("info_events", train_config.get("info_events_json"))
    if _non_empty_config_value(info_events_value):
        info_events = _configured_info_event_map(info_events_value, label=label)
    else:
        info_events = {}
    missing = [
        name
        for name in done_event_names
        if name not in info_events and name not in provider_owned_events
    ]
    if missing:
        raise ValueError(
            f"{label}.done_on_events references unconfigured info event(s): {', '.join(missing)}"
        )


def validate_launch_seed_config(
    train_config: Mapping[str, Any],
    *,
    seed: int | None = None,
    label: str = "train_config",
) -> None:
    config_seed = train_config.get("seed")
    seed_span = train_config.get("n_envs", 1)
    if _non_empty_config_value(config_seed):
        validate_training_seed(config_seed, label=f"{label}.seed", seed_span=seed_span)
    if seed is not None:
        validate_training_seed(seed, label="seed", seed_span=seed_span)


def recipe_goal_slug(document: Mapping[str, Any]) -> str:
    return _goal_slug_from_value(document.get("goal")) or _goal_slug_from_value(
        document.get("goal_slug")
    )


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
