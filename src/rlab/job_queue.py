from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.config_loader import (
    YAML_EXTENSIONS,
    ComposedDocument,
    TEMPLATE_VARS_KEY,
    deep_merge,
    load_composed_mapping,
    render_template_vars,
)
from rlab.compute_targets import instance_defaults, load_json_file
from rlab.dotenv import load_env_file
from rlab.env_identity import attach_environment_identity, train_config_from_environment
from rlab.json_utils import json_safe
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    latest_runtime_image_ref,
    normalize_runtime_image_ref,
    runtime_image_ref_from_file,
)
from rlab.seeds import validate_training_seed
from rlab.spec_schema import (
    require_explicit_queue_train_config,
    train_recipe_id,
    validate_train_recipe_schema,
)


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
SPEC_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("description",): QUEUE_TEMPLATE_FIELDS,
    ("goal", "description"): QUEUE_TEMPLATE_FIELDS,
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
    **SPEC_DEFERRED_TEMPLATE_FIELDS,
    ("tags", "1"): frozenset({"slug", "recipe_id", "recipe_slug"}),
}
GOAL_OWNED_ENV_CONFIG_KEYS = frozenset(
    {
        "env_provider",
        "provider",
        "env_id",
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
        "early_stop_metric",
        "early_stop_operator",
        "early_stop_threshold",
    }
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS train_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_slug TEXT NOT NULL,
  recipe_slug TEXT,
  recipe_path TEXT,
  recipe_sha256 TEXT,
  repo_git_commit TEXT,
  repo_dirty BOOLEAN NOT NULL DEFAULT FALSE,
  recipe_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  runtime_image_ref TEXT NOT NULL,
  run_target TEXT,
  train_config JSONB NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 1,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  drain_requested BOOLEAN NOT NULL DEFAULT FALSE,
  run_name TEXT,
  run_description TEXT,
  seed INTEGER,
  wandb_group TEXT,
  wandb_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT
);

CREATE TABLE IF NOT EXISTS job_launches (
  id BIGSERIAL PRIMARY KEY,
  launch_id TEXT NOT NULL UNIQUE,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL,
  backend TEXT NOT NULL,
  machine TEXT NOT NULL,
  runtime_image_ref TEXT NOT NULL,
  container_name TEXT,
  provider_run_id TEXT,
  output_uri TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'launching',
  exit_code INTEGER,
  error TEXT,
  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  last_observed_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DROP INDEX IF EXISTS train_jobs_runtime_claim_idx;
DROP INDEX IF EXISTS train_jobs_claim_idx;
DROP INDEX IF EXISTS train_jobs_spec_status_idx;

ALTER TABLE train_jobs DROP COLUMN IF EXISTS priority;

CREATE INDEX IF NOT EXISTS train_jobs_claim_idx
  ON train_jobs (status, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS train_jobs_runtime_claim_idx
  ON train_jobs (runtime_image_ref, status, id)
  WHERE status IN ('pending', 'running') AND runtime_image_ref IS NOT NULL;

CREATE INDEX IF NOT EXISTS train_jobs_goal_status_idx
  ON train_jobs (goal_slug, status);

CREATE INDEX IF NOT EXISTS train_jobs_recipe_status_idx
  ON train_jobs (goal_slug, recipe_slug, status);

CREATE INDEX IF NOT EXISTS job_launches_machine_state_idx
  ON job_launches (machine, state, created_at);

CREATE INDEX IF NOT EXISTS job_launches_job_idx
  ON job_launches (job_kind, job_id, created_at DESC);

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events (job_kind, job_id, created_at DESC);
"""

RESET_TABLES = (
    "job_events",
    "job_launches",
    "train_jobs",
)


def json_arg(value: Any) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(value)


def database_url(use_direct: bool = False) -> str:
    load_env_file()
    if use_direct:
        value = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    else:
        value = (
            os.environ.get("TRAIN_QUEUE_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
            or os.environ.get("DIRECT_DATABASE_URL")
        )
    if not value:
        raise SystemExit(
            "TRAIN_QUEUE_DATABASE_URL, DATABASE_URL, or DIRECT_DATABASE_URL must be set"
        )
    return value


def normalize_run_target(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def canonicalize_run_target(
    value: str | None,
    *,
    instances_path: Path | None = None,
) -> str | None:
    target = normalize_run_target(value)
    if target is None:
        return None
    path = instances_path or Path("experiments/instances.yaml")
    if not path.is_file():
        return target
    return str(instance_defaults(load_json_file(path), target).get("name", target))


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def apply_schema(conn) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%(table_name)s) AS table_name", {"table_name": table_name})
        row = cur.fetchone()
    return bool(row and row.get("table_name"))


def export_existing_tables(conn, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "tables": [],
    }
    for table_name in RESET_TABLES:
        if not _table_exists(conn, table_name):
            continue
        path = export_dir / f"{table_name}.jsonl"
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} ORDER BY id")
            rows = [dict(row) for row in cur.fetchall()]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        manifest["tables"].append({"table": table_name, "rows": len(rows), "path": str(path)})
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_dir


def reset_schema(conn, *, export_dir: Path) -> Path:
    exported = export_existing_tables(conn, export_dir)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS
                  job_events,
                  job_launches,
                  train_jobs
                CASCADE
                """
            )
            cur.execute(SCHEMA_SQL)
    return exported


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
        if part in {"recipes", "specs"} and index > 0:
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
    if inferred_path.parent.name in {"recipes", "specs"}:
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


def _load_rendered_goal_composition(path: Path) -> ComposedDocument:
    composition = load_composed_mapping(path, cycle_label="goal")
    return ComposedDocument(
        document=render_template_vars(
            composition.document,
            path=path,
            label=f"goal file {path}",
            deferred_fields_by_path=GOAL_DEFERRED_TEMPLATE_FIELDS,
        ),
        sources=composition.sources,
    )


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
                materialized["tags"] = render_template_vars(
                    {"tags": materialized["tags"]},
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
        deferred_fields_by_path=SPEC_DEFERRED_TEMPLATE_FIELDS,
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
        "goal_slug": spec_goal_slug(document),
        "recipe_slug": slug,
        "recipe_path": str(path),
        "recipe_sha256": digest,
        "recipe_payload": payload,
        "repo_git_commit": repo_git_commit(),
        "repo_dirty": repo_is_dirty(),
    }


def record_job_event(
    conn,
    *,
    job_kind: str,
    job_id: int,
    event_type: str,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if job_kind != "train":
        raise ValueError(f"invalid job_kind: {job_kind}")
    metadata = dict(metadata or {})
    assert_no_secrets(metadata, label="event metadata")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO job_events (job_kind, job_id, event_type, message, metadata_json)
            VALUES (%(job_kind)s, %(job_id)s, %(event_type)s, %(message)s, %(metadata_json)s)
            """,
            {
                "job_kind": job_kind,
                "job_id": job_id,
                "event_type": event_type,
                "message": message,
                "metadata_json": json_arg(metadata),
            },
        )


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
    raise ValueError(f"{label} must define info_events_json as an object")


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
            "use info_events_json plus done_on_events for new launches"
        )
    done_event_names = _configured_event_names(
        train_config.get("done_on_events"),
        label=f"{label}.done_on_events",
    )
    if not done_event_names:
        return
    provider_owned_events = _provider_owned_info_event_names(train_config)
    info_events_value = train_config.get("info_events_json")
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


def spec_goal_slug(document: Mapping[str, Any]) -> str:
    return _goal_slug_from_value(document.get("goal")) or _goal_slug_from_value(
        document.get("goal_slug")
    )


def spec_tags(document: Mapping[str, Any]) -> list[str]:
    tags = []
    seen: set[str] = set()
    for raw_tag in document.get("tags") or []:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _format_seed_template(
    template: str | None, *, seed: int | None, recipe_id: str, utc: str, group_id: str = ""
) -> str | None:
    if not template:
        return None
    return str(template).format(
        seed="" if seed is None else seed,
        recipe_id=recipe_id,
        timestamp=utc,
        utc=utc,
        group_id=group_id,
    )


def _format_run_name_template(
    template: str | None,
    *,
    seed: int | None,
    recipe_id: str,
    utc: str,
    group_id: str,
) -> str | None:
    if not template:
        return None
    return str(template).format(
        seed="" if seed is None else seed,
        recipe_id=recipe_id,
        timestamp=utc,
        utc=utc,
        group_id=group_id,
    )


def _run_name_slug(value: str, *, limit: int = 32) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "run"
    return slug[:limit].strip("-") or "run"


def _run_name_batch_id(group_id: str) -> str:
    group = _run_name_slug(group_id)
    match = re.match(r"^(b\d+)(?:-|$)", group)
    return match.group(1) if match else group


def _format_default_run_name(
    group_id: str,
    *,
    label: str,
    seed: int | None,
    utc: str,
) -> str:
    batch_id = _run_name_batch_id(group_id)
    description = _run_name_slug(label, limit=24)
    seed_label = f"s{seed}" if seed is not None else "s"
    return f"{batch_id}-{description}-{seed_label}-{utc}"


def _document_seeds(
    document: Mapping[str, Any], override_seeds: Sequence[int] = ()
) -> list[int | None]:
    if override_seeds:
        return [int(seed) for seed in override_seeds]
    seeds = document.get("seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, str):
        return [int(seed) for seed in seeds]
    train_config = document.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("seed") is not None:
        return [int(train_config["seed"])]
    return [None]


def enqueue_train_jobs_from_recipe_document(
    conn,
    *,
    document: Mapping[str, Any],
    runtime_image_ref: str,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    seeds: Sequence[int] = (),
) -> list[dict[str, Any]]:
    validate_train_recipe_schema(document)
    goal_slug = spec_goal_slug(document)
    document_slug = recipe_slug(document)
    utc = _utc_stamp()
    rows = []
    for seed in _document_seeds(document, seeds):
        train_config = dict(document["train_config"])
        recipe_overrides = document.get("recipe_overrides")
        if isinstance(recipe_overrides, Sequence) and not isinstance(recipe_overrides, str | bytes):
            train_config["recipe_overrides"] = [str(item) for item in recipe_overrides]
        if seed is not None:
            validate_training_seed(
                seed,
                label="recipe seed",
                seed_span=train_config.get("n_envs", 1),
            )
            train_config["seed"] = seed
        train_config.setdefault("recipe_slug", document_slug)
        if recipe_path:
            train_config.setdefault("recipe_path", recipe_path)
        group_id = str(document["group_id"])
        payload = dict(document)
        payload.setdefault("recipe_id", document_slug)
        row = enqueue_train_job(
            conn,
            goal_slug=goal_slug,
            recipe_slug=document_slug,
            recipe_path=recipe_path,
            recipe_sha256=recipe_sha256,
            repo_git_commit=repo_git_commit,
            repo_dirty=repo_dirty,
            recipe_payload=payload,
            runtime_image_ref=runtime_image_ref,
            train_config=train_config,
            max_attempts=int(document.get("max_attempts") or 1),
            run_name=(
                _format_run_name_template(
                    document.get("run_name_template"),
                    seed=seed,
                    recipe_id=document_slug,
                    utc=utc,
                    group_id=group_id,
                )
                or _format_default_run_name(
                    group_id,
                    label=str(document.get("run_name_label") or document_slug),
                    seed=seed,
                    utc=utc,
                )
            ),
            run_description=_format_seed_template(
                document.get("description"),
                seed=seed,
                recipe_id=document_slug,
                utc=utc,
                group_id=group_id,
            ),
            seed=seed,
            wandb_group=group_id,
            wandb_tags=spec_tags(document),
        )
        rows.append(row)
    return rows


def enqueue_train_jobs_from_recipe_file(
    conn,
    *,
    path: Path,
    runtime_image_ref: str,
    seeds: Sequence[int] = (),
    recipe_overrides: Sequence[str] = (),
) -> list[dict[str, Any]]:
    document = load_recipe_document(path, recipe_overrides=recipe_overrides)
    metadata = recipe_metadata(path, document)
    return enqueue_train_jobs_from_recipe_document(
        conn,
        document=document,
        runtime_image_ref=runtime_image_ref,
        recipe_path=metadata["recipe_path"],
        recipe_sha256=metadata["recipe_sha256"],
        repo_git_commit=metadata["repo_git_commit"],
        repo_dirty=metadata["repo_dirty"],
        seeds=seeds,
    )


def enqueue_train_job(
    conn,
    *,
    goal_slug: str,
    recipe_slug: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    recipe_payload: Mapping[str, Any] | None = None,
    runtime_image_ref: str,
    train_config: Mapping[str, Any],
    run_target: str | None = None,
    max_attempts: int = 1,
    run_name: str | None = None,
    run_description: str | None = None,
    seed: int | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] = (),
) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    if not goal_slug:
        raise ValueError("goal_slug is required")
    config = dict(train_config)
    assert_no_secrets(config, label="train_config")
    assert_no_secrets(recipe_payload or {}, label="recipe_payload")
    require_explicit_queue_train_config(config)
    validate_launch_seed_config(config, seed=seed)
    validate_launch_event_config(config)
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    run_target = normalize_run_target(run_target)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_slug, recipe_slug, recipe_path, recipe_sha256, repo_git_commit,
                  repo_dirty, recipe_payload_json, runtime_image_ref,
                  run_target, train_config, max_attempts, run_name,
                  run_description, seed, wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_slug)s, %(recipe_slug)s, %(recipe_path)s, %(recipe_sha256)s,
                  %(repo_git_commit)s, %(repo_dirty)s, %(recipe_payload_json)s,
                  %(runtime_image_ref)s, %(run_target)s,
                  %(train_config)s, %(max_attempts)s, %(run_name)s,
                  %(run_description)s, %(seed)s, %(wandb_group)s, %(wandb_tags)s
                )
                RETURNING *
                """,
                {
                    "goal_slug": goal_slug,
                    "recipe_slug": recipe_slug,
                    "recipe_path": recipe_path,
                    "recipe_sha256": recipe_sha256,
                    "repo_git_commit": repo_git_commit,
                    "repo_dirty": bool(repo_dirty),
                    "recipe_payload_json": json_arg(dict(recipe_payload or {})),
                    "runtime_image_ref": runtime_image_ref,
                    "run_target": run_target,
                    "train_config": json_arg(config),
                    "max_attempts": max_attempts,
                    "run_name": run_name,
                    "run_description": run_description,
                    "seed": seed,
                    "wandb_group": wandb_group,
                    "wandb_tags": list(wandb_tags),
                },
            )
            row = dict(cur.fetchone())
            record_job_event(
                conn,
                job_kind="train",
                job_id=int(row["id"]),
                event_type="enqueued",
                message="train job enqueued",
                metadata={"goal_slug": goal_slug, "recipe_slug": recipe_slug},
            )
            return row


def new_launch_id(job_kind: str, job_id: int | None = None) -> str:
    suffix = uuid.uuid4().hex[:12]
    if job_id is None:
        return f"{job_kind}-{suffix}"
    return f"{job_kind}-{int(job_id)}-{suffix}"


def _job_table(job_kind: str) -> str:
    if job_kind == "train":
        return "train_jobs"
    raise ValueError(f"invalid job_kind: {job_kind}")


def _job_event_kind(job_kind: str) -> str:
    if job_kind != "train":
        raise ValueError(f"invalid job_kind: {job_kind}")
    return job_kind


def claim_job_launch(
    conn,
    *,
    job_kind: str,
    machine: str,
    backend: str,
    runtime_image_ref: str | None = None,
    run_target: str | None = None,
    job_id: int | None = None,
    launch_id: str | None = None,
    container_name: str | None = None,
    output_uri: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    job_kind = _job_event_kind(job_kind)
    table = _job_table(job_kind)
    runtime_image_ref = (
        normalize_runtime_image_ref(runtime_image_ref) if runtime_image_ref else None
    )
    filters = ["cancel_requested = FALSE", "status = 'pending'"]
    params: dict[str, Any] = {
        "machine": str(machine),
        "backend": str(backend),
        "output_uri": str(output_uri),
        "launch_id": launch_id or new_launch_id(job_kind, job_id),
        "container_name": container_name,
        "job_kind": job_kind,
    }
    if job_id is not None:
        filters.append("id = %(job_id)s")
        params["job_id"] = int(job_id)
    if runtime_image_ref is not None:
        filters.append("runtime_image_ref = %(runtime_image_ref)s")
        params["runtime_image_ref"] = runtime_image_ref
    if job_kind == "train" and run_target is not None:
        filters.append("run_target = %(run_target)s")
        params["run_target"] = normalize_run_target(run_target)
    where = "\n    AND ".join(filters)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH next_job AS (
                  SELECT *
                  FROM {table}
                  WHERE {where}
                  ORDER BY id ASC
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                ),
                updated AS (
                  UPDATE {table} AS job
                  SET status = 'launching',
                      lease_owner = %(launch_id)s,
                      lease_expires_at = NULL,
                      heartbeat_at = now(),
                      error = NULL
                  FROM next_job
                  WHERE job.id = next_job.id
                  RETURNING job.*
                ),
                inserted_launch AS (
                  INSERT INTO job_launches (
                    launch_id, job_kind, job_id, backend, machine, runtime_image_ref,
                    container_name, output_uri, state, last_observed_at
                  )
                  SELECT
                    %(launch_id)s, %(job_kind)s, updated.id, %(backend)s, %(machine)s,
                    updated.runtime_image_ref, %(container_name)s, %(output_uri)s,
                    'launching', now()
                  FROM updated
                  RETURNING *
                )
                SELECT
                  row_to_json(updated) AS job_json,
                  row_to_json(inserted_launch) AS launch_json
                FROM updated, inserted_launch
                """,
                params,
            )
            row = cur.fetchone()
            if not row:
                return None
            job = dict(row["job_json"])
            launch = dict(row["launch_json"])
            record_job_event(
                conn,
                job_kind=job_kind,
                job_id=int(job["id"]),
                event_type="launching",
                message=f"job launch claimed on {machine}",
                metadata={"launch_id": launch["launch_id"], "machine": machine, "backend": backend},
            )
            return job, launch


def mark_job_launch_running(
    conn,
    *,
    launch_id: str,
    container_name: str | None = None,
    provider_run_id: str | None = None,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches AS launch
                SET state = 'running',
                    container_name = COALESCE(%(container_name)s, container_name),
                    provider_run_id = COALESCE(%(provider_run_id)s, provider_run_id),
                    started_at = COALESCE(started_at, now()),
                    last_observed_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {
                    "launch_id": launch_id,
                    "container_name": container_name,
                    "provider_run_id": provider_run_id,
                },
            )
            launch = cur.fetchone()
            if not launch:
                return None
            table = _job_table(str(launch["job_kind"]))
            cur.execute(
                f"""
                UPDATE {table}
                SET status = 'running',
                    started_at = COALESCE(started_at, now()),
                    heartbeat_at = now(),
                    attempts = attempts + 1
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status = 'launching'
                """,
                {"job_id": launch["job_id"], "launch_id": launch_id},
            )
            record_job_event(
                conn,
                job_kind=str(launch["job_kind"]),
                job_id=int(launch["job_id"]),
                event_type="running",
                message="job container started",
                metadata={"launch_id": launch_id},
            )
            return dict(launch)


def release_job_launch(
    conn,
    *,
    launch_id: str,
    error: str | None = None,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches
                SET state = 'released',
                    error = %(error)s,
                    last_observed_at = now(),
                    finished_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {"launch_id": launch_id, "error": error},
            )
            launch = cur.fetchone()
            if not launch:
                return None
            table = _job_table(str(launch["job_kind"]))
            cur.execute(
                f"""
                UPDATE {table}
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status = 'launching'
                """,
                {
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                    "error": error,
                },
            )
            record_job_event(
                conn,
                job_kind=str(launch["job_kind"]),
                job_id=int(launch["job_id"]),
                event_type="released",
                message=error,
                metadata={"launch_id": launch_id},
            )
            return dict(launch)


def active_job_launches(
    conn,
    *,
    machine: str | None = None,
    states: Sequence[str] = ("launching", "running"),
) -> list[dict[str, Any]]:
    filters = ["state = ANY(%(states)s)"]
    params: dict[str, Any] = {"states": list(states)}
    if machine:
        filters.append("machine = %(machine)s")
        params["machine"] = machine
    where = "\n    AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM job_launches
            WHERE {where}
            ORDER BY created_at ASC, id ASC
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def job_payload_for_launch(job: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "job_kind": launch["job_kind"],
        "job": dict(job),
        "launch_id": launch["launch_id"],
        "machine": launch["machine"],
        "backend": launch["backend"],
        "runtime_image_ref": launch["runtime_image_ref"],
        "output_uri": launch["output_uri"],
    }
    assert_no_secrets(payload, label="job payload")
    return json_safe(payload)


def request_cancel_train_job(conn, *, job_id: int) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET cancel_requested = TRUE,
                    status = CASE WHEN status IN ('pending', 'launching') THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status IN ('pending', 'launching') THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'launching', 'running')
                """,
                {"job_id": job_id},
            )
            return int(cur.rowcount)


def _normalize_positive_ids(values: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(value) for value in values)
    invalid = [value for value in ids if value <= 0]
    if invalid:
        raise ValueError(f"job ids must be positive integers: {invalid}")
    return ids


def _normalize_stale_limit(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _terminal_status_from_result(result: Mapping[str, Any]) -> str:
    status = str(result.get("status") or "").strip()
    if status in {"succeeded", "failed", "canceled"}:
        return status
    exit_code = result.get("exit_code")
    return "succeeded" if exit_code == 0 else "failed"


def _strip_metric_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_metric_payloads(nested)
            for key, nested in value.items()
            if key != "metrics_json"
        }
    if isinstance(value, list):
        return [_strip_metric_payloads(item) for item in value]
    return value


def launch_result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep launch bookkeeping useful without mirroring W&B metrics in Postgres."""

    return json_safe(_strip_metric_payloads(dict(result)))


def finish_train_launch_from_result(
    conn,
    *,
    launch_id: str,
    result: Mapping[str, Any],
) -> None:
    status = _terminal_status_from_result(result)
    exit_code = result.get("exit_code")
    error = str(result.get("error") or "") or None
    train_result = result.get("train")
    train_payload = train_result.get("result") if isinstance(train_result, Mapping) else {}
    train_payload = dict(train_payload or {})
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches
                SET state = %(state)s,
                    exit_code = %(exit_code)s,
                    error = %(error)s,
                    result_json = %(result_json)s,
                    last_observed_at = now(),
                    finished_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {
                    "state": status,
                    "exit_code": exit_code,
                    "error": error,
                    "result_json": json_arg(launch_result_metadata(result)),
                    "launch_id": launch_id,
                },
            )
            launch = cur.fetchone()
            if not launch:
                raise RuntimeError(f"unknown launch_id {launch_id}")
            if launch["job_kind"] != "train":
                raise RuntimeError(f"launch {launch_id} is not a train launch")
            cur.execute(
                """
                UPDATE train_jobs
                SET status = %(status)s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = now(),
                    finished_at = now(),
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status IN ('launching', 'running')
                RETURNING *
                """,
                {
                    "status": status,
                    "error": error,
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                },
            )
            job = cur.fetchone()
            if not job:
                raise RuntimeError(f"could not finish train job for launch {launch_id}")
            record_job_event(
                conn,
                job_kind="train",
                job_id=int(job["id"]),
                event_type=status,
                message=error,
                metadata={
                    "launch_id": launch_id,
                    "exit_code": exit_code,
                    "run_name": train_payload.get("run_name") or job.get("run_name"),
                    "wandb_run_id": train_payload.get("wandb_run_id"),
                    "wandb_url": train_payload.get("wandb_url"),
                },
            )


def finish_job_launch_from_result(
    conn,
    *,
    launch_id: str,
    result: Mapping[str, Any],
) -> None:
    job_kind = str(result.get("job_kind") or "")
    if job_kind == "train":
        finish_train_launch_from_result(conn, launch_id=launch_id, result=result)
    else:
        raise ValueError(f"result does not identify train job kind: {job_kind!r}")


def _metric_float(metrics: Mapping[str, Any], key: str, default: float = float("-inf")) -> float:
    value = metrics.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


COMPLETION_GOAL_RATE = 0.99


def eval_selection_score(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Eval-first policy ranking: completion, solved timestep, then reward."""

    completion_min = _metric_float(
        metrics,
        "eval/done/level_change/from_rate/min",
        default=_metric_float(
            metrics,
            "eval/done/level_change/rate",
            default=_metric_float(metrics, "completion_rate"),
        ),
    )
    completion_mean = _metric_float(
        metrics,
        "eval/done/level_change/from_rate/mean",
        default=_metric_float(
            metrics,
            "eval/done/level_change/rate",
            default=_metric_float(metrics, "completion_rate"),
        ),
    )
    checkpoint_step = _metric_float(metrics, "checkpoint_step")
    steps_to_goal = (
        checkpoint_step
        if completion_min >= COMPLETION_GOAL_RATE and checkpoint_step > float("-inf")
        else float("inf")
    )
    return (
        completion_min,
        completion_mean,
        -steps_to_goal,
        _metric_float(metrics, "reward_mean"),
    )


def queue_status(conn, *, goal_slug: str) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            WHERE goal_slug = %(goal_slug)s
            GROUP BY status
            ORDER BY status
            """,
            {"goal_slug": goal_slug},
        )
        train_jobs = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT id, goal_slug, recipe_slug, status, run_name,
                   run_target, lease_owner, heartbeat_at, created_at
            FROM train_jobs
            WHERE goal_slug = %(goal_slug)s
              AND status IN ('pending', 'launching', 'running')
            ORDER BY
              CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
              id ASC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        active_train_jobs = [dict(row) for row in cur.fetchall()]
    return {
        "goal_slug": goal_slug,
        "train_jobs": train_jobs,
        "active_train_jobs": active_train_jobs,
    }


def print_status(report: Mapping[str, Any]) -> None:
    print(f"goal: {report['goal_slug']}")
    print(f"train_jobs: {json.dumps(report['train_jobs'], sort_keys=True)}")
    print("active_train_jobs:")
    for row in report.get("active_train_jobs", []):
        print(
            "  "
            f"job={row['id']} status={row['status']} image={row.get('runtime_image_ref') or ''} "
            f"run={row.get('run_name') or ''}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage rlab train job queues.")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create queue tables")
    setup.set_defaults(func=cmd_setup)

    reset = subparsers.add_parser(
        "reset-schema",
        help="Export old queue tables, then drop and recreate the queue schema.",
    )
    reset.add_argument(
        "--export-dir",
        type=Path,
        help="Directory for JSONL exports; defaults to logs/campaign-db-export-<utc>.",
    )
    add_dry_run_arg(reset)
    reset.set_defaults(func=cmd_reset_schema)

    cancel = subparsers.add_parser("cancel-train", help="Request cancellation for a train job")
    cancel.add_argument("job_id", type=int)
    cancel.set_defaults(func=cmd_cancel_train)

    status = subparsers.add_parser("status", help="Print compact queue status")
    status.add_argument("--goal", required=True, dest="goal_slug")
    status.set_defaults(func=cmd_status)
    return parser


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Preview planned changes without applying them.",
    )


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(args.direct))


def runtime_image_ref_from_args(
    args: argparse.Namespace, *, default_latest: bool = False
) -> str | None:
    if getattr(args, "runtime_image_ref_file", None):
        return runtime_image_ref_from_file(args.runtime_image_ref_file)
    if getattr(args, "runtime_image_ref", None):
        return normalize_runtime_image_ref(args.runtime_image_ref)
    if default_latest or getattr(args, "latest_image", False):
        return latest_runtime_image_ref(
            workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
            branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
            artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
        )
    return None


def cmd_setup(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        apply_schema(conn)
    finally:
        conn.close()
    print("queue_schema=ok")
    return 0


def default_export_dir() -> Path:
    return Path("logs") / f"campaign-db-export-{_utc_stamp()}"


def cmd_reset_schema(args: argparse.Namespace) -> int:
    export_dir = args.export_dir or default_export_dir()
    if not args.execute:
        print(f"dry_run: would export queue tables to {export_dir} and reset schema")
        print("dry_run: rerun without --dry-run to apply")
        return 0
    conn = _connect_from_args(args)
    try:
        exported = reset_schema(conn, export_dir=export_dir)
    finally:
        conn.close()
    print(f"queue_schema_reset=ok export_dir={exported}")
    return 0


def cmd_enqueue_train(args: argparse.Namespace) -> int:
    runtime_image_ref = runtime_image_ref_from_args(args, default_latest=True)
    if not runtime_image_ref:
        raise SystemExit(
            "--runtime-image-ref, --runtime-image-ref-file, or latest image resolution is required"
        )
    conn = _connect_from_args(args)
    try:
        rows = enqueue_train_jobs_from_recipe_file(
            conn,
            path=args.recipe_file,
            runtime_image_ref=runtime_image_ref,
            seeds=args.seed,
            recipe_overrides=args.recipe_overrides,
        )
    finally:
        conn.close()
    for row in rows:
        print(
            f"train_job_id={row['id']} image={row.get('runtime_image_ref') or ''} "
            f"run_name={row.get('run_name') or ''}"
        )
    return 0


def cmd_cancel_train(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        count = request_cancel_train_job(conn, job_id=args.job_id)
    finally:
        conn.close()
    print(f"cancel_requested={count}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        report = queue_status(
            conn,
            goal_slug=args.goal_slug,
        )
    finally:
        conn.close()
    print_status(report)
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
