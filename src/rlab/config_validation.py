from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlab.benchmark_profiles import load_benchmark_profiles
from rlab.compute_targets import load_instance_config
from rlab.config_loader import load_composed_mapping, load_mapping_document, render_template_vars
from rlab.early_stop import normalize_early_stop_config
from rlab.env import EnvConfig
from rlab.env_config_aliases import STABLE_RETRO_TURBO_ENV_CONFIG_KEYS
from rlab.env_wrappers import normalize_env_wrapper_specs
from rlab.env_registry import qualify_env_id, resolve_env_id
from rlab.fleet import load_capacity_policy, load_fleet_config, validate_capacity_policy
from rlab.job_queue import load_spec_document
from rlab.seeds import validate_eval_seed


BENCHMARK_BASELINES_SCHEMA_VERSION = 1
ENV_CONFIG_ALLOWED_KEYS = frozenset(EnvConfig.__dataclass_fields__) | {"env_provider"}
ENV_CONFIG_ALLOWED_KEYS = ENV_CONFIG_ALLOWED_KEYS | STABLE_RETRO_TURBO_ENV_CONFIG_KEYS | {"n_envs"}
GOAL_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    (
        "release",
        "huggingface",
        "checkpoint_filename",
    ): frozenset({"checkpoint_step"}),
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": dict(sorted(self.counts.items())),
            "issues": [issue.to_json() for issue in self.issues],
        }


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{_label_path(label, key)} is required")
    return document[key]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_non_empty_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _require_key(document, key, label=label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    return value.strip()


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


def _require_number(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    value = _require_key(document, key, label=label)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum:g}")
    return number


def _require_bool(document: Mapping[str, Any], key: str, *, label: str) -> bool:
    value = _require_key(document, key, label=label)
    if not isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a boolean")
    return value


def _require_string_list(document: Mapping[str, Any], key: str, *, label: str) -> list[str]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-empty string")
        result.append(item.strip())
    return result


def _require_int_list(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    length: int,
    minimum: int | None = None,
) -> list[int]:
    value = _require_key(document, key, label=label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if len(value) != length:
        raise ValueError(f"{_label_path(label, key)} must contain {length} integers")
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item):
            raise ValueError(f"{_label_path(label, key)}[{index}] must be an integer")
        if minimum is not None and item < minimum:
            raise ValueError(f"{_label_path(label, key)}[{index}] must be >= {minimum}")
        result.append(item)
    return result


def _validate_obs_crop(preprocessing: Mapping[str, Any], *, label: str) -> None:
    if "hud_crop_top" in preprocessing:
        raise ValueError(f"{label}.hud_crop_top is redundant; use obs_crop")
    if "obs_crop" not in preprocessing:
        raise ValueError(f"{label}.obs_crop is required")
    value = preprocessing["obs_crop"]
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{label}.obs_crop must be [top, right, bottom, left]")
    for index, item in enumerate(value):
        if not _is_int(item) or item < 0:
            raise ValueError(f"{label}.obs_crop[{index}] must be a non-negative integer")


def _validate_obs_resize(preprocessing: Mapping[str, Any], *, label: str) -> None:
    if "observation_size" in preprocessing:
        raise ValueError(f"{label}.observation_size is redundant; use obs_resize")
    if "obs_resize" not in preprocessing:
        raise ValueError(f"{label}.obs_resize is required")
    value = preprocessing["obs_resize"]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise ValueError(f"{label}.obs_resize must be [height, width]")
    for index, item in enumerate(value):
        if not _is_int(item) or item <= 0:
            raise ValueError(f"{label}.obs_resize[{index}] must be a positive integer")


def _require_schema_version(document: Mapping[str, Any], expected: int, *, label: str) -> None:
    schema_version = _require_int(document, "schema_version", label=label, minimum=1)
    if schema_version != expected:
        raise ValueError(f"{_label_path(label, 'schema_version')} must be {expected}")


def _validate_environment_identity(
    document: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    environment = _require_mapping(
        _require_key(document, "environment", label=label),
        label=f"{label}.environment",
    )
    env_config = environment.get("env_config")
    if isinstance(env_config, Mapping):
        _validate_environment_env_config(
            environment,
            env_config,
            label=f"{label}.environment",
            require_game=True,
        )
        return environment

    for old_key in ("provider", "env_provider", "provider_env_id"):
        if old_key in environment:
            raise ValueError(
                f"{label}.environment.{old_key} was replaced by fully-qualified env_id"
            )
    env_id = _require_non_empty_string(environment, "env_id", label=f"{label}.environment")
    try:
        resolve_env_id(env_id)
    except ValueError as exc:
        raise ValueError(f"{label}.environment.env_id is invalid: {exc}") from exc
    action = _require_mapping(
        _require_key(environment, "action", label=f"{label}.environment"),
        label=f"{label}.environment.action",
    )
    _require_non_empty_string(action, "action_set", label=f"{label}.environment.action")
    preprocessing = _require_mapping(
        _require_key(environment, "preprocessing", label=f"{label}.environment"),
        label=f"{label}.environment.preprocessing",
    )
    _validate_obs_crop(preprocessing, label=f"{label}.environment.preprocessing")
    _validate_obs_resize(preprocessing, label=f"{label}.environment.preprocessing")
    _require_mapping(
        _require_key(environment, "termination", label=f"{label}.environment"),
        label=f"{label}.environment.termination",
    )
    return environment


def _validate_environment_env_config(
    environment: Mapping[str, Any],
    env_config: Mapping[str, Any],
    *,
    label: str,
    require_game: bool,
    allowed_extra_keys: set[str] | None = None,
) -> None:
    combined = dict(env_config)
    if "env_provider" in environment and "env_provider" not in combined:
        combined["env_provider"] = environment["env_provider"]
    _validate_env_config(
        combined,
        label=f"{label}.env_config",
        require_game=require_game,
        allowed_extra_keys=allowed_extra_keys,
    )


def _validate_env_config(
    env_config: Mapping[str, Any],
    *,
    label: str,
    require_game: bool,
    require_provider: bool = True,
    allowed_extra_keys: set[str] | None = None,
) -> None:
    extra_keys = sorted(set(env_config) - ENV_CONFIG_ALLOWED_KEYS - (allowed_extra_keys or set()))
    if extra_keys:
        raise ValueError(f"{label} has non-EnvConfig key(s): {extra_keys}")
    if require_provider:
        env_provider = _require_non_empty_string(env_config, "env_provider", label=label)
    elif "env_provider" in env_config:
        env_provider = _require_non_empty_string(env_config, "env_provider", label=label)
    else:
        env_provider = None
    if require_game:
        game = _require_non_empty_string(env_config, "game", label=label)
    elif "game" in env_config:
        game = _require_non_empty_string(env_config, "game", label=label)
    else:
        game = None
    if game and env_provider:
        try:
            qualify_env_id(env_provider, game)
        except ValueError as exc:
            raise ValueError(f"{label}.env_provider is invalid: {exc}") from exc
    if "state" in env_config and "states" in env_config:
        raise ValueError(f"{label} must define only one of state or states")
    if "state" in env_config:
        state = env_config["state"]
        if not isinstance(state, str) or not state.strip():
            raise ValueError(f"{label}.state must be a non-empty string")
    if "states" in env_config:
        _require_string_list(env_config, "states", label=label)
    if "action_set" in env_config:
        _require_non_empty_string(env_config, "action_set", label=label)
    for key in (
        "frame_skip",
        "max_episode_steps",
        "observation_size",
        "hud_crop_top",
        "no_progress_timeout_steps",
        "no_progress_min_delta",
    ):
        if key in env_config:
            minimum = 1 if key == "frame_skip" else 0
            _require_int(env_config, key, label=label, minimum=minimum)
    if "n_envs" in env_config:
        _require_int(env_config, "n_envs", label=label, minimum=1)
    if "env_threads" in env_config:
        _require_int(env_config, "env_threads", label=label, minimum=0)
    if "max_pool_frames" in env_config:
        _require_bool(env_config, "max_pool_frames", label=label)
    if "sticky_action_prob" in env_config:
        _require_number(env_config, "sticky_action_prob", label=label)
    if "obs_resize_algorithm" in env_config:
        _require_non_empty_string(env_config, "obs_resize_algorithm", label=label)
    if "info_events" in env_config:
        _require_mapping(env_config["info_events"], label=f"{label}.info_events")
    if "info_events_json" in env_config:
        _require_mapping(env_config["info_events_json"], label=f"{label}.info_events_json")
    if "done_on_events" in env_config:
        _require_string_list(env_config, "done_on_events", label=label)
    if "frame_stack" in env_config:
        _require_int(env_config, "frame_stack", label=label, minimum=1)
    if "maxpool_last_two" in env_config:
        _require_bool(env_config, "maxpool_last_two", label=label)
    if "frame_maxpool" in env_config:
        _require_bool(env_config, "frame_maxpool", label=label)
    if "action_sticky_prob" in env_config:
        _require_number(env_config, "action_sticky_prob", label=label)
    if "obs_resize" in env_config:
        _require_int_list(env_config, "obs_resize", label=label, length=2, minimum=1)
    if "obs_crop" in env_config:
        _require_int_list(env_config, "obs_crop", label=label, length=4, minimum=0)
    if "obs_grayscale" in env_config:
        _require_bool(env_config, "obs_grayscale", label=label)
    if "obs_layout" in env_config:
        obs_layout = _require_non_empty_string(env_config, "obs_layout", label=label)
        if obs_layout not in {"hwc", "chw"}:
            raise ValueError(f"{label}.obs_layout must be hwc or chw")
    if "obs_copy" in env_config:
        obs_copy = _require_non_empty_string(env_config, "obs_copy", label=label)
        if obs_copy not in {"copy", "safe_view", "unsafe_view"}:
            raise ValueError(f"{label}.obs_copy must be copy, safe_view, or unsafe_view")
    if "noop_reset_max" in env_config:
        _require_int(env_config, "noop_reset_max", label=label, minimum=0)
    if "reset_noops" in env_config:
        _require_int(env_config, "reset_noops", label=label, minimum=0)
    if "reward_clip" in env_config:
        _require_bool(env_config, "reward_clip", label=label)
    if "info_filter" in env_config and not isinstance(env_config["info_filter"], str | Mapping):
        raise ValueError(f"{label}.info_filter must be a string or mapping")
    if "done_on" in env_config:
        if isinstance(env_config["done_on"], Mapping):
            _require_mapping(env_config["done_on"], label=f"{label}.done_on")
        else:
            _require_string_list(env_config, "done_on", label=label)
    if "env_wrappers" in env_config:
        normalize_env_wrapper_specs(env_config["env_wrappers"], label=f"{label}.env_wrappers")


def _goal_train_section(document: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    return _require_mapping(
        _require_key(document, "train", label=label),
        label=f"{label}.train",
    )


def _goal_train_environment(
    document: Mapping[str, Any],
    train: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    if "environment" in train:
        return _require_mapping(train["environment"], label=f"{label}.train.environment")
    return _require_mapping(
        _require_key(document, "environment", label=label),
        label=f"{label}.environment",
    )


def _validate_goal_eval(document: Mapping[str, Any], *, label: str) -> None:
    if "eval_spec" in document:
        raise ValueError(f"{label}.eval_spec moved to eval")
    eval_section = _require_mapping(
        _require_key(document, "eval", label=label),
        label=f"{label}.eval",
    )
    eval_environment = eval_section.get("environment")
    if isinstance(eval_environment, Mapping):
        eval_environment_keys = {"env_provider", "env_config"}
        extra_keys = sorted(set(eval_environment) - eval_environment_keys)
        if extra_keys:
            raise ValueError(f"{label}.eval.environment has unexpected keys: {extra_keys}")
        eval_env_config = _require_mapping(
            _require_key(eval_environment, "env_config", label=f"{label}.eval.environment"),
            label=f"{label}.eval.environment.env_config",
        )
        _validate_environment_env_config(
            eval_environment,
            eval_env_config,
            label=f"{label}.eval.environment",
            require_game=True,
            allowed_extra_keys={"max_episodes", "seed", "n_envs", "max_steps"},
        )
        _require_int(
            eval_env_config,
            "max_episodes",
            label=f"{label}.eval.environment.env_config",
            minimum=1,
        )
        if "seed" in eval_env_config:
            seed = _require_int(
                eval_env_config,
                "seed",
                label=f"{label}.eval.environment.env_config",
            )
            validate_eval_seed(seed, label=f"{label}.eval.environment.env_config.seed")
        if "n_envs" in eval_env_config and "num_envs" in eval_env_config:
            raise ValueError(
                f"{label}.eval.environment.env_config must define only one of n_envs or num_envs"
            )
        if "n_envs" in eval_env_config or "num_envs" in eval_env_config:
            n_envs_key = "num_envs" if "num_envs" in eval_env_config else "n_envs"
            _require_int(
                eval_env_config, n_envs_key, label=f"{label}.eval.environment.env_config", minimum=1
            )
        if "max_steps" in eval_env_config:
            _require_int(
                eval_env_config,
                "max_steps",
                label=f"{label}.eval.environment.env_config",
                minimum=1,
            )
    elif "env_config" in eval_section:
        eval_env_config = _require_mapping(
            eval_section["env_config"],
            label=f"{label}.eval.env_config",
        )
        _validate_env_config(
            eval_env_config,
            label=f"{label}.eval.env_config",
            require_game=False,
        )
    if "eval_config" in eval_section:
        raise ValueError(f"{label}.eval.eval_config moved to eval.policy")
    if "eval" in eval_section:
        raise ValueError(f"{label}.eval.eval moved to eval.policy")
    policy = eval_section.get("policy")
    if isinstance(policy, Mapping):
        for moved_key in ("max_episodes", "seed", "n_envs", "max_steps"):
            if moved_key in policy:
                raise ValueError(
                    f"{label}.eval.policy.{moved_key} moved to "
                    f"{label}.eval.environment.env_config.{moved_key}"
                )
        if "stochastic" in policy:
            _require_bool(policy, "stochastic", label=f"{label}.eval.policy")
        if "done_on_events" in policy:
            _require_string_list(policy, "done_on_events", label=f"{label}.eval.policy")
    elif policy is not None:
        raise ValueError(f"{label}.eval.policy must be an object")


def _validate_rank_order(rank_order: Any, *, label: str) -> None:
    if (
        not isinstance(rank_order, Sequence)
        or isinstance(rank_order, str | bytes)
        or not rank_order
    ):
        raise ValueError(f"{label} must be a non-empty list")
    for index, raw_item in enumerate(rank_order):
        item_label = f"{label}[{index}]"
        if isinstance(raw_item, str):
            continue
        item = _require_mapping(raw_item, label=item_label)
        allowed_keys = {"metric", "aggregation", "direction"}
        extra_keys = sorted(set(item) - allowed_keys)
        if extra_keys:
            raise ValueError(f"{item_label} has unexpected keys: {extra_keys}")
        _require_non_empty_string(item, "metric", label=item_label)
        if "aggregation" in item:
            _require_non_empty_string(item, "aggregation", label=item_label)
        if "direction" in item:
            direction = _require_non_empty_string(item, "direction", label=item_label)
            if direction not in {"maximize", "minimize"}:
                raise ValueError(f"{item_label}.direction must be maximize or minimize")


def _validate_objective_rank(objective: Mapping[str, Any], *, label: str) -> None:
    rank = _require_key(objective, "rank", label=label)
    _validate_rank_order(rank, label=f"{label}.rank")


def _validate_selection_policy(selection_policy: Mapping[str, Any], *, label: str) -> None:
    allowed_selection_keys = {"rank_order"}
    extra_selection_keys = sorted(set(selection_policy) - allowed_selection_keys)
    if extra_selection_keys:
        raise ValueError(
            f"{label} must contain only rank_order; unexpected keys: {extra_selection_keys}"
        )
    rank_order = _require_key(selection_policy, "rank_order", label=label)
    _validate_rank_order(rank_order, label=f"{label}.rank_order")


def _validate_goal_release(document: Mapping[str, Any], *, label: str) -> None:
    if "release" not in document:
        return
    release = _require_mapping(document["release"], label=f"{label}.release")
    allowed_release_keys = {"huggingface"}
    extra_release_keys = sorted(set(release) - allowed_release_keys)
    if extra_release_keys:
        raise ValueError(f"{label}.release has unexpected keys: {extra_release_keys}")
    huggingface = _require_mapping(
        _require_key(release, "huggingface", label=f"{label}.release"),
        label=f"{label}.release.huggingface",
    )
    allowed_hf_keys = {
        "card_template",
        "checkpoint_filename",
        "include_youtube_preview",
        "preview_filename",
        "repo",
    }
    extra_hf_keys = sorted(set(huggingface) - allowed_hf_keys)
    if extra_hf_keys:
        raise ValueError(f"{label}.release.huggingface has unexpected keys: {extra_hf_keys}")
    repo = _require_non_empty_string(huggingface, "repo", label=f"{label}.release.huggingface")
    if "/" in repo:
        raise ValueError(f"{label}.release.huggingface.repo must not contain '/'")
    template = _require_non_empty_string(
        huggingface,
        "card_template",
        label=f"{label}.release.huggingface",
    )
    if template != "stable-retro-sb3":
        raise ValueError(
            f"{label}.release.huggingface.card_template has unsupported value: {template!r}"
        )
    checkpoint_filename = _require_non_empty_string(
        huggingface,
        "checkpoint_filename",
        label=f"{label}.release.huggingface",
    )
    if "{checkpoint_step}" not in checkpoint_filename:
        raise ValueError(
            f"{label}.release.huggingface.checkpoint_filename must contain {{checkpoint_step}}"
        )
    _require_non_empty_string(
        huggingface,
        "preview_filename",
        label=f"{label}.release.huggingface",
    )
    _require_bool(
        huggingface,
        "include_youtube_preview",
        label=f"{label}.release.huggingface",
    )


def load_goal_contract(
    path: Path,
    repo_root: Path | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Return a goal contract with Hydra defaults resolved."""
    repo_root = (repo_root or Path(".")).resolve()
    path = path.resolve()
    document = render_template_vars(
        load_composed_mapping(path, cycle_label="goal").document,
        path=path,
        label=f"goal file {_display_path(path, repo_root)}",
        deferred_fields_by_path=GOAL_DEFERRED_TEMPLATE_FIELDS,
    )
    if validate:
        _validate_goal_contract_document(document, path, repo_root)
    return document


def validate_goal_contract(path: Path, repo_root: Path | None = None) -> None:
    repo_root = (repo_root or Path(".")).resolve()
    path = path.resolve()
    document = load_goal_contract(path, repo_root, validate=False)
    _validate_goal_contract_document(document, path, repo_root)


def _validate_goal_contract_document(
    document: Mapping[str, Any],
    path: Path,
    repo_root: Path,
) -> None:
    label = f"goal file {_display_path(path, repo_root)}"
    if "schema_version" in document:
        raise ValueError(f"{label}.schema_version is not part of goal contracts")
    if "status" in document:
        raise ValueError(f"{label}.status is not part of goal contracts")
    narrative_top_level_keys = {
        "batch_record_fields",
        "capacity_policy_file",
        "cap_policy",
        "constraints",
        "default_eval_profile",
        "default_train_profile",
        "default_train_profile_note",
        "determinism",
        "environment_hash",
        "execution",
        "notes",
        "runtime",
        "search_protocol",
    }
    present_narrative_keys = sorted(set(document) & narrative_top_level_keys)
    if present_narrative_keys:
        raise ValueError(
            f"{label} must be script-readable; remove narrative keys: {present_narrative_keys}"
        )
    if "selection_policy" in document:
        raise ValueError(f"{label}.selection_policy moved to objective.rank")
    _validate_goal_release(document, label=label)
    goal_id = _require_non_empty_string(document, "goal_id", label=label)
    _require_non_empty_string(document, "title", label=label)
    goal_dir = path.parent
    if goal_dir.name != goal_id:
        raise ValueError(
            f"{_label_path(label, 'goal_id')} must match goal directory name: {goal_dir.name}"
        )
    objective = _require_mapping(
        _require_key(document, "objective", label=label), label=f"{label}.objective"
    )
    narrative_objective_keys = {"algorithm", "forbidden_stop_rules", "game", "success_requirement"}
    present_objective_narrative_keys = sorted(set(objective) & narrative_objective_keys)
    if present_objective_narrative_keys:
        raise ValueError(
            f"{label}.objective must be script-readable; "
            f"remove narrative keys: {present_objective_narrative_keys}"
        )
    if "success" in objective:
        raise ValueError(f"{label}.objective.success moved to train.early_stop")
    _validate_objective_rank(objective, label=f"{label}.objective")

    train = _goal_train_section(document, label=label)
    if "training" in document:
        raise ValueError(f"{label}.training is not part of goal contracts")
    if "max_train_timesteps" in train:
        raise ValueError(f"{label}.train.max_train_timesteps is not part of goal contracts")
    policy = train.get("policy")
    if isinstance(policy, Mapping):
        policy_early_stop_keys = {
            "early_stop",
            "early_stop_metric",
            "early_stop_operator",
            "early_stop_threshold",
        }
        present = sorted(set(policy) & policy_early_stop_keys)
        if present:
            raise ValueError(
                f"{label}.train.policy early-stop fields moved to train.early_stop: {present}"
            )
    if "early_stop" in train:
        normalize_early_stop_config(train["early_stop"], label=f"{label}.train.early_stop")
    environment = _goal_train_environment(document, train, label=label)
    _validate_environment_identity({"environment": environment}, label=f"{label}.train")
    env_config = (
        environment.get("env_config")
        if isinstance(environment.get("env_config"), Mapping)
        else environment
    )
    if "state" in env_config and "states" in env_config:
        raise ValueError(
            f"{label}.train.environment.env_config must define only one of state or states"
        )
    if "state" in env_config:
        state = env_config["state"]
        if not isinstance(state, str) or not state.strip():
            raise ValueError(
                f"{label}.train.environment.env_config.state must be a non-empty string"
            )
        environment_states = [state.strip()]
    elif "states" in env_config:
        environment_states = _require_string_list(
            env_config, "states", label=f"{label}.train.environment.env_config"
        )
    else:
        raise ValueError(f"{label}.train.environment.env_config must define state or states")
    if "states" in objective:
        objective_states = _require_string_list(objective, "states", label=f"{label}.objective")
    else:
        objective_states = environment_states
    if environment_states != objective_states:
        raise ValueError(
            f"{label}.objective.states must match environment.state when present: "
            f"{environment_states!r} != {objective_states!r}"
        )

    _validate_goal_eval(document, label=label)


def validate_env_config_file(path: Path) -> None:
    document = load_composed_mapping(path, cycle_label="env config").document
    label = f"env config file {path}"
    env_config = document.get("env_config")
    if isinstance(env_config, Mapping):
        extra_keys = sorted(set(document) - {"env_provider", "env_config"})
        if extra_keys:
            raise ValueError(f"{label} has unexpected keys: {extra_keys}")
        _validate_environment_env_config(document, env_config, label=label, require_game=True)
    else:
        _validate_env_config(document, label=label, require_game=True)
    if "state" in document or "states" in document:
        raise ValueError(f"{label} must not define state or states")


def validate_instance_config(path: Path, repo_root: Path | None = None) -> None:
    repo_root = repo_root or Path(".")
    config = load_instance_config(repo_root, path)
    instances = _require_mapping(config.get("instances"), label=f"instance config {path}.instances")
    if not instances:
        raise ValueError(f"instance config {path}.instances must not be empty")
    for name, raw in instances.items():
        label = f"instance config {path}.instances.{name}"
        instance = _require_mapping(raw, label=label)
        _require_non_empty_string(instance, "kind", label=label)
        default_workers = _require_int(instance, "default_workers", label=label, minimum=1)
        max_workers = _require_int(
            instance, "hardware_max_workers", label=label, minimum=default_workers
        )
        if max_workers < default_workers:
            raise ValueError(f"{label}.hardware_max_workers must be >= default_workers")


def validate_fleet_and_capacity(repo_root: Path) -> None:
    config = load_fleet_config(repo_root)
    policy = load_capacity_policy(repo_root)
    _require_schema_version(policy, 1, label="capacity policy")
    validate_capacity_policy(policy, config)
    lanes = policy.get("lanes")
    if not isinstance(lanes, Sequence) or isinstance(lanes, str | bytes) or not lanes:
        raise ValueError("capacity policy lanes must be a non-empty list")
    for index, raw in enumerate(lanes):
        label = f"capacity policy lanes[{index}]"
        lane = _require_mapping(raw, label=label)
        _require_non_empty_string(lane, "name", label=label)
        _require_non_empty_string(lane, "target", label=label)
        _require_non_empty_string(lane, "manager", label=label)
        _require_int(lane, "max_runner_workers", label=label, minimum=1)
        _require_int(lane, "env_threads", label=label, minimum=1)
        _require_string_list(lane, "use_for", label=label)


def validate_machine_config(repo_root: Path) -> None:
    load_fleet_config(repo_root)


def validate_benchmark_baselines(path: Path) -> None:
    document = load_mapping_document(path, label=f"benchmark baselines file {path}")
    label = f"benchmark baselines file {path}"
    _require_schema_version(document, BENCHMARK_BASELINES_SCHEMA_VERSION, label=label)
    baselines = _require_mapping(
        _require_key(document, "baselines", label=label), label=f"{label}.baselines"
    )
    if not baselines:
        raise ValueError(f"{label}.baselines must not be empty")
    for name, raw in baselines.items():
        baseline_label = f"{label}.baselines.{name}"
        baseline = _require_mapping(raw, label=baseline_label)
        _require_non_empty_string(baseline, "target", label=baseline_label)
        _require_non_empty_string(baseline, "host", label=baseline_label)
        _require_int(baseline, "workers", label=baseline_label, minimum=1)
        _require_int(baseline, "env_threads", label=baseline_label, minimum=1)


def _capture_issue(issues: list[ValidationIssue], path: Path, repo_root: Path, action: Any) -> None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - validation should aggregate all schema failures.
        issues.append(ValidationIssue(path=_display_path(path, repo_root), message=str(exc)))


def _active_experiment_path(path: Path) -> bool:
    return ".deprecated" not in path.parts


def validate_experiment_tree(repo_root: Path | str = Path(".")) -> ValidationReport:
    repo_root = Path(repo_root).resolve()
    experiments_dir = repo_root / "experiments"
    issues: list[ValidationIssue] = []
    counts: dict[str, int] = {}

    if not experiments_dir.is_dir():
        return ValidationReport(
            issues=(
                ValidationIssue(path="experiments", message="experiments directory does not exist"),
            ),
            counts={},
        )

    yaml_files = sorted(experiments_dir.rglob("*.yaml")) + sorted(experiments_dir.rglob("*.yml"))
    json_files = sorted(experiments_dir.rglob("*.json"))
    counts["yaml_files"] = len(yaml_files)
    counts["json_files"] = len(json_files)
    for path in json_files:
        issues.append(
            ValidationIssue(
                path=_display_path(path, repo_root), message="experiments configs must be YAML"
            )
        )

    goals_dir = experiments_dir / "goals"
    goals = sorted(
        path
        for path in [*goals_dir.rglob("_goal.yaml"), *goals_dir.rglob("goal.yaml")]
        if _active_experiment_path(path)
    )
    counts["goals"] = len(goals)
    for path in goals:
        _capture_issue(
            issues, path, repo_root, lambda path=path: validate_goal_contract(path, repo_root)
        )

    specs = sorted(
        path
        for path in (experiments_dir / "goals").rglob("specs/*.yaml")
        if _active_experiment_path(path)
    )
    counts["train_specs"] = len(specs)
    for path in specs:
        _capture_issue(issues, path, repo_root, lambda path=path: load_spec_document(path))

    env_configs_dir = experiments_dir / "envs"
    env_configs = sorted(env_configs_dir.rglob("*.yaml")) if env_configs_dir.is_dir() else []
    counts["env_configs"] = len(env_configs)
    for path in env_configs:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_env_config_file(path))

    instances_path = experiments_dir / "instances.yaml"
    counts["instance_configs"] = int(instances_path.is_file())
    if instances_path.is_file():
        _capture_issue(
            issues,
            instances_path,
            repo_root,
            lambda: validate_instance_config(instances_path, repo_root),
        )

    machines_path = experiments_dir / "machines.yaml"
    capacity_path = experiments_dir / "policies" / "capacity_policy.yaml"
    counts["machine_configs"] = int(machines_path.is_file())
    counts["capacity_policies"] = int(capacity_path.is_file())
    if machines_path.is_file():
        _capture_issue(issues, machines_path, repo_root, lambda: validate_machine_config(repo_root))
    else:
        issues.append(ValidationIssue(path="experiments/machines.yaml", message="file is required"))
    if machines_path.is_file() and capacity_path.is_file():
        _capture_issue(
            issues, capacity_path, repo_root, lambda: validate_fleet_and_capacity(repo_root)
        )
    elif not capacity_path.is_file():
        issues.append(
            ValidationIssue(
                path="experiments/policies/capacity_policy.yaml", message="file is required"
            )
        )

    benchmark_dir = experiments_dir / "benchmarks"
    benchmark_baselines = benchmark_dir / "baselines.yaml"
    counts["benchmark_baselines"] = int(benchmark_baselines.is_file())
    if benchmark_baselines.is_file():
        _capture_issue(
            issues,
            benchmark_baselines,
            repo_root,
            lambda: validate_benchmark_baselines(benchmark_baselines),
        )

    profile_dir = benchmark_dir / "profiles"
    if profile_dir.is_dir():
        _capture_issue(issues, profile_dir, repo_root, lambda: load_benchmark_profiles(profile_dir))
        counts["benchmark_profiles"] = len(sorted(profile_dir.glob("*.yaml")))
    else:
        counts["benchmark_profiles"] = 0

    return ValidationReport(issues=tuple(issues), counts=counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab validate",
        description="Validate checked-in YAML experiment, goal, spec, benchmark, and ops configs.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    parser.add_argument(
        "--load-goal",
        type=Path,
        help="Print the final composed goal contract for a _goal.yaml path.",
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format for --load-goal.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.load_goal is not None:
        document = load_goal_contract(args.load_goal, args.repo_root)
        output_format = "json" if args.json else args.format
        if output_format == "json":
            print(json.dumps(document, indent=2, sort_keys=True))
        else:
            print(yaml.safe_dump(document, sort_keys=False), end="")
        return 0

    report = validate_experiment_tree(args.repo_root)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    elif report.ok:
        counts = ", ".join(f"{name}={value}" for name, value in sorted(report.counts.items()))
        print(f"YAML config validation passed ({counts}).")
    else:
        print("YAML config validation failed:", file=sys.stderr)
        for issue in report.issues:
            print(f"- {issue.path}: {issue.message}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
