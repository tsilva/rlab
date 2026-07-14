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
from rlab.checkpoint_eval_config import normalize_checkpoint_eval_stages
from rlab.config_loader import load_composed_mapping
from rlab.early_stop import normalize_early_stop_config
from rlab.env_identity import validate_task_config
from rlab.env_registry import (
    env_supports_states,
    qualify_env_id,
    resolve_env_id,
    validate_provider_constructor_args,
)
from rlab.goal_schema import validate_goal_document_shape
from rlab.machines import DEFAULT_MACHINE_REGISTRY, load_machine_registry
from rlab.modal_eval_config import load_modal_eval_config
from rlab.metric_names import metric_path_segment
from rlab.recipe_documents import (
    load_goal_contract_document,
    load_recipe_source_document,
    materialize_train_recipe_document,
)
from rlab.ranking import parse_objective_rank
from rlab.seeds import validate_eval_seed
from rlab.train_config import (
    env_config_allowed_keys,
    validate_and_normalize_train_config,
    validate_train_config_fields,
)
from rlab.validation import (
    is_int as _is_int,
    label_path as _label_path,
    require_int as _require_int,
    require_key as _require_key,
    require_mapping as _require_mapping,
    require_non_empty_string as _require_non_empty_string,
    normalize_obs_crop,
    string_list,
)


ENV_CONFIG_ALLOWED_KEYS = env_config_allowed_keys() | {"n_envs", "task"}

GOAL_REQUIRED_ENV_CONFIG_KEYS = frozenset(
    {
        "frame_skip",
        "max_pool_frames",
        "n_envs",
        "obs_crop",
        "obs_crop_fill",
        "obs_crop_mode",
        "obs_resize_algorithm",
        "observation_size",
        "sticky_action_prob",
    }
)


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


def _require_bool(document: Mapping[str, Any], key: str, *, label: str) -> bool:
    value = _require_key(document, key, label=label)
    if not isinstance(value, bool):
        raise ValueError(f"{_label_path(label, key)} must be a boolean")
    return value


def _require_string_list(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    value = _require_key(document, key, label=label)
    return string_list(value, label=_label_path(label, key), allow_empty=allow_empty)


def _validate_obs_crop(preprocessing: Mapping[str, Any], *, label: str) -> None:
    if "hud_crop_top" in preprocessing:
        raise ValueError(f"{label}.hud_crop_top is redundant; use obs_crop")
    if "obs_crop" not in preprocessing:
        raise ValueError(f"{label}.obs_crop is required")
    normalize_obs_crop(preprocessing["obs_crop"], label=f"{label}.obs_crop")


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
        task = _require_mapping(
            _require_key(environment, "task", label=f"{label}.environment"),
            label=f"{label}.environment.task",
        )
        validate_task_config(task, label=f"{label}.environment.task")
        return environment

    env_id = _require_non_empty_string(environment, "env_id", label=f"{label}.environment")
    try:
        resolve_env_id(env_id)
    except ValueError as exc:
        raise ValueError(f"{label}.environment.env_id is invalid: {exc}") from exc
    task = _require_mapping(
        _require_key(environment, "task", label=f"{label}.environment"),
        label=f"{label}.environment.task",
    )
    validate_task_config(task, label=f"{label}.environment.task")
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
    if "task" in environment and "task" not in combined:
        combined["task"] = environment["task"]
    _validate_env_config(
        combined,
        label=f"{label}.env_config",
        require_game=require_game,
        allowed_extra_keys=allowed_extra_keys,
    )


def _validate_explicit_goal_environment_args(
    environment: Mapping[str, Any],
    env_config: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject goal environments that silently inherit provider defaults."""

    explicit_config = dict(env_config)
    preprocessing = environment.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        for key, value in preprocessing.items():
            explicit_config.setdefault(key, value)
    missing_config = sorted(GOAL_REQUIRED_ENV_CONFIG_KEYS - set(explicit_config))
    if missing_config:
        raise ValueError(
            f"{label}.env_config missing explicit environment field(s): "
            + ", ".join(missing_config)
        )

    provider_id = str(
        environment.get("env_provider") or explicit_config.get("env_provider") or ""
    ).strip()
    if env_supports_states(provider_id, str(explicit_config.get("game") or "")) and not (
        "state" in explicit_config or "states" in explicit_config
    ):
        raise ValueError(f"{label}.env_config must explicitly define state or states")
    env_args = explicit_config.get("env_args")
    validate_provider_constructor_args(
        provider_id,
        env_args,
        label=f"{label}.env_config.env_args",
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
    if "env_args" in env_config and not isinstance(env_config["env_args"], Mapping):
        raise ValueError(f"{label}.env_args must be an object")
    validation_config = dict(env_config)
    env_args = env_config.get("env_args")
    if isinstance(env_args, Mapping) and "game" in env_args and "game" not in validation_config:
        validation_config["game"] = env_args["game"]
    required_keys = []
    if require_provider:
        required_keys.append("env_provider")
    if require_game:
        required_keys.append("game")
    validate_train_config_fields(
        validation_config,
        label=label,
        keys=tuple((set(validation_config) & ENV_CONFIG_ALLOWED_KEYS) - {"task"}),
        required_keys=tuple(required_keys),
    )
    if require_provider:
        env_provider = str(validation_config["env_provider"]).strip()
    elif "env_provider" in validation_config:
        env_provider = str(validation_config["env_provider"]).strip()
    else:
        env_provider = None
    if require_game:
        game = str(validation_config["game"]).strip()
    elif "game" in validation_config:
        game = str(validation_config["game"]).strip()
    else:
        game = None
    if game and env_provider:
        try:
            qualify_env_id(env_provider, game)
        except ValueError as exc:
            raise ValueError(f"{label}.env_provider is invalid: {exc}") from exc
    if "state" in env_config and "states" in env_config:
        raise ValueError(f"{label} must define only one of state or states")
    state_values: list[str] = []
    if isinstance(env_config.get("state"), str):
        state_values = [str(env_config["state"]).strip()]
    elif isinstance(env_config.get("states"), list | tuple):
        state_values = [str(value).strip() for value in env_config["states"]]
    if len(set(state_values)) != len(state_values):
        raise ValueError(f"{label}.states must contain unique identifiers")
    for state in state_values:
        try:
            metric_path_segment(state)
        except ValueError as exc:
            raise ValueError(f"{label} start identifiers must be safe metric dimensions") from exc
    if "task" in env_config:
        task = _require_mapping(env_config["task"], label=f"{label}.task")
        validate_task_config(task, label=f"{label}.task")


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
    return _require_mapping(
        _require_key(train, "environment", label=f"{label}.train"),
        label=f"{label}.train.environment",
    )


def _validate_goal_eval(document: Mapping[str, Any], *, label: str) -> None:
    if "eval_spec" in document:
        raise ValueError(f"{label}.eval_spec moved to eval")
    eval_section = _require_mapping(
        _require_key(document, "eval", label=label),
        label=f"{label}.eval",
    )
    _require_int(eval_section, "episodes", label=f"{label}.eval", minimum=1)
    eval_environment = eval_section.get("environment")
    if isinstance(eval_environment, Mapping):
        eval_environment_keys = {
            "env_provider",
            "env_config",
            "preprocessing",
            "task",
        }
        extra_keys = sorted(set(eval_environment) - eval_environment_keys)
        if extra_keys:
            raise ValueError(f"{label}.eval.environment has unexpected keys: {extra_keys}")
        eval_env_config = _require_mapping(
            _require_key(eval_environment, "env_config", label=f"{label}.eval.environment"),
            label=f"{label}.eval.environment.env_config",
        )
        if "max_episodes" in eval_env_config:
            raise ValueError(
                f"{label}.eval.environment.env_config.max_episodes moved to {label}.eval.episodes"
            )
        _validate_environment_env_config(
            eval_environment,
            eval_env_config,
            label=f"{label}.eval.environment",
            require_game=True,
            allowed_extra_keys={"seed", "n_envs", "max_steps"},
        )
        _validate_explicit_goal_environment_args(
            eval_environment,
            eval_env_config,
            label=f"{label}.eval.environment",
        )
        if "seed" in eval_env_config:
            seed = _require_int(
                eval_env_config,
                "seed",
                label=f"{label}.eval.environment.env_config",
            )
            validate_eval_seed(seed, label=f"{label}.eval.environment.env_config.seed")
        if "n_envs" in eval_env_config:
            _require_int(
                eval_env_config, "n_envs", label=f"{label}.eval.environment.env_config", minimum=1
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
        for moved_key in ("seed", "n_envs", "max_steps"):
            if moved_key in policy:
                raise ValueError(
                    f"{label}.eval.policy.{moved_key} moved to "
                    f"{label}.eval.environment.env_config.{moved_key}"
                )
        if "stochastic" in policy:
            stochastic = _require_bool(policy, "stochastic", label=f"{label}.eval.policy")
            if not stochastic:
                raise ValueError(
                    f"{label}.eval.policy.stochastic must be true; "
                    "all policy evaluation uses stochastic sampling"
                )
    elif policy is not None:
        raise ValueError(f"{label}.eval.policy must be an object")


def _validate_rank_order(rank_order: Any, *, label: str) -> None:
    if parse_objective_rank(rank_order):
        return
    raise ValueError(f"{label} must be a non-empty list of max(metric) or min(metric) strings")


def _validate_objective_rank(objective: Mapping[str, Any], *, label: str) -> None:
    rank = _require_key(objective, "rank", label=label)
    _validate_rank_order(rank, label=f"{label}.rank")


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
    if huggingface:
        raise ValueError(
            f"{label}.release.huggingface is marker-only; repository identity, filenames, "
            "license, card format, and preview requirements are generated by rlab"
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
    document = load_goal_contract_document(
        path,
        label=f"goal file {_display_path(path, repo_root)}",
    )
    if validate:
        validate_goal_contract_document(document, path, repo_root)
    return document


def validate_goal_contract(path: Path, repo_root: Path | None = None) -> None:
    repo_root = (repo_root or Path(".")).resolve()
    path = path.resolve()
    document = load_goal_contract(path, repo_root, validate=False)
    validate_goal_contract_document(document, path, repo_root)


def validate_goal_contract_document(
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
    if "policy" in train:
        raise ValueError(
            f"{label}.train.policy is retired; use train.backend with an explicit id and config"
        )
    if "early_stop" in train:
        normalize_early_stop_config(train["early_stop"], label=f"{label}.train.early_stop")
    if "checkpoint_eval_stages" in train:
        normalize_checkpoint_eval_stages(
            train["checkpoint_eval_stages"],
            label=f"{label}.train.checkpoint_eval_stages",
        )
    environment = _goal_train_environment(document, train, label=label)
    _validate_environment_identity({"environment": environment}, label=f"{label}.train")
    env_config = (
        environment.get("env_config")
        if isinstance(environment.get("env_config"), Mapping)
        else environment
    )
    _validate_explicit_goal_environment_args(
        environment,
        env_config,
        label=f"{label}.train.environment",
    )
    env_provider = (
        str(environment["env_provider"]).strip()
        if "env_provider" in environment
        else str(env_config.get("env_provider", "")).strip()
    )
    supports_states = env_supports_states(env_provider, str(env_config.get("game") or ""))
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
    elif not supports_states:
        environment_states = []
    else:
        raise ValueError(f"{label}.train.environment.env_config must define state or states")
    if "states" in objective:
        objective_states = _require_string_list(objective, "states", label=f"{label}.objective")
    else:
        objective_states = environment_states
    if len(set(environment_states)) != len(environment_states):
        raise ValueError(f"{label}.train.environment start identifiers must be unique")
    if environment_states != objective_states:
        raise ValueError(
            f"{label}.objective.states must match environment.state when present: "
            f"{environment_states!r} != {objective_states!r}"
        )
    for state in environment_states:
        try:
            metric_path_segment(state)
        except ValueError as exc:
            raise ValueError(
                f"{label}.train.environment start identifiers must be safe metric dimensions"
            ) from exc

    validate_goal_document_shape(document, label=label)
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


def validate_machine_config(repo_root: Path) -> None:
    load_machine_registry(repo_root / DEFAULT_MACHINE_REGISTRY)


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
    goals = sorted(path for path in goals_dir.rglob("_goal.yaml") if _active_experiment_path(path))
    counts["goals"] = len(goals)
    for path in goals:
        _capture_issue(
            issues, path, repo_root, lambda path=path: validate_goal_contract(path, repo_root)
        )

    legacy_goal_recipes = sorted(
        path
        for path in (experiments_dir / "goals").rglob("recipes/*.yaml")
        if _active_experiment_path(path)
    )
    for path in legacy_goal_recipes:
        issues.append(
            ValidationIssue(
                path=_display_path(path, repo_root),
                message="active goal-local recipes are unsupported; use experiments/recipes",
            )
        )
    recipes_root = experiments_dir / "recipes"
    recipes = sorted(
        path
        for path in recipes_root.rglob("*.yaml")
        if _active_experiment_path(path) and (not path.is_relative_to(recipes_root / "_presets"))
    )
    counts["train_recipes"] = len(recipes)
    for path in recipes:

        def validate_recipe(path: Path = path) -> None:
            composition = load_recipe_source_document(path)
            source = materialize_train_recipe_document(composition.document)
            validate_and_normalize_train_config(
                source.get("train_config") or {},
                label=f"recipe file {path} train_config",
                required_keys=("timesteps", "wandb", "wandb_mode", "wandb_artifact_storage_uri"),
            )

        _capture_issue(issues, path, repo_root, validate_recipe)

    env_configs = sorted(
        path for path in goals_dir.glob("*/_env-*.yaml") if _active_experiment_path(path)
    )
    counts["env_configs"] = len(env_configs)
    for path in env_configs:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_env_config_file(path))

    machines_path = experiments_dir / "machines.yaml"
    counts["machine_configs"] = int(machines_path.is_file())
    if machines_path.is_file():
        _capture_issue(issues, machines_path, repo_root, lambda: validate_machine_config(repo_root))
    else:
        issues.append(ValidationIssue(path="experiments/machines.yaml", message="file is required"))

    modal_eval_path = experiments_dir / "modal_eval.yaml"
    counts["modal_eval_configs"] = int(modal_eval_path.is_file())
    if modal_eval_path.is_file():
        _capture_issue(
            issues,
            modal_eval_path,
            repo_root,
            lambda: load_modal_eval_config(modal_eval_path),
        )
    else:
        issues.append(
            ValidationIssue(path="experiments/modal_eval.yaml", message="file is required")
        )

    benchmark_dir = experiments_dir / "benchmarks"
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
        description="Validate checked-in YAML experiment, goal, recipe, benchmark, and ops configs.",
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
