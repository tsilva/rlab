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
from rlab.config_loader import YAML_EXTENSIONS, load_composed_mapping, load_mapping_document
from rlab.env_identity import environment_hash, environment_identity_from_train_config
from rlab.fleet import load_capacity_policy, load_fleet_config, validate_capacity_policy
from rlab.job_queue import load_spec_document


GOAL_SCHEMA_VERSION = 1
RECIPE_SCHEMA_VERSION = 1
BENCHMARK_BASELINES_SCHEMA_VERSION = 1


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


def _resolve_repo_path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


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


def _require_existing_file(repo_root: Path, document: Mapping[str, Any], key: str, *, label: str) -> Path:
    value = _require_non_empty_string(document, key, label=label)
    path = _resolve_repo_path(repo_root, value)
    if not path.is_file():
        raise ValueError(f"{_label_path(label, key)} does not exist: {value}")
    return path


def _validate_environment_identity(
    document: Mapping[str, Any],
    *,
    label: str,
    require_hash: bool,
) -> Mapping[str, Any]:
    environment = _require_mapping(
        _require_key(document, "environment", label=label),
        label=f"{label}.environment",
    )
    _require_non_empty_string(environment, "provider", label=f"{label}.environment")
    if not (
        isinstance(environment.get("env_id"), str)
        and environment["env_id"].strip()
        or isinstance(environment.get("provider_env_id"), str)
        and environment["provider_env_id"].strip()
    ):
        raise ValueError(f"{label}.environment must define env_id")
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
    if require_hash:
        configured_hash = _require_non_empty_string(document, "environment_hash", label=label)
        canonical = environment_identity_from_train_config({}, environment=environment)
        expected_hash = environment_hash(canonical)
        if configured_hash != expected_hash:
            raise ValueError(
                f"{label}.environment_hash must be {expected_hash}, got {configured_hash}"
            )
    return environment


def load_goal_contract(
    path: Path,
    repo_root: Path | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Return a goal contract with Hydra defaults resolved."""
    repo_root = (repo_root or Path(".")).resolve()
    path = path.resolve()
    document = load_composed_mapping(path, cycle_label="goal").document
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
    _require_schema_version(document, GOAL_SCHEMA_VERSION, label=label)
    goal_slug = _require_non_empty_string(document, "goal_slug", label=label)
    _require_non_empty_string(document, "title", label=label)
    _require_non_empty_string(document, "status", label=label)
    goal_dir = path.parent
    if goal_dir.name != goal_slug:
        raise ValueError(
            f"{_label_path(label, 'goal_slug')} must match goal directory name: {goal_dir.name}"
        )

    objective = _require_mapping(_require_key(document, "objective", label=label), label=f"{label}.objective")
    _require_non_empty_string(objective, "game", label=f"{label}.objective")
    _require_non_empty_string(objective, "algorithm", label=f"{label}.objective")
    _require_non_empty_string(objective, "primary_metric", label=f"{label}.objective")
    _require_number(objective, "success_threshold", label=f"{label}.objective")
    _require_int(objective, "success_window_attempts", label=f"{label}.objective", minimum=1)
    _require_int(objective, "max_train_timesteps", label=f"{label}.objective", minimum=1)

    environment = _validate_environment_identity(document, label=label, require_hash=True)
    state_section = _require_mapping(
        _require_key(environment, "state", label=f"{label}.environment"),
        label=f"{label}.environment.state",
    )
    if "state" in state_section:
        environment_states = [str(state_section["state"])]
    elif "states" in state_section:
        environment_states = _require_string_list(state_section, "states", label=f"{label}.environment.state")
    else:
        raise ValueError(f"{label}.environment.state must define state or states")
    if "states" in objective:
        objective_states = _require_string_list(objective, "states", label=f"{label}.objective")
    else:
        objective_states = environment_states
    if environment_states != objective_states:
        raise ValueError(
            f"{label}.objective.states must match environment.state when present: "
            f"{environment_states!r} != {objective_states!r}"
        )

    selection_policy = _require_mapping(
        _require_key(document, "selection_policy", label=label),
        label=f"{label}.selection_policy",
    )
    _require_string_list(selection_policy, "rank_order", label=f"{label}.selection_policy")

    capacity_path = _require_existing_file(repo_root, document, "capacity_policy_file", label=label)
    if capacity_path.suffix.lower() not in YAML_EXTENSIONS:
        raise ValueError(f"{_label_path(label, 'capacity_policy_file')} must be YAML")

    execution = _require_mapping(_require_key(document, "execution", label=label), label=f"{label}.execution")
    for key in ("hardware_config_file", "fleet_config_file"):
        config_path = _require_existing_file(repo_root, execution, key, label=f"{label}.execution")
        if config_path.suffix.lower() not in YAML_EXTENSIONS:
            raise ValueError(f"{_label_path(f'{label}.execution', key)} must be YAML")


def validate_train_recipe(path: Path) -> None:
    document = load_mapping_document(path, label=f"recipe file {path}")
    label = f"recipe file {path}"
    _require_schema_version(document, RECIPE_SCHEMA_VERSION, label=label)
    kind = _require_non_empty_string(document, "kind", label=label)
    if kind != "train_recipe":
        raise ValueError(f"{label}.kind must be train_recipe")
    _require_non_empty_string(document, "slug", label=label)
    _require_non_empty_string(document, "algorithm", label=label)
    reward = None
    if "environment" in document:
        environment = _validate_environment_identity(document, label=label, require_hash=False)
        reward = _require_mapping(
            _require_key(environment, "reward", label=f"{label}.environment"),
            label=f"{label}.environment.reward",
        )
    else:
        env = _require_mapping(_require_key(document, "env", label=label), label=f"{label}.env")
        _require_non_empty_string(env, "game", label=f"{label}.env")
        _require_non_empty_string(env, "action_set", label=f"{label}.env")
        reward = _require_mapping(_require_key(document, "reward", label=label), label=f"{label}.reward")

    train = _require_mapping(_require_key(document, "train", label=label), label=f"{label}.train")
    logging = _require_mapping(_require_key(document, "logging", label=label), label=f"{label}.logging")

    for key in ("n_steps", "batch_size", "n_epochs"):
        _require_int(train, key, label=f"{label}.train", minimum=1)
    if "reward_mode" in reward:
        _require_non_empty_string(reward, "reward_mode", label=f"{label}.reward")
    _require_int(logging, "timesteps", label=f"{label}.logging", minimum=1)
    wandb = _require_key(logging, "wandb", label=f"{label}.logging")
    if not isinstance(wandb, bool):
        raise ValueError(f"{label}.logging.wandb must be a boolean")
    if "wandb_mode" in logging:
        wandb_mode = _require_non_empty_string(logging, "wandb_mode", label=f"{label}.logging")
        if wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"{label}.logging.wandb_mode must be one of online, offline, disabled")


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
        max_workers = _require_int(instance, "hardware_max_workers", label=label, minimum=default_workers)
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


def validate_benchmark_baselines(path: Path) -> None:
    document = load_mapping_document(path, label=f"benchmark baselines file {path}")
    label = f"benchmark baselines file {path}"
    _require_schema_version(document, BENCHMARK_BASELINES_SCHEMA_VERSION, label=label)
    baselines = _require_mapping(_require_key(document, "baselines", label=label), label=f"{label}.baselines")
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


def validate_experiment_tree(repo_root: Path | str = Path(".")) -> ValidationReport:
    repo_root = Path(repo_root).resolve()
    experiments_dir = repo_root / "experiments"
    issues: list[ValidationIssue] = []
    counts: dict[str, int] = {}

    if not experiments_dir.is_dir():
        return ValidationReport(
            issues=(ValidationIssue(path="experiments", message="experiments directory does not exist"),),
            counts={},
        )

    yaml_files = sorted(experiments_dir.rglob("*.yaml")) + sorted(experiments_dir.rglob("*.yml"))
    json_files = sorted(experiments_dir.rglob("*.json"))
    counts["yaml_files"] = len(yaml_files)
    counts["json_files"] = len(json_files)
    for path in json_files:
        issues.append(ValidationIssue(path=_display_path(path, repo_root), message="experiments configs must be YAML"))

    goals = sorted((experiments_dir / "goals").glob("*/goal.yaml"))
    counts["goals"] = len(goals)
    for path in goals:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_goal_contract(path, repo_root))

    specs = sorted((experiments_dir / "goals").glob("*/specs/*.yaml"))
    counts["train_specs"] = len(specs)
    for path in specs:
        _capture_issue(issues, path, repo_root, lambda path=path: load_spec_document(path))

    recipes_dir = experiments_dir / "recipes"
    recipes = sorted(recipes_dir.rglob("*.yaml")) if recipes_dir.is_dir() else []
    counts["recipes"] = len(recipes)
    for path in recipes:
        _capture_issue(issues, path, repo_root, lambda path=path: validate_train_recipe(path))

    instances_path = experiments_dir / "instances.yaml"
    counts["instance_configs"] = int(instances_path.is_file())
    if instances_path.is_file():
        _capture_issue(
            issues,
            instances_path,
            repo_root,
            lambda: validate_instance_config(instances_path, repo_root),
        )

    fleet_path = experiments_dir / "fleet.yaml"
    capacity_path = experiments_dir / "policies" / "capacity_policy.yaml"
    counts["fleet_configs"] = int(fleet_path.is_file())
    counts["capacity_policies"] = int(capacity_path.is_file())
    if fleet_path.is_file() and capacity_path.is_file():
        _capture_issue(issues, capacity_path, repo_root, lambda: validate_fleet_and_capacity(repo_root))
    else:
        if not fleet_path.is_file():
            issues.append(ValidationIssue(path="experiments/fleet.yaml", message="file is required"))
        if not capacity_path.is_file():
            issues.append(ValidationIssue(path="experiments/policies/capacity_policy.yaml", message="file is required"))

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
        description="Validate checked-in YAML experiment, goal, spec, recipe, and ops configs.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    parser.add_argument(
        "--load-goal",
        type=Path,
        help="Print the final composed goal contract for a goal.yaml path.",
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
