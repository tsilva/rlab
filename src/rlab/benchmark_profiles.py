from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.config_loader import load_mapping_document
from rlab.metric_names import validate_metric_name
from rlab.recipe_documents import compose_train_document
from rlab.train_config import validate_and_normalize_train_config
from rlab.validation import int_list
from rlab.validation import require_int
from rlab.validation import require_mapping
from rlab.validation import require_non_empty_string
from rlab.validation import require_schema_version
from rlab.validation import string_list


BENCHMARK_PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_DIR = Path("experiments/benchmarks/profiles")
DEFAULT_RESULT_DIR = Path("logs/benchmarks")
ALLOWED_KINDS = {
    "env_throughput",
    "local_smoke",
    "train_loop_comparison",
    "train_loop_throughput",
}
COMMON_PROFILE_FIELDS = frozenset({"schema_version", "name", "kind", "description"})
PROFILE_FIELDS_BY_KIND = {
    "env_throughput": frozenset(
        {
            "allow_state_none",
            "env_provider",
            "envs",
            "game",
            "max_runtime_overhead",
            "modes",
            "repeats",
            "script",
            "seed",
            "state",
            "steps",
            "warmup",
        }
    ),
    "local_smoke": frozenset(
        {
            "goal_file",
            "max_duration",
            "recipe_file",
            "runtime_image_ref_file",
            "seed",
            "target",
        }
    ),
    "train_loop_throughput": frozenset(
        {
            "goal_file",
            "recipe_file",
            "recipe_overrides",
            "required_metrics",
            "run_description",
            "run_name",
            "seed",
        }
    ),
    "train_loop_comparison": frozenset(
        {
            "goal_file",
            "baseline_recipe_file",
            "candidate_recipe_file",
            "recipe_overrides",
            "required_metrics",
            "candidate_required_metrics",
            "run_description",
            "seed",
            "repeats",
            "max_candidate_slowdown",
        }
    ),
}
STATE_NONE_VALUES = {"", "none", "state.none"}


@dataclass(frozen=True)
class BenchmarkCommand:
    label: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    stdin: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "argv": list(self.argv),
        }
        if self.cwd is not None:
            payload["cwd"] = str(self.cwd)
        if self.env:
            payload["env"] = dict(self.env)
        if self.stdin is not None:
            payload["stdin_json"] = json.loads(self.stdin)
        return payload


@dataclass(frozen=True)
class BenchmarkProfile:
    path: Path
    payload: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def kind(self) -> str:
        return str(self.payload["kind"])

    @property
    def description(self) -> str:
        return str(self.payload.get("description") or "")


def _is_state_none(value: Any) -> bool:
    return str(value or "").strip().lower() in STATE_NONE_VALUES


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return text or "benchmark"


def _profile_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"benchmark profile must be YAML: {path}")
    return load_mapping_document(path, label=f"benchmark profile {path}")


def validate_benchmark_profile(payload: Mapping[str, Any], *, label: str = "profile") -> None:
    require_mapping(payload, label=label)
    require_schema_version(
        payload,
        BENCHMARK_PROFILE_SCHEMA_VERSION,
        label=label,
        require_present=False,
    )
    require_non_empty_string(payload, "name", label=label, require_present=False)
    kind = require_non_empty_string(payload, "kind", label=label, require_present=False)
    if kind not in ALLOWED_KINDS:
        known = ", ".join(sorted(ALLOWED_KINDS))
        raise ValueError(f"{label}.kind must be one of {known}")
    if "expectations" in payload or "gates" in payload:
        raise ValueError(
            f"{label} must use executable kind-specific gate fields, not expectations or gates"
        )
    if "environment_contract" in payload:
        raise ValueError(
            f"{label}.environment_contract is unsupported; derive it from executed inputs"
        )

    if kind == "env_throughput":
        require_non_empty_string(payload, "env_provider", label=label, require_present=False)
        game = require_non_empty_string(payload, "game", label=label, require_present=False)
        state = require_non_empty_string(payload, "state", label=label, require_present=False)
        if _is_state_none(state) and not payload.get("allow_state_none"):
            raise ValueError(
                f"{label}.state must be an actual saved state for {game}; "
                "set allow_state_none=true only for emulator hot-path diagnostics"
            )
        modes = string_list(payload.get("modes", ["compare"]), label=f"{label}.modes")
        if modes != ["compare"]:
            raise ValueError(f"{label}.modes must be [compare] for an executable overhead gate")
        envs = int_list(payload.get("envs", [1]), label=f"{label}.envs")
        if any(env_count < 1 for env_count in envs):
            raise ValueError(f"{label}.envs values must be >= 1")
        require_int(payload, "steps", label=label, minimum=1, require_present=False)
        require_int(payload, "warmup", label=label, minimum=0, require_present=False)
        if "repeats" in payload:
            require_int(payload, "repeats", label=label, minimum=1)
        max_overhead = payload.get("max_runtime_overhead")
        if (
            isinstance(max_overhead, bool)
            or not isinstance(max_overhead, int | float)
            or max_overhead < 0
        ):
            raise ValueError(f"{label}.max_runtime_overhead must be a non-negative number")

    if kind == "train_loop_throughput":
        require_non_empty_string(payload, "goal_file", label=label, require_present=False)
        require_non_empty_string(payload, "recipe_file", label=label, require_present=False)
        string_list(
            payload.get("recipe_overrides", ()),
            label=f"{label}.recipe_overrides",
            allow_empty=True,
        )
        required_metrics = string_list(
            payload.get("required_metrics", ()), label=f"{label}.required_metrics"
        )
        if not required_metrics:
            raise ValueError(f"{label}.required_metrics must not be empty")
        for metric_name in required_metrics:
            validate_metric_name(metric_name)
        config = _train_loop_config(payload)
        require_non_empty_string(
            config,
            "game",
            label=f"{label}.train_config",
            require_present=False,
        )
        require_int(config, "timesteps", label=f"{label}.train_config", require_present=False)
        require_int(payload, "seed", label=label, minimum=0, require_present=False)

    if kind == "train_loop_comparison":
        require_non_empty_string(payload, "goal_file", label=label, require_present=False)
        require_non_empty_string(
            payload,
            "baseline_recipe_file",
            label=label,
            require_present=False,
        )
        require_non_empty_string(
            payload,
            "candidate_recipe_file",
            label=label,
            require_present=False,
        )
        string_list(
            payload.get("recipe_overrides", ()),
            label=f"{label}.recipe_overrides",
            allow_empty=True,
        )
        required_metrics = string_list(
            payload.get("required_metrics", ()), label=f"{label}.required_metrics"
        )
        candidate_metrics = string_list(
            payload.get("candidate_required_metrics", ()),
            label=f"{label}.candidate_required_metrics",
            allow_empty=True,
        )
        if not required_metrics:
            raise ValueError(f"{label}.required_metrics must not be empty")
        for metric_name in (*required_metrics, *candidate_metrics):
            validate_metric_name(metric_name)
        for recipe_field in ("baseline_recipe_file", "candidate_recipe_file"):
            _train_loop_config(payload, recipe_file=str(payload[recipe_field]))
        require_int(payload, "seed", label=label, minimum=0, require_present=False)
        if "repeats" in payload:
            require_int(payload, "repeats", label=label, minimum=1)
        slowdown = payload.get("max_candidate_slowdown")
        if (
            isinstance(slowdown, bool)
            or not isinstance(slowdown, int | float)
            or not 0.0 <= float(slowdown) < 1.0
        ):
            raise ValueError(f"{label}.max_candidate_slowdown must be in [0, 1)")

    if kind == "local_smoke":
        if "workers" in payload:
            raise ValueError(f"{label} does not support invocation-local workers")
        require_non_empty_string(payload, "goal_file", label=label, require_present=False)
        require_non_empty_string(payload, "recipe_file", label=label, require_present=False)
        require_non_empty_string(payload, "target", label=label, require_present=False)
        require_int(payload, "seed", label=label, minimum=0, require_present=False)

    allowed_fields = COMMON_PROFILE_FIELDS | PROFILE_FIELDS_BY_KIND[kind]
    unknown_fields = sorted(set(payload) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"{label} has unknown field(s) for {kind}: {', '.join(unknown_fields)}")


def load_benchmark_profile(path: Path) -> BenchmarkProfile:
    payload = _profile_payload(path)
    validate_benchmark_profile(payload, label=f"profile file {path}")
    return BenchmarkProfile(path=path, payload=dict(payload))


def load_benchmark_profiles(profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[BenchmarkProfile]:
    if not profile_dir.is_dir():
        raise ValueError(f"benchmark profile directory does not exist: {profile_dir}")
    paths = sorted([*profile_dir.glob("*.yaml"), *profile_dir.glob("*.yml")])
    return [load_benchmark_profile(path) for path in paths]


def find_benchmark_profile(
    name_or_path: str, *, profile_dir: Path = DEFAULT_PROFILE_DIR
) -> BenchmarkProfile:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_benchmark_profile(candidate)
    for profile in load_benchmark_profiles(profile_dir):
        if profile.name == name_or_path or profile.path.stem == name_or_path:
            return profile
    raise ValueError(f"unknown benchmark profile {name_or_path!r}")


def _command(
    label: str,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        label=label,
        argv=tuple(str(part) for part in argv),
        cwd=cwd,
        env=env,
        stdin=stdin,
    )


def _local_smoke_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    goal_file = str(profile["goal_file"])
    recipe_file = str(profile["recipe_file"])
    target = str(profile.get("target") or "b3")
    enqueue = [
        "rlab",
        "experiment",
        "launch",
        "--goal-file",
        goal_file,
        "--recipe-file",
        recipe_file,
        "--seed",
        str(profile.get("seed", 123)),
        "--run-description",
        str(profile.get("description") or "dstack local integration smoke"),
        "--compute",
        "local",
        "--target",
        target,
        "--max-duration",
        str(profile.get("max_duration") or "30m"),
        "--json",
    ]
    if profile.get("runtime_image_ref_file"):
        enqueue.extend(["--runtime-image-ref-file", str(profile["runtime_image_ref_file"])])
    return [_command("train-local-smoke", enqueue)]


def _env_throughput_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    script = str(profile.get("script") or "experiments/scripts/benchmarks/benchmark_env_sps.py")
    commands: list[BenchmarkCommand] = []
    for mode in string_list(profile.get("modes", ["compare"]), label="modes"):
        for envs in int_list(profile.get("envs", [1]), label="envs"):
            commands.append(
                _command(
                    f"{mode}-{envs}env",
                    [
                        sys.executable,
                        script,
                        "--env-provider",
                        str(profile["env_provider"]),
                        "--game",
                        str(profile["game"]),
                        "--state",
                        str(profile["state"]),
                        "--mode",
                        mode,
                        "--envs",
                        str(envs),
                        "--steps",
                        str(profile["steps"]),
                        "--warmup",
                        str(profile["warmup"]),
                        "--seed",
                        str(profile.get("seed", 123)),
                        "--repeats",
                        str(profile.get("repeats", 3)),
                        "--max-overhead",
                        str(profile.get("max_runtime_overhead", 0.05)),
                    ],
                    env={"STABLE_RETRO_DISABLE_AUDIO": "1"},
                )
            )
    return commands


def _train_loop_config(
    profile: Mapping[str, Any],
    *,
    recipe_file: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    document = compose_train_document(
        Path(str(profile["goal_file"])),
        Path(str(recipe_file or profile["recipe_file"])),
        recipe_overrides=profile.get("recipe_overrides", ()),
    )
    config = dict(require_mapping(document["train_config"], label="train_config"))
    config["checkpoint_eval_backend"] = "none"
    config["checkpoint_eval_stages"] = []
    config["early_stop"] = None
    config["post_train_eval_episodes"] = 0
    config.pop("rom_asset_manifest", None)
    config["wandb"] = False
    config["wandb_mode"] = "disabled"
    config["seed"] = int(profile.get("seed", 123))
    config["run_name"] = str(
        run_name or profile.get("run_name") or f"benchmark_{_slug(str(profile['name']))}"
    )
    config["run_description"] = str(
        profile.get("run_description")
        or f"Benchmark profile {profile['name']} training-loop probe."
    )
    return validate_and_normalize_train_config(config, label="train_loop_benchmark.train_config")


def _train_loop_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    config = _train_loop_config(profile)
    return [
        _command(
            "train",
            [sys.executable, "-m", "rlab.train", "--train-config-json", "/dev/stdin"],
            env={"RLAB_INTERNAL_LEARNER": "1"},
            stdin=json.dumps(config, sort_keys=True),
        )
    ]


def _train_loop_comparison_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    commands: list[BenchmarkCommand] = []
    variants = {
        "baseline": str(profile["baseline_recipe_file"]),
        "candidate": str(profile["candidate_recipe_file"]),
    }
    repeats = int(profile.get("repeats", 2))
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            label = f"{variant}-{repeat + 1}"
            config = _train_loop_config(
                profile,
                recipe_file=variants[variant],
                run_name=f"benchmark_{_slug(str(profile['name']))}_{label}",
            )
            commands.append(
                _command(
                    label,
                    [sys.executable, "-m", "rlab.train", "--train-config-json", "/dev/stdin"],
                    env={"RLAB_INTERNAL_LEARNER": "1"},
                    stdin=json.dumps(config, sort_keys=True),
                )
            )
    return commands


def build_benchmark_commands(profile: BenchmarkProfile) -> list[BenchmarkCommand]:
    payload = profile.payload
    kind = profile.kind
    if kind == "local_smoke":
        return _local_smoke_commands(payload)
    if kind == "env_throughput":
        return _env_throughput_commands(payload)
    if kind == "train_loop_throughput":
        return _train_loop_commands(payload)
    if kind == "train_loop_comparison":
        return _train_loop_comparison_commands(payload)
    raise ValueError(f"unsupported benchmark profile kind {kind!r}")
