from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.config_loader import load_mapping_document
from rlab.machines import load_machine_registry, resolve_machine
from rlab.recipe_documents import load_recipe_document
from rlab.runtime_refs import docker_image_ref, runtime_image_ref_from_file
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
    "artifact_storage_smoke",
    "container_smoke",
    "env_throughput",
    "eval_contract",
    "fleet_capacity",
    "local_smoke",
    "ppo_loop_throughput",
}
STATE_NONE_VALUES = {"", "none", "state.none"}


@dataclass(frozen=True)
class BenchmarkCommand:
    label: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "argv": list(self.argv),
        }
        if self.cwd is not None:
            payload["cwd"] = str(self.cwd)
        if self.env:
            payload["env"] = dict(self.env)
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
    if "gates" in payload:
        raise ValueError(f"{label}.gates is unsupported; use informational expectations")
    if "environment_contract" in payload:
        raise ValueError(
            f"{label}.environment_contract is unsupported; derive it from executed inputs"
        )
    require_mapping(payload.get("expectations", {}), label=f"{label}.expectations")

    if kind == "env_throughput":
        game = require_non_empty_string(payload, "game", label=label, require_present=False)
        state = require_non_empty_string(payload, "state", label=label, require_present=False)
        if _is_state_none(state) and not payload.get("allow_state_none"):
            raise ValueError(
                f"{label}.state must be an actual saved state for {game}; "
                "set allow_state_none=true only for emulator hot-path diagnostics"
            )
        string_list(payload.get("modes", ["fast"]), label=f"{label}.modes")
        int_list(payload.get("envs", [1]), label=f"{label}.envs")
        require_int(payload, "steps", label=label, require_present=False)
        require_int(payload, "warmup", label=label, require_present=False)
        if "repeats" in payload:
            require_int(payload, "repeats", label=label, minimum=1)

    if kind in {"ppo_loop_throughput", "artifact_storage_smoke"}:
        require_non_empty_string(payload, "recipe_file", label=label, require_present=False)
        string_list(payload.get("recipe_overrides", ()), label=f"{label}.recipe_overrides")
        document = load_recipe_document(
            Path(str(payload["recipe_file"])),
            recipe_overrides=payload.get("recipe_overrides", ()),
        )
        config = require_mapping(document.get("train_config"), label=f"{label}.train_config")
        validate_and_normalize_train_config(config, label=f"{label}.train_config")
        require_non_empty_string(
            config,
            "game",
            label=f"{label}.train_config",
            require_present=False,
        )
        require_int(config, "timesteps", label=f"{label}.train_config", require_present=False)

    if kind == "local_smoke":
        require_non_empty_string(payload, "recipe_file", label=label, require_present=False)
        require_non_empty_string(payload, "run_target", label=label, require_present=False)
        require_non_empty_string(payload, "machine", label=label, require_present=False)
        string_list(payload.get("recipe_overrides", ()), label=f"{label}.recipe_overrides")

    if kind == "fleet_capacity":
        if "target" in payload or "workers" in payload:
            raise ValueError(
                f"{label} must use host and requested_workers; machine target and capacity "
                "come from experiments/machines.yaml"
            )
        if not payload.get("recipe_file"):
            raise ValueError(f"{label}.recipe_file must be a non-empty string")
        require_non_empty_string(payload, "recipe_file", label=label, require_present=False)
        require_non_empty_string(
            payload,
            "runtime_image_ref_file",
            label=label,
            require_present=False,
        )
        host = require_non_empty_string(payload, "host", label=label, require_present=False)
        requested_workers = require_int(
            payload,
            "requested_workers",
            label=label,
            minimum=1,
            require_present=False,
        )
        machine = resolve_machine(load_machine_registry(), host)
        if requested_workers > machine.limits.max_parallel_containers:
            raise ValueError(
                f"{label}.requested_workers={requested_workers} exceeds "
                f"{host} capacity={machine.limits.max_parallel_containers}"
            )

    if kind == "eval_contract":
        if not payload.get("artifact_ref") and not payload.get("model_path"):
            raise ValueError(f"{label} must define artifact_ref or model_path")


def load_benchmark_profile(path: Path) -> BenchmarkProfile:
    payload = _profile_payload(path)
    validate_benchmark_profile(payload, label=f"profile file {path}")
    return BenchmarkProfile(path=path, payload=dict(payload))


def load_benchmark_profiles(profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[BenchmarkProfile]:
    if not profile_dir.is_dir():
        raise ValueError(f"benchmark profile directory does not exist: {profile_dir}")
    paths = sorted([*profile_dir.glob("*.yaml"), *profile_dir.glob("*.yml")])
    return [load_benchmark_profile(path) for path in paths]


def find_benchmark_profile(name_or_path: str, *, profile_dir: Path = DEFAULT_PROFILE_DIR) -> BenchmarkProfile:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_benchmark_profile(candidate)
    for profile in load_benchmark_profiles(profile_dir):
        if profile.name == name_or_path or profile.path.stem == name_or_path:
            return profile
    raise ValueError(f"unknown benchmark profile {name_or_path!r}")


def _command(label: str, argv: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> BenchmarkCommand:
    return BenchmarkCommand(label=label, argv=tuple(str(part) for part in argv), cwd=cwd, env=env)


def _runtime_image_from_profile(profile: Mapping[str, Any]) -> str:
    if profile.get("runtime_image_ref"):
        return docker_image_ref(str(profile["runtime_image_ref"]))
    path = Path(str(profile.get("runtime_image_ref_file") or "rlab-train-image.json"))
    if not path.is_file():
        return f"<runtime-image-from:{path}>"
    return docker_image_ref(runtime_image_ref_from_file(path))


def _local_smoke_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    recipe_file = str(profile["recipe_file"])
    machine = str(profile.get("machine") or "local-macbook")
    run_target = str(profile.get("run_target") or machine)
    enqueue = [
        "rlab",
        "train",
        "--recipe-file",
        recipe_file,
        "--run-target",
        run_target,
    ]
    if profile.get("runtime_image_ref_file"):
        enqueue.extend(["--runtime-image-ref-file", str(profile["runtime_image_ref_file"])])
    if profile.get("runtime_image_ref"):
        enqueue.extend(["--runtime-image-ref", str(profile["runtime_image_ref"])])
    for override in string_list(profile.get("recipe_overrides", ()), label="recipe_overrides"):
        enqueue.extend(["--set", override])
    return [
        _command("enqueue-local-smoke", enqueue),
        _command(
            "local-fleet-shepherd-once",
            [
                "rlab",
                "fleet",
                "shepherd",
                "--machine",
                machine,
                "--limit",
                str(profile.get("workers", 1)),
                "--once",
            ],
        ),
        _command(
            "local-fleet-watch",
            [
                "rlab",
                "fleet",
                "watch",
                "--machine",
                machine,
                "--once",
                "--no-tui",
            ],
        ),
    ]


def _env_throughput_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    script = str(profile.get("script") or "experiments/scripts/benchmarks/benchmark_env_sps.py")
    commands: list[BenchmarkCommand] = []
    for mode in string_list(profile.get("modes", ["fast"]), label="modes"):
        for envs in int_list(profile.get("envs", [1]), label="envs"):
            commands.append(
                _command(
                    f"{mode}-{envs}env",
                    [
                        sys.executable,
                        script,
                        "--env-provider",
                        str(profile.get("env_provider", "stable-retro-turbo")),
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


def _train_profile_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    document = load_recipe_document(
        Path(str(profile["recipe_file"])),
        recipe_overrides=profile.get("recipe_overrides", ()),
    )
    config = dict(require_mapping(document["train_config"], label="train_config"))
    config.pop("early_stop", None)
    config.pop("checkpoint_eval_stages", None)
    config["run_name"] = str(
        profile.get("run_name") or f"benchmark_{_slug(str(profile['name']))}"
    )
    config["run_description"] = str(
        profile.get("run_description")
        or f"Benchmark profile {profile['name']} PPO loop probe."
    )
    from rlab.train_config import build_train_command_from_fields

    return [_command("train", build_train_command_from_fields(config))]


def _container_smoke_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    image = _runtime_image_from_profile(profile)
    argv = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "rlab-container-entrypoint",
        image,
        "rlab-container-smoke",
    ]
    return [_command("container-smoke", argv)]


def _fleet_capacity_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    recipe_file = str(profile["recipe_file"])
    host = str(profile["host"])
    commands = [
        _command(
            "enqueue-train",
            [
                "rlab",
                "train",
                "--recipe-file",
                recipe_file,
                "--runtime-image-ref-file",
                str(profile["runtime_image_ref_file"]),
            ],
        ),
        _command(
            "fleet-shepherd-once",
            [
                "rlab",
                "fleet",
                "shepherd",
                "--machine",
                host,
                "--limit",
                str(profile["requested_workers"]),
                "--once",
                "--dry-run",
            ],
        ),
        _command(
            "fleet-watch",
            [
                "rlab",
                "fleet",
                "watch",
                "--machine",
                host,
                "--once",
                "--no-tui",
            ],
        ),
    ]
    return commands


def _eval_contract_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    argv = [
        "rlab",
        "eval",
        "--game",
        str(profile.get("game", "SuperMarioBros-Nes-v0")),
        "--episodes",
        str(profile.get("episodes", 5)),
        "--max-steps",
        str(profile.get("max_steps", 4500)),
    ]
    if profile.get("artifact_ref"):
        argv.extend(["--artifact", str(profile["artifact_ref"])])
    else:
        argv.extend(["--model", str(profile["model_path"])])
    return [_command("eval-contract", argv)]


def build_benchmark_commands(profile: BenchmarkProfile) -> list[BenchmarkCommand]:
    payload = profile.payload
    kind = profile.kind
    if kind == "local_smoke":
        return _local_smoke_commands(payload)
    if kind == "container_smoke":
        return _container_smoke_commands(payload)
    if kind == "env_throughput":
        return _env_throughput_commands(payload)
    if kind in {"artifact_storage_smoke", "ppo_loop_throughput"}:
        return _train_profile_commands(payload)
    if kind == "fleet_capacity":
        return _fleet_capacity_commands(payload)
    if kind == "eval_contract":
        return _eval_contract_commands(payload)
    raise ValueError(f"unsupported benchmark profile kind {kind!r}")
