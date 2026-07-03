from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlab.cli import build_train_command
from rlab.fleet import docker_image_ref
from rlab.runtime_refs import runtime_image_ref_from_file


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


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(document: Mapping[str, Any], key: str, *, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _require_int(document: Mapping[str, Any], key: str, *, label: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _string_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    return result


def _int_list(value: Any, *, label: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{label}[{index}] must be an integer")
        result.append(item)
    return result


def _is_state_none(value: Any) -> bool:
    return str(value or "").strip().lower() in STATE_NONE_VALUES


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return text or "benchmark"


def _profile_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a profile object")
    return payload


def validate_benchmark_profile(payload: Mapping[str, Any], *, label: str = "profile") -> None:
    _require_mapping(payload, label=label)
    schema_version = _require_int(payload, "schema_version", label=label)
    if schema_version != BENCHMARK_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"{label}.schema_version must be {BENCHMARK_PROFILE_SCHEMA_VERSION}, got {schema_version}"
        )
    _require_string(payload, "name", label=label)
    kind = _require_string(payload, "kind", label=label)
    if kind not in ALLOWED_KINDS:
        known = ", ".join(sorted(ALLOWED_KINDS))
        raise ValueError(f"{label}.kind must be one of {known}")
    _require_mapping(payload.get("gates", {}), label=f"{label}.gates")

    if kind == "env_throughput":
        game = _require_string(payload, "game", label=label)
        state = _require_string(payload, "state", label=label)
        if _is_state_none(state) and not payload.get("allow_state_none"):
            raise ValueError(
                f"{label}.state must be an actual saved state for {game}; "
                "set allow_state_none=true only for emulator hot-path diagnostics"
            )
        _string_list(payload.get("modes", ["fast"]), label=f"{label}.modes")
        _int_list(payload.get("envs", [1]), label=f"{label}.envs")
        _require_int(payload, "steps", label=label)
        _require_int(payload, "warmup", label=label)

    if kind in {"local_smoke", "ppo_loop_throughput", "artifact_storage_smoke"}:
        config = _require_mapping(payload.get("train_config"), label=f"{label}.train_config")
        _require_string(config, "game", label=f"{label}.train_config")
        if kind != "local_smoke":
            _require_int(config, "timesteps", label=f"{label}.train_config")

    if kind == "fleet_capacity":
        if not payload.get("recipe_file") and not payload.get("spec_file"):
            raise ValueError(f"{label}.recipe_file must be a non-empty string")
        if payload.get("recipe_file"):
            _require_string(payload, "recipe_file", label=label)
        else:
            _require_string(payload, "spec_file", label=label)
        _require_string(payload, "runtime_image_ref_file", label=label)

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
    if not paths:
        paths = sorted(profile_dir.glob("*.json"))
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


def _python_train_command(config: Mapping[str, Any]) -> list[str]:
    command = build_train_command(config)
    if command[:3] == ["python", "-m", "rlab.train"]:
        command[0] = sys.executable
    elif command[:3] == ["rlab", "train", "local"]:
        command = [sys.executable, "-m", "rlab.main", *command[1:]]
    return command


def _runtime_image_from_profile(profile: Mapping[str, Any]) -> str:
    if profile.get("runtime_image_ref"):
        return docker_image_ref(str(profile["runtime_image_ref"]))
    path = Path(str(profile.get("runtime_image_ref_file") or "rlab-train-image.json"))
    if not path.is_file():
        return f"<runtime-image-from:{path}>"
    return docker_image_ref(runtime_image_ref_from_file(path))


def _local_smoke_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    config = dict(_require_mapping(profile["train_config"], label="train_config"))
    run_name = str(
        config.get("run_name")
        or profile.get("run_name")
        or f"benchmark_{_slug(str(profile['name']))}"
    )
    config.setdefault("preset", "smoke")
    config.setdefault("run_name", run_name)
    config.setdefault(
        "run_description",
        f"Benchmark profile {profile['name']} local smoke run.",
    )
    commands = [_command("train-smoke", _python_train_command(config))]
    eval_cfg = dict(_require_mapping(profile.get("eval", {}), label="eval"))
    if eval_cfg.get("enabled", True):
        commands.append(
            _command(
                "eval-smoke",
                [
                    sys.executable,
                    "-m",
                    "rlab.eval",
                    "--game",
                    str(config["game"]),
                    "--model",
                    str(Path(str(config.get("runs_dir") or "runs")) / run_name / "final_model.zip"),
                    "--episodes",
                    str(eval_cfg.get("episodes", 2)),
                    "--max-steps",
                    str(eval_cfg.get("max_steps", config.get("max_episode_steps", 600))),
                ],
            )
        )
    return commands


def _env_throughput_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    script = str(profile.get("script") or "experiments/scripts/benchmarks/benchmark_env_sps.py")
    commands: list[BenchmarkCommand] = []
    for mode in _string_list(profile.get("modes", ["fast"]), label="modes"):
        for envs in _int_list(profile.get("envs", [1]), label="envs"):
            commands.append(
                _command(
                    f"{mode}-{envs}env",
                    [
                        sys.executable,
                        script,
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
                    ],
                    env={"STABLE_RETRO_DISABLE_AUDIO": "1"},
                )
            )
    return commands


def _train_profile_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    config = dict(_require_mapping(profile["train_config"], label="train_config"))
    config.setdefault("run_name", f"benchmark_{_slug(str(profile['name']))}")
    config.setdefault("run_description", f"Benchmark profile {profile['name']} PPO loop probe.")
    config.setdefault("eval_freq", 0)
    config.setdefault("eval_episodes", 0)
    return [_command("train", _python_train_command(config))]


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
    recipe_file = str(profile.get("recipe_file") or profile.get("spec_file"))
    commands = [
        _command(
            "enqueue-train",
            [
                sys.executable,
                "-m",
                "rlab.main",
                "train",
                "--recipe-file",
                recipe_file,
                "--runtime-image-ref-file",
                str(profile["runtime_image_ref_file"]),
            ],
        ),
        _command("fleet-plan", [sys.executable, "-m", "rlab.main", "fleet", "plan"]),
        _command("fleet-reconcile", [sys.executable, "-m", "rlab.main", "fleet", "reconcile"]),
    ]
    return commands


def _eval_contract_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    argv = [
        sys.executable,
        "-m",
        "rlab.eval",
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
