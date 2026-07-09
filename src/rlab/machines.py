from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MACHINE_REGISTRY = Path("experiments/machines.yaml")


@dataclass(frozen=True)
class MachineLimits:
    max_parallel_containers: int
    max_train_containers: int | None = None


@dataclass(frozen=True)
class MachinePaths:
    host_root: str
    payloads_dir: str
    outputs_dir: str
    logs_dir: str
    roms_dir: str
    env_file: str
    container_payloads_dir: str
    container_outputs_dir: str
    container_roms_dir: str


@dataclass(frozen=True)
class MachineConfig:
    name: str
    backend: str
    run_target: str
    ssh_target: str
    ssh_options: tuple[str, ...]
    docker_command: tuple[str, ...]
    docker_gpu_args: tuple[str, ...]
    pull_policy: str
    limits: MachineLimits
    paths: MachinePaths

    def max_containers_for_kind(self, job_kind: str) -> int:
        if job_kind == "train" and self.limits.max_train_containers is not None:
            return self.limits.max_train_containers
        return self.limits.max_parallel_containers


@dataclass(frozen=True)
class MachineRegistry:
    machines: dict[str, MachineConfig]


def _tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected string/list value, got {type(value).__name__}")


def _positive_int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{label} must be at least 1")
    return result


def _optional_positive_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label=label)


def load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a config object")
    return data


def _machine_from_raw(name: str, raw: Mapping[str, Any]) -> MachineConfig:
    backend = str(raw.get("backend") or "").strip()
    if backend not in {"docker_ssh", "local_docker"}:
        raise ValueError(f"machine {name!r} backend must be docker_ssh or local_docker")
    run_target = str(raw.get("run_target") or "").strip()
    if not run_target:
        raise ValueError(f"machine {name!r} must define run_target")
    ssh_target = str(raw.get("ssh_target") or "").strip()
    if backend == "docker_ssh" and not ssh_target:
        raise ValueError(f"machine {name!r} backend docker_ssh requires ssh_target")
    docker = raw.get("docker") if isinstance(raw.get("docker"), Mapping) else {}
    default_gpu_args: tuple[str, ...] = ("--gpus", "all") if backend == "docker_ssh" else ()
    limits_raw = raw.get("limits") if isinstance(raw.get("limits"), Mapping) else {}
    paths_raw = raw.get("paths") if isinstance(raw.get("paths"), Mapping) else {}
    host_root = str(paths_raw.get("host_root") or "/home/tsilva/rlab")
    return MachineConfig(
        name=name,
        backend=backend,
        run_target=run_target,
        ssh_target=ssh_target,
        ssh_options=_tuple(raw.get("ssh_options")),
        docker_command=_tuple(docker.get("command") or ("docker",)),
        docker_gpu_args=_tuple(docker.get("gpu_args", default_gpu_args)),
        pull_policy=str(docker.get("pull_policy") or "always"),
        limits=MachineLimits(
            max_parallel_containers=_positive_int(
                limits_raw.get("max_parallel_containers"),
                label=f"machine {name} limits.max_parallel_containers",
            ),
            max_train_containers=_optional_positive_int(
                limits_raw.get("max_train_containers"),
                label=f"machine {name} limits.max_train_containers",
            ),
        ),
        paths=MachinePaths(
            host_root=host_root,
            payloads_dir=str(paths_raw.get("payloads_dir") or f"{host_root}/payloads"),
            outputs_dir=str(paths_raw.get("outputs_dir") or f"{host_root}/outputs"),
            logs_dir=str(paths_raw.get("logs_dir") or f"{host_root}/logs"),
            roms_dir=str(paths_raw.get("roms_dir") or "/home/tsilva/roms"),
            env_file=str(paths_raw.get("env_file") or f"{host_root}/.env.runner"),
            container_payloads_dir=str(
                paths_raw.get("container_payloads_dir") or "/input/payloads"
            ),
            container_outputs_dir=str(
                paths_raw.get("container_outputs_dir") or "/output"
            ),
            container_roms_dir=str(paths_raw.get("container_roms_dir") or "/roms"),
        ),
    )


def load_machine_registry(path: Path = DEFAULT_MACHINE_REGISTRY) -> MachineRegistry:
    data = load_config_file(path)
    raw_machines = data.get("machines")
    if not isinstance(raw_machines, Mapping) or not raw_machines:
        raise ValueError(f"{path} must define machines")
    machines = {
        str(name): _machine_from_raw(str(name), raw)
        for name, raw in raw_machines.items()
        if isinstance(raw, Mapping)
    }
    if not machines:
        raise ValueError(f"{path} must define at least one machine")
    return MachineRegistry(machines=machines)


def resolve_machine(registry: MachineRegistry, name: str) -> MachineConfig:
    machine_name = str(name or "").strip()
    if machine_name not in registry.machines:
        known = ", ".join(sorted(registry.machines))
        raise ValueError(f"unknown machine {machine_name!r}; known machines: {known}")
    return registry.machines[machine_name]
