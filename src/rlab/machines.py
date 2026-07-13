from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.config_loader import load_mapping_document


DEFAULT_MACHINE_REGISTRY = Path("experiments/machines.yaml")


@dataclass(frozen=True)
class MachineLimits:
    max_parallel_containers: int


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
    ssh_target: str
    ssh_options: tuple[str, ...]
    docker_command: tuple[str, ...]
    docker_gpu_args: tuple[str, ...]
    pull_policy: str
    limits: MachineLimits
    paths: MachinePaths

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


def load_config_file(path: Path) -> dict[str, Any]:
    return load_mapping_document(path, label=str(path))


def _machine_from_raw(name: str, raw: Mapping[str, Any]) -> MachineConfig:
    allowed_keys = {
        "backend",
        "ssh_target",
        "ssh_options",
        "docker",
        "limits",
        "paths",
        "pull_policy",
    }
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"machine {name!r} has unknown field(s): {', '.join(unknown_keys)}")
    backend = str(raw.get("backend") or "").strip()
    if backend not in {"docker_ssh", "local_docker"}:
        raise ValueError(f"machine {name!r} backend must be docker_ssh or local_docker")
    ssh_target = str(raw.get("ssh_target") or "").strip()
    if backend == "docker_ssh" and not ssh_target:
        raise ValueError(f"machine {name!r} backend docker_ssh requires ssh_target")
    docker = raw.get("docker") if isinstance(raw.get("docker"), Mapping) else {}
    default_gpu_args: tuple[str, ...] = ("--gpus", "all") if backend == "docker_ssh" else ()
    limits_raw = raw.get("limits") if isinstance(raw.get("limits"), Mapping) else {}
    unknown_limit_keys = sorted(set(limits_raw) - {"max_parallel_containers"})
    if unknown_limit_keys:
        raise ValueError(
            f"machine {name!r} limits has unknown field(s): {', '.join(unknown_limit_keys)}"
        )
    paths_raw = raw.get("paths") if isinstance(raw.get("paths"), Mapping) else {}
    host_root = str(paths_raw.get("host_root") or "/home/tsilva/rlab")
    return MachineConfig(
        name=name,
        backend=backend,
        ssh_target=ssh_target,
        ssh_options=_tuple(raw.get("ssh_options")),
        docker_command=_tuple(docker.get("command") or ("docker",)),
        docker_gpu_args=_tuple(docker.get("gpu_args", default_gpu_args)),
        pull_policy=str(docker.get("pull_policy") or raw.get("pull_policy") or "always"),
        limits=MachineLimits(
            max_parallel_containers=_positive_int(
                limits_raw.get("max_parallel_containers"),
                label=f"machine {name} limits.max_parallel_containers",
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
