from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.config_loader import load_mapping_document


DEFAULT_MACHINE_REGISTRY = Path("experiments/machines.yaml")
MACHINE_REGISTRY_KEYS = frozenset({"machines"})
MACHINE_DOCKER_KEYS = frozenset({"command", "gpu_args", "pull_policy"})
MACHINE_PATH_KEYS = frozenset(
    {
        "host_root",
        "payloads_dir",
        "outputs_dir",
        "logs_dir",
        "roms_dir",
        "env_file",
        "container_payloads_dir",
        "container_outputs_dir",
        "container_roms_dir",
    }
)


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
    prewarm_latest_runtime: bool
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


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


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
        "prewarm_latest_runtime",
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
    raw_docker = raw.get("docker")
    if raw_docker is not None and not isinstance(raw_docker, Mapping):
        raise ValueError(f"machine {name!r} docker must be an object")
    docker = raw_docker if isinstance(raw_docker, Mapping) else {}
    unknown_docker_keys = sorted(set(docker) - MACHINE_DOCKER_KEYS)
    if unknown_docker_keys:
        raise ValueError(
            f"machine {name!r} docker has unknown field(s): "
            f"{', '.join(unknown_docker_keys)}"
        )
    default_gpu_args: tuple[str, ...] = ("--gpus", "all") if backend == "docker_ssh" else ()
    raw_limits = raw.get("limits")
    if not isinstance(raw_limits, Mapping):
        raise ValueError(f"machine {name!r} limits must be an object")
    limits_raw = raw_limits
    unknown_limit_keys = sorted(set(limits_raw) - {"max_parallel_containers"})
    if unknown_limit_keys:
        raise ValueError(
            f"machine {name!r} limits has unknown field(s): {', '.join(unknown_limit_keys)}"
        )
    raw_paths = raw.get("paths")
    if raw_paths is not None and not isinstance(raw_paths, Mapping):
        raise ValueError(f"machine {name!r} paths must be an object")
    paths_raw = raw_paths if isinstance(raw_paths, Mapping) else {}
    unknown_path_keys = sorted(set(paths_raw) - MACHINE_PATH_KEYS)
    if unknown_path_keys:
        raise ValueError(
            f"machine {name!r} paths has unknown field(s): {', '.join(unknown_path_keys)}"
        )
    host_root = str(paths_raw.get("host_root") or "/home/tsilva/rlab")
    return MachineConfig(
        name=name,
        backend=backend,
        ssh_target=ssh_target,
        ssh_options=_tuple(raw.get("ssh_options")),
        docker_command=_tuple(docker.get("command") or ("docker",)),
        docker_gpu_args=_tuple(docker.get("gpu_args", default_gpu_args)),
        pull_policy=str(docker.get("pull_policy") or raw.get("pull_policy") or "always"),
        prewarm_latest_runtime=_bool(
            raw.get("prewarm_latest_runtime", False),
            label=f"machine {name} prewarm_latest_runtime",
        ),
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
    unknown_root_keys = sorted(set(data) - MACHINE_REGISTRY_KEYS)
    if unknown_root_keys:
        raise ValueError(
            f"{path} has unknown root field(s): {', '.join(unknown_root_keys)}"
        )
    raw_machines = data.get("machines")
    if not isinstance(raw_machines, Mapping) or not raw_machines:
        raise ValueError(f"{path} must define machines")
    invalid_machines = sorted(str(name) for name, raw in raw_machines.items() if not isinstance(raw, Mapping))
    if invalid_machines:
        raise ValueError(
            f"{path} machine definitions must be objects: {', '.join(invalid_machines)}"
        )
    machines = {
        str(name): _machine_from_raw(str(name), raw) for name, raw in raw_machines.items()
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
