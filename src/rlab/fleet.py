from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

try:  # Rich gives the watch TUI real terminal panels while keeping plain fallback available.
    from rich import box as rich_box
    from rich.columns import Columns as RichColumns
    from rich.console import Console as RichConsole
    from rich.console import Group as RichGroup
    from rich.panel import Panel as RichPanel
    from rich.table import Table as RichTable
    from rich.text import Text as RichText
except ImportError:  # pragma: no cover - exercised only when optional transitive dep is absent.
    rich_box = None
    RichColumns = None
    RichConsole = None
    RichGroup = None
    RichPanel = None
    RichTable = None
    RichText = None

from rlab.job_queue import (
    active_job_launches,
    claim_job_launch,
    connect,
    database_url,
    finish_job_launch_from_result,
    job_payload_for_launch,
    list_stale_train_jobs,
    mark_stale_train_jobs_failed,
    mark_job_launch_running,
    new_launch_id,
    release_job_launch,
)
from rlab.compute_targets import instance_defaults, load_instance_config
from rlab.config_loader import load_composed_mapping
from rlab.json_utils import json_safe
from rlab.machines import (
    DEFAULT_MACHINE_REGISTRY,
    MachineConfig,
    MachineRegistry,
    load_machine_registry,
    resolve_machine,
)
from rlab.monitoring.state import (
    DeviceProbe,
    device_key_from_run_target,
    devices_from_jobs,
    infer_device_key,
    live_device_probes,
)
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    latest_runtime_image_ref,
    normalize_runtime_image_ref,
    RuntimeImageInfo,
    recent_runtime_images,
    runtime_image_digest_slug,
    runtime_image_ref_from_file,
)


DEFAULT_INSTANCES_CONFIG = Path("experiments/instances.yaml")
DEFAULT_CAPACITY_POLICY = Path("experiments/policies/capacity_policy.yaml")
DEFAULT_WATCH_LATEST_INTERVAL_SECONDS = 15.0
DEFAULT_WATCH_STALE_OLDER_THAN_SECONDS = 300
DEFAULT_WATCH_STALE_LIMIT = 50
LABEL_PREFIX = "rlab."
MANAGED_LABEL = f"{LABEL_PREFIX}managed"
CONFIG_HASH_LABEL = f"{LABEL_PREFIX}config-hash"
DEFAULT_RUNNER_AUTOSCALE_MIN_WORKERS = 1
DEFAULT_RUNNER_AUTOSCALE_MAX_WORKERS = 16
WORKER_KIND_TRAIN = "train"
JOB_CONTAINER_LABEL = f"{LABEL_PREFIX}job-container"
JOB_ID_LABEL = f"{LABEL_PREFIX}job-id"
JOB_KIND_LABEL = f"{LABEL_PREFIX}job-kind"
LAUNCH_ID_LABEL = f"{LABEL_PREFIX}launch-id"
MACHINE_LABEL = f"{LABEL_PREFIX}machine"
OUTPUT_URI_LABEL = f"{LABEL_PREFIX}output-uri"


@dataclass(frozen=True)
class HostConfig:
    name: str
    ssh_target: str
    ssh_options: tuple[str, ...]
    run_target: str
    max_workers: int
    base_dir: str
    env_file: str
    runs_dir: str
    logs_dir: str
    rom_dir: str
    state_dir: str
    container_runs_dir: str
    container_logs_dir: str
    container_rom_dir: str
    log_dir_in_container: str
    gpu_test_image: str
    docker_command: tuple[str, ...]
    docker_network: str | None
    pull_policy: str
    extra_env: tuple[str, ...]


@dataclass(frozen=True)
class ProfilePolicy:
    profile_id: str
    hosts: tuple[str, ...]


@dataclass(frozen=True)
class FleetConfig:
    hosts: dict[str, HostConfig]
    profile_policies: tuple[ProfilePolicy, ...]


@dataclass(frozen=True)
class QueueDemand:
    profile_id: str | None
    runtime_image_ref: str
    run_target: str | None
    pending_count: int
    running_count: int
    oldest_job_id: int

    @property
    def total(self) -> int:
        return self.pending_count + self.running_count


@dataclass(frozen=True)
class ActiveLease:
    lease_owner: str
    profile_id: str | None
    runtime_image_ref: str
    run_target: str | None
    running_count: int
    worker_kind: str = WORKER_KIND_TRAIN


@dataclass(frozen=True)
class RunningJob:
    id: int
    lease_owner: str
    profile_id: str | None
    runtime_image_ref: str
    run_target: str | None
    run_name: str | None
    started_at: Any
    heartbeat_at: Any


@dataclass(frozen=True)
class StaleTrainJob:
    host: str
    id: int
    profile_id: str | None
    runtime_image_ref: str | None
    run_target: str | None
    run_name: str | None
    lease_owner: str | None
    heartbeat_at: Any
    execute: bool


@dataclass(frozen=True)
class DeploymentKey:
    host: str
    profile_id: str | None
    runtime_image_ref: str
    run_target: str | None
    worker_kind: str = WORKER_KIND_TRAIN
    replica: int | None = None


@dataclass(frozen=True)
class DesiredDeployment:
    key: DeploymentKey
    name: str
    worker_prefix: str
    workers: int
    config_hash: str
    labels: dict[str, str]
    command: list[str]
    pending_count: int
    running_count: int


@dataclass(frozen=True)
class ExistingContainer:
    host: str
    name: str
    state: str
    status: str
    image: str
    labels: dict[str, str]

    @property
    def key(self) -> DeploymentKey | None:
        profile_id = self.labels.get(f"{LABEL_PREFIX}profile")
        runtime_image_ref = self.labels.get(f"{LABEL_PREFIX}runtime-image-ref")
        if not runtime_image_ref:
            return None
        profile_id = profile_id or None
        run_target = self.labels.get(f"{LABEL_PREFIX}run-target") or None
        worker_kind = self.labels.get(f"{LABEL_PREFIX}worker-kind") or WORKER_KIND_TRAIN
        replica_label = self.labels.get(f"{LABEL_PREFIX}replica")
        replica = int(replica_label) if replica_label not in (None, "") else None
        return DeploymentKey(
            host=self.host,
            profile_id=profile_id,
            runtime_image_ref=runtime_image_ref,
            run_target=run_target,
            worker_kind=worker_kind,
            replica=replica,
        )


@dataclass(frozen=True)
class FleetAction:
    kind: str
    host: str
    container: str
    reason: str
    commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class FleetPlan:
    desired: tuple[DesiredDeployment, ...]
    existing: tuple[ExistingContainer, ...]
    actions: tuple[FleetAction, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ActionResult:
    kind: str
    host: str
    container: str
    exit_code: int
    output: str = ""


@dataclass(frozen=True)
class LatestWatchSnapshot:
    captured_at: datetime
    config: FleetConfig
    runtime_image_ref: str
    demands: tuple[QueueDemand, ...]
    leases: tuple[ActiveLease, ...]
    jobs: tuple[RunningJob, ...]
    plan: FleetPlan
    recent_images: tuple[RuntimeImageInfo, ...] = ()
    devices: tuple[dict[str, Any], ...] = ()
    stale_train_jobs: tuple[StaleTrainJob, ...] = ()
    down_hosts: tuple[str, ...] = ()
    action_results: tuple[ActionResult, ...] = ()
    execute: bool = False
    interval: float = 30.0


@dataclass(frozen=True)
class RuntimeImageContext:
    runtime_image_ref: str | None
    recent_images: tuple[RuntimeImageInfo, ...] = ()
    warnings: tuple[str, ...] = ()


class RuntimeImageResolver:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        default_latest: bool = False,
        cache_seconds: float = 0.0,
    ) -> None:
        self.args = args
        self.default_latest = default_latest
        self.cache_seconds = cache_seconds
        self._cached_context: RuntimeImageContext | None = None
        self._cached_at = 0.0

    def resolve(self) -> RuntimeImageContext:
        now = time.monotonic()
        if (
            self._cached_context is not None
            and self.cache_seconds > 0
            and now - self._cached_at < self.cache_seconds
        ):
            return self._cached_context
        runtime_image_ref, recent_images, warnings = runtime_image_context_from_args(
            self.args,
            default_latest=self.default_latest,
        )
        context = RuntimeImageContext(
            runtime_image_ref=runtime_image_ref,
            recent_images=recent_images,
            warnings=warnings,
        )
        self._cached_context = context
        self._cached_at = now
        return context


@dataclass
class WatchLatestLock:
    path: Path
    handle: TextIO


@dataclass(frozen=True)
class ShepherdLock:
    machine: str
    key: str


class WatchLatestLockBusy(RuntimeError):
    def __init__(self, path: Path, owner: str) -> None:
        super().__init__(f"another watch session is already running: {path}")
        self.path = path
        self.owner = owner


class ShepherdLockBusy(RuntimeError):
    def __init__(self, machine: str) -> None:
        super().__init__(f"another shepherd is already running for machine={machine}")
        self.machine = machine


@dataclass(frozen=True)
class MachineWatchSnapshot:
    captured_at: datetime
    machine: MachineConfig
    containers: tuple["JobContainer", ...]
    launches: tuple[dict[str, Any], ...]
    queue_counts: Mapping[str, Mapping[str, int]]
    result_present: Mapping[str, bool]
    warnings: tuple[str, ...] = ()


def load_json_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a config object")
    return data


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected string or list, got {type(value).__name__}")


def _host_config_from_machine(
    *,
    machine: MachineConfig,
    instances: Mapping[str, Any],
) -> HostConfig:
    if machine.backend != "docker_ssh":
        raise ValueError(f"fleet host {machine.name!r} must use backend docker_ssh")
    run_target = machine.run_target
    instance = instance_defaults(dict(instances), run_target)
    max_workers = int(machine.limits.max_parallel_containers)
    if max_workers < 1:
        raise ValueError(f"machine {machine.name!r} max_parallel_containers must be at least 1")
    return HostConfig(
        name=machine.name,
        ssh_target=machine.ssh_target,
        ssh_options=machine.ssh_options,
        run_target=str(instance.get("name", run_target)),
        max_workers=max_workers,
        base_dir=machine.paths.host_root,
        env_file=machine.paths.env_file,
        runs_dir=f"{machine.paths.host_root.rstrip('/')}/runs",
        logs_dir=machine.paths.logs_dir,
        rom_dir=machine.paths.roms_dir,
        state_dir=f"{machine.paths.host_root.rstrip('/')}/fleet",
        container_runs_dir="/root/rlab/runs",
        container_logs_dir="/root/rlab/logs",
        container_rom_dir=machine.paths.container_roms_dir,
        log_dir_in_container="/root/rlab/logs/train_runner",
        gpu_test_image="nvidia/cuda:12.9.1-base-ubuntu22.04",
        docker_command=machine.docker_command,
        docker_network=None,
        pull_policy=machine.pull_policy,
        extra_env=(),
    )


def load_fleet_config(
    repo_root: Path,
    *,
    instances_path: Path | None = None,
    machines_path: Path | None = None,
) -> FleetConfig:
    machines_config_path = resolve_repo_path(repo_root, machines_path, DEFAULT_MACHINE_REGISTRY)
    machines_data = load_json_file(machines_config_path)
    registry = load_machine_registry(machines_config_path)
    instances = load_instance_config(
        repo_root,
        resolve_repo_path(repo_root, instances_path, DEFAULT_INSTANCES_CONFIG),
    )
    hosts = {
        name: _host_config_from_machine(machine=machine, instances=instances)
        for name, machine in registry.machines.items()
        if machine.backend == "docker_ssh"
    }
    if not hosts:
        raise ValueError(f"{machines_config_path} must define at least one docker_ssh machine")
    policies_raw = machines_data.get("profile_policies", [{"profile_id": "*", "hosts": list(hosts)}])
    if not isinstance(policies_raw, list):
        raise ValueError("profile_policies must be a list")
    policies = []
    for item in policies_raw:
        if not isinstance(item, dict):
            raise ValueError("profile_policies entries must be objects")
        profile_id = str(item.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_policies entries must define profile_id")
        policy_hosts = _tuple(item.get("hosts"))
        unknown = [host for host in policy_hosts if host not in hosts]
        if unknown:
            raise ValueError(f"profile policy {profile_id!r} references unknown hosts: {unknown}")
        policies.append(ProfilePolicy(profile_id=profile_id, hosts=policy_hosts))
    return FleetConfig(hosts=hosts, profile_policies=tuple(policies))


def load_capacity_policy(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    return load_json_file(resolve_repo_path(repo_root, path, DEFAULT_CAPACITY_POLICY))


def goal_contract_path(repo_root: Path, goal_slug: str) -> Path:
    goals_dir = repo_root / "experiments" / "goals"
    for filename in ("_goal.yaml", "goal.yaml"):
        for candidate in sorted(goals_dir.rglob(f"{goal_slug}/{filename}")):
            if ".deprecated" not in candidate.parts and candidate.is_file():
                return candidate
    return goals_dir / goal_slug / "goal.yaml"


def load_goal_document(repo_root: Path, goal_slug: str) -> Mapping[str, Any]:
    path = goal_contract_path(repo_root, goal_slug)
    if not path.is_file():
        raise ValueError(f"unknown goal {goal_slug!r}: {path} does not exist")
    return load_composed_mapping(path, cycle_label="goal").document


def discover_active_goal_slugs(repo_root: Path) -> tuple[str, ...]:
    return ()


def validate_capacity_policy(policy: Mapping[str, Any], config: FleetConfig) -> None:
    lanes = policy.get("lanes", [])
    if not isinstance(lanes, list):
        raise ValueError("capacity_policy lanes must be a list")
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise ValueError("capacity_policy lane entries must be objects")
        name = str(lane.get("name") or "<unnamed>")
        manager = str(lane.get("manager") or "").strip()
        host_name = str(lane.get("host") or "").strip()
        if not host_name:
            if manager in {"rlab_fleet", "rlab fleet"}:
                raise ValueError(f"capacity_policy lane {name!r} uses rlab_fleet but has no host")
            continue
        if host_name not in config.hosts:
            raise ValueError(f"capacity_policy lane {name!r} references unknown host {host_name!r}")
        max_runner_workers = lane.get("max_runner_workers")
        if max_runner_workers is None:
            continue
        try:
            runner_limit = int(max_runner_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"capacity_policy lane {name!r} max_runner_workers must be an integer"
            ) from exc
        if runner_limit < 1:
            raise ValueError(f"capacity_policy lane {name!r} max_runner_workers must be at least 1")
        host_limit = config.hosts[host_name].max_workers
        if runner_limit > host_limit:
            raise ValueError(
                f"capacity_policy lane {name!r} max_runner_workers={runner_limit} "
                f"exceeds {host_name} max_workers={host_limit}"
            )


def resolve_repo_path(repo_root: Path, path: Path | None, default: Path) -> Path:
    candidate = path or default
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _candidate_repo_roots(start: Path) -> tuple[Path, ...]:
    base = start if start.is_dir() else start.parent
    return (base, *base.parents)


def _is_fleet_repo_root(path: Path) -> bool:
    return (path / DEFAULT_MACHINE_REGISTRY).is_file()


def default_repo_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in _candidate_repo_roots(start):
            if _is_fleet_repo_root(candidate):
                return candidate.resolve()
    return Path.cwd().resolve()


def filter_config_to_host(config: FleetConfig, host_name: str | None) -> FleetConfig:
    if not host_name:
        return config
    if host_name not in config.hosts:
        known = ", ".join(sorted(config.hosts))
        raise ValueError(f"unknown fleet host {host_name!r}; known hosts: {known}")
    policies = []
    for policy in config.profile_policies:
        hosts = tuple(host for host in policy.hosts if host == host_name)
        if hosts:
            policies.append(ProfilePolicy(profile_id=policy.profile_id, hosts=hosts))
    if not policies:
        policies.append(ProfilePolicy(profile_id="*", hosts=(host_name,)))
    return FleetConfig(hosts={host_name: config.hosts[host_name]}, profile_policies=tuple(policies))


def docker_image_ref(runtime_image_ref: str) -> str:
    normalized = normalize_runtime_image_ref(runtime_image_ref)
    return normalized.removeprefix("docker:")


def sanitize_slug(value: str, *, limit: int = 40) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "value"
    return slug[:limit].strip("-") or "value"


def deployment_name(key: DeploymentKey) -> str:
    digest = runtime_image_digest_slug(key.runtime_image_ref)
    host = sanitize_slug(key.host, limit=16)
    profile = sanitize_slug(key.profile_id or "any-profile", limit=44)
    target = sanitize_slug(key.run_target or "any", limit=16)
    return f"rlab-{host}-{target}-{profile}-{digest}"[:120].strip("-")


def config_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def shell_join(parts: Sequence[str]) -> str:
    return shlex.join([str(part) for part in parts])


def docker_command(host: HostConfig, args: Sequence[str]) -> list[str]:
    return [*host.docker_command, *args]


def docker_run_command(host: HostConfig, desired: DesiredDeployment) -> list[str]:
    image = docker_image_ref(desired.key.runtime_image_ref)
    cmd = docker_command(
        host,
        [
            "run",
            "-d",
            "--name",
            desired.name,
            "--restart",
            "unless-stopped",
            "--gpus",
            "all",
            "--env-file",
            host.env_file,
            "-e",
            f"RLAB_ROM_DIR={host.container_rom_dir}",
            "-v",
            f"{host.rom_dir}:{host.container_rom_dir}:ro",
            "-v",
            f"{host.runs_dir}:{host.container_runs_dir}",
            "-v",
            f"{host.logs_dir}:{host.container_logs_dir}",
        ],
    )
    for value in host.extra_env:
        cmd.extend(["-e", value])
    if host.docker_network:
        cmd.extend(["--network", host.docker_network])
    for key, value in sorted(desired.labels.items()):
        cmd.extend(["--label", f"{key}={value}"])
    cmd.extend([image, "rlab-container-entrypoint", "rlab", WORKER_KIND_TRAIN, "worker"])
    cmd.extend(desired.command)
    return cmd


def build_desired_deployment(
    *,
    host: HostConfig,
    key: DeploymentKey,
    workers: int,
    pending_count: int,
    running_count: int,
) -> DesiredDeployment:
    name = deployment_name(key)
    worker_prefix = name
    command = [
        "--runtime-image-ref",
        key.runtime_image_ref,
        "--run-target",
        key.run_target or host.run_target,
        "--workers",
        str(workers),
        "--autoscale",
        "--min-workers",
        str(DEFAULT_RUNNER_AUTOSCALE_MIN_WORKERS),
        "--max-workers",
        str(DEFAULT_RUNNER_AUTOSCALE_MAX_WORKERS),
        "--worker-id",
        worker_prefix,
        "--log-dir",
        host.log_dir_in_container,
    ]
    if key.profile_id:
        command = ["--profile", key.profile_id, *command]
    hash_input = {
        "host": host.name,
        "worker_kind": key.worker_kind,
        "profile_id": key.profile_id,
        "runtime_image_ref": key.runtime_image_ref,
        "run_target": key.run_target,
        "replica": key.replica,
        "workers": workers,
        "env_file": host.env_file,
        "runs_dir": host.runs_dir,
        "logs_dir": host.logs_dir,
        "rom_dir": host.rom_dir,
        "docker_command": host.docker_command,
        "command": command,
    }
    digest = runtime_image_digest_slug(key.runtime_image_ref)
    labels = {
        MANAGED_LABEL: "true",
        f"{LABEL_PREFIX}host": host.name,
        f"{LABEL_PREFIX}profile": key.profile_id or "",
        f"{LABEL_PREFIX}runtime-image-ref": key.runtime_image_ref,
        f"{LABEL_PREFIX}runtime-digest": digest,
        f"{LABEL_PREFIX}run-target": key.run_target or "",
        f"{LABEL_PREFIX}worker-kind": key.worker_kind,
        f"{LABEL_PREFIX}worker-prefix": worker_prefix,
    }
    if key.replica is not None:
        labels[f"{LABEL_PREFIX}replica"] = str(key.replica)
    labels[CONFIG_HASH_LABEL] = config_hash({**hash_input, "labels": labels})
    return DesiredDeployment(
        key=key,
        name=name,
        worker_prefix=worker_prefix,
        workers=workers,
        config_hash=labels[CONFIG_HASH_LABEL],
        labels=labels,
        command=command,
        pending_count=pending_count,
        running_count=running_count,
    )


def _matching_policy_hosts(config: FleetConfig, profile_id: str) -> tuple[str, ...]:
    wildcard: tuple[str, ...] = ()
    for policy in config.profile_policies:
        if policy.profile_id == profile_id:
            return policy.hosts
        if policy.profile_id == "*":
            wildcard = policy.hosts
    return wildcard or tuple(config.hosts)


def eligible_hosts(config: FleetConfig, demand: QueueDemand) -> list[HostConfig]:
    names = _matching_policy_hosts(config, demand.profile_id or "*")
    hosts = []
    for name in names:
        host = config.hosts[name]
        if demand.run_target and demand.run_target != host.run_target:
            continue
        hosts.append(host)
    return hosts


def allocate_desired_deployments(
    config: FleetConfig,
    demands: Sequence[QueueDemand],
) -> tuple[tuple[DesiredDeployment, ...], tuple[str, ...]]:
    warnings: list[str] = []
    remaining = {name: host.max_workers for name, host in config.hosts.items()}
    desired: list[DesiredDeployment] = []
    sorted_demands = sorted(
        demands,
        key=lambda item: (
            item.running_count == 0,
            item.oldest_job_id,
            item.profile_id or "",
            item.runtime_image_ref,
            item.run_target or "",
        ),
    )
    for demand in sorted_demands:
        hosts = eligible_hosts(config, demand)
        if not hosts:
            warnings.append(
                "no eligible host for "
                f"profile={demand.profile_id or 'any'} target={demand.run_target or 'any'}"
            )
            continue
        chosen = next((host for host in hosts if remaining[host.name] > 0), None)
        if chosen is None:
            warnings.append(
                "capacity exhausted for "
                f"profile={demand.profile_id or 'any'} target={demand.run_target or 'any'}"
            )
            continue
        requested = max(demand.running_count, demand.pending_count, 1)
        workers = min(requested, remaining[chosen.name])
        key = DeploymentKey(
            host=chosen.name,
            profile_id=demand.profile_id,
            runtime_image_ref=demand.runtime_image_ref,
            run_target=demand.run_target,
        )
        desired.append(
            build_desired_deployment(
                host=chosen,
                key=key,
                workers=workers,
                pending_count=demand.pending_count,
                running_count=demand.running_count,
            )
        )
        remaining[chosen.name] -= workers
        if requested > workers:
            warnings.append(
                f"partially allocated {workers}/{requested} workers for "
                f"{demand.profile_id or 'any'} on {chosen.name}"
            )
    return tuple(desired), tuple(warnings)


def demand_index(demands: Sequence[QueueDemand]) -> dict[tuple[str | None, str, str | None], QueueDemand]:
    return {
        (demand.profile_id, demand.runtime_image_ref, demand.run_target): demand
        for demand in demands
    }


def demand_matches_key(demand: QueueDemand, key: DeploymentKey) -> bool:
    if key.worker_kind != WORKER_KIND_TRAIN:
        return False
    if key.profile_id is not None and demand.profile_id != key.profile_id:
        return False
    return demand.runtime_image_ref == key.runtime_image_ref and demand.run_target == key.run_target


def container_can_serve_desired(
    container: ExistingContainer,
    desired: DesiredDeployment,
) -> bool:
    key = container.key
    if key is None:
        return False
    return (
        key.host == desired.key.host
        and key.worker_kind == desired.key.worker_kind
        and key.profile_id is None
        and key.runtime_image_ref == desired.key.runtime_image_ref
        and key.run_target == desired.key.run_target
    )


def active_prefixes(leases: Sequence[ActiveLease]) -> tuple[str, ...]:
    return tuple(sorted({lease.lease_owner for lease in leases if lease.lease_owner}))


def container_has_active_lease(container: ExistingContainer, leases: Sequence[ActiveLease]) -> bool:
    prefix = container.labels.get(f"{LABEL_PREFIX}worker-prefix") or container.name
    return any(lease.lease_owner.startswith(prefix) for lease in leases)


def matching_demand_for_container(
    container: ExistingContainer,
    demands: Sequence[QueueDemand],
) -> QueueDemand | None:
    key = container.key
    if key is None:
        return None
    if key.worker_kind != WORKER_KIND_TRAIN:
        return None
    for demand in demands:
        if key.profile_id is None:
            if demand_matches_key(demand, key) and demand.total > 0:
                return demand
            continue
        if (
            demand.profile_id == key.profile_id
            and demand.runtime_image_ref == key.runtime_image_ref
            and demand.run_target == key.run_target
            and demand.total > 0
        ):
            return demand
    return None


def pull_command(host: HostConfig, runtime_image_ref: str) -> str:
    return shell_join(docker_command(host, ["pull", docker_image_ref(runtime_image_ref)]))


def remove_command(host: HostConfig, name: str) -> str:
    return shell_join(docker_command(host, ["rm", "-f", name]))


def restart_commands(host: HostConfig, desired: DesiredDeployment) -> tuple[str, ...]:
    return (
        pull_command(host, desired.key.runtime_image_ref),
        remove_command(host, desired.name),
        shell_join(docker_run_command(host, desired)),
    )


def start_commands(host: HostConfig, desired: DesiredDeployment) -> tuple[str, ...]:
    return (
        pull_command(host, desired.key.runtime_image_ref),
        shell_join(docker_run_command(host, desired)),
    )


def build_fleet_plan(
    config: FleetConfig,
    demands: Sequence[QueueDemand],
    existing: Sequence[ExistingContainer],
    leases: Sequence[ActiveLease],
) -> FleetPlan:
    desired, allocation_warnings = allocate_desired_deployments(config, demands)
    warnings = list(allocation_warnings)
    desired_by_name = {item.name: item for item in desired}
    existing_by_name = {item.name: item for item in existing}
    actions: list[FleetAction] = []

    for desired_item in desired:
        host = config.hosts[desired_item.key.host]
        current = existing_by_name.get(desired_item.name)
        if current is None:
            wildcard_current = next(
                (
                    item
                    for item in existing
                    if item.state.lower() == "running" and container_can_serve_desired(item, desired_item)
                ),
                None,
            )
            if wildcard_current is not None:
                actions.append(
                    FleetAction(
                        kind="keep",
                        host=host.name,
                        container=wildcard_current.name,
                        reason="unprofiled container already serves this profile demand",
                    )
                )
                continue
            actions.append(
                FleetAction(
                    kind="start",
                    host=host.name,
                    container=desired_item.name,
                    reason="queued or running demand exists",
                    commands=start_commands(host, desired_item),
                )
            )
            continue
        if current.state.lower() != "running":
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} is not running but still owns an active lease")
                continue
            actions.append(
                FleetAction(
                    kind="restart",
                    host=host.name,
                    container=desired_item.name,
                    reason=f"container state is {current.state or 'unknown'}",
                    commands=restart_commands(host, desired_item),
                )
            )
            continue
        if current.labels.get(CONFIG_HASH_LABEL) != desired_item.config_hash:
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} config changed but active lease prevents restart")
                continue
            actions.append(
                FleetAction(
                    kind="recreate",
                    host=host.name,
                    container=desired_item.name,
                    reason="managed container config changed",
                    commands=restart_commands(host, desired_item),
                )
            )
            continue
        actions.append(
            FleetAction(
                kind="keep",
                host=host.name,
                container=desired_item.name,
                reason="container already matches desired state",
            )
        )

    for current in existing:
        key = current.key
        if key is not None and key.worker_kind != WORKER_KIND_TRAIN:
            continue
        if current.name in desired_by_name:
            continue
        matching_demand = matching_demand_for_container(current, demands)
        if matching_demand is not None and matching_demand.total > 0:
            warnings.append(
                f"{current.name} has demand but was not allocated new capacity; leaving it alone"
            )
            continue
        if container_has_active_lease(current, leases):
            warnings.append(f"{current.name} is obsolete but still owns an active lease")
            continue
        actions.append(
            FleetAction(
                kind="remove",
                host=current.host,
                container=current.name,
                reason="no pending or running jobs for this digest/profile/target",
                commands=(remove_command(config.hosts[current.host], current.name),),
            )
        )

    return FleetPlan(
        desired=desired,
        existing=tuple(existing),
        actions=tuple(actions),
        warnings=tuple(warnings),
    )


def build_ensure_runner_plan(
    config: FleetConfig,
    *,
    host_name: str,
    profile_id: str | None,
    runtime_image_ref: str,
    run_target: str | None,
    workers: int | None,
    existing: Sequence[ExistingContainer],
    leases: Sequence[ActiveLease],
) -> FleetPlan:
    if host_name not in config.hosts:
        known = ", ".join(sorted(config.hosts))
        raise ValueError(f"unknown fleet host {host_name!r}; known hosts: {known}")
    host = config.hosts[host_name]
    profile_id = profile_id.strip() if profile_id else None
    target = run_target or host.run_target
    if target != host.run_target:
        raise ValueError(
            f"host {host.name} has run_target={host.run_target!r}; cannot ensure target={target!r}"
        )
    worker_count = workers if workers is not None else host.max_workers
    if worker_count < 1:
        raise ValueError("--workers must be at least 1")
    if worker_count > host.max_workers:
        raise ValueError(f"--workers {worker_count} exceeds {host.name} max_workers={host.max_workers}")
    key = DeploymentKey(
        host=host.name,
        profile_id=profile_id,
        runtime_image_ref=normalize_runtime_image_ref(runtime_image_ref),
        run_target=target,
    )
    desired = build_desired_deployment(
        host=host,
        key=key,
        workers=worker_count,
        pending_count=0,
        running_count=0,
    )
    existing_by_name = {item.name: item for item in existing}
    current = existing_by_name.get(desired.name)
    warnings: list[str] = []
    actions: list[FleetAction] = []
    if current is None:
        actions.append(
            FleetAction(
                kind="start",
                host=host.name,
                container=desired.name,
                reason="explicit ensure-runner request",
                commands=start_commands(host, desired),
            )
        )
    elif current.state.lower() != "running":
        if container_has_active_lease(current, leases):
            warnings.append(f"{current.name} is not running but still owns an active lease")
        else:
            actions.append(
                FleetAction(
                    kind="restart",
                    host=host.name,
                    container=desired.name,
                    reason=f"container state is {current.state or 'unknown'}",
                    commands=restart_commands(host, desired),
                )
            )
    elif current.labels.get(CONFIG_HASH_LABEL) != desired.config_hash:
        if container_has_active_lease(current, leases):
            warnings.append(f"{current.name} config changed but active lease prevents restart")
        else:
            actions.append(
                FleetAction(
                    kind="recreate",
                    host=host.name,
                    container=desired.name,
                    reason="managed container config changed",
                    commands=restart_commands(host, desired),
                )
            )
    else:
        actions.append(
            FleetAction(
                kind="keep",
                host=host.name,
                container=desired.name,
                reason="container already matches desired state",
            )
        )
    return FleetPlan(
        desired=(desired,),
        existing=tuple(existing),
        actions=tuple(actions),
        warnings=tuple(warnings),
    )


def build_explicit_desired_plan(
    config: FleetConfig,
    desired: Sequence[DesiredDeployment],
    existing: Sequence[ExistingContainer],
    leases: Sequence[ActiveLease],
    *,
    reason: str,
) -> FleetPlan:
    existing_by_name = {item.name: item for item in existing}
    warnings: list[str] = []
    actions: list[FleetAction] = []
    for item in desired:
        host = config.hosts[item.key.host]
        current = existing_by_name.get(item.name)
        if current is None:
            actions.append(
                FleetAction(
                    kind="start",
                    host=host.name,
                    container=item.name,
                    reason=reason,
                    commands=start_commands(host, item),
                )
            )
            continue
        if current.state.lower() != "running":
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} is not running but still owns an active lease")
                continue
            actions.append(
                FleetAction(
                    kind="restart",
                    host=host.name,
                    container=item.name,
                    reason=f"container state is {current.state or 'unknown'}",
                    commands=restart_commands(host, item),
                )
            )
            continue
        if current.labels.get(CONFIG_HASH_LABEL) != item.config_hash:
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} config changed but active lease prevents restart")
                continue
            actions.append(
                FleetAction(
                    kind="recreate",
                    host=host.name,
                    container=item.name,
                    reason="managed container config changed",
                    commands=restart_commands(host, item),
                )
            )
            continue
        actions.append(
            FleetAction(
                kind="keep",
                host=host.name,
                container=item.name,
                reason="container already matches desired state",
            )
        )
    return FleetPlan(
        desired=tuple(desired),
        existing=tuple(existing),
        actions=tuple(actions),
        warnings=tuple(warnings),
    )


def combine_plans(primary: FleetPlan, secondary: FleetPlan) -> FleetPlan:
    return FleetPlan(
        desired=(*primary.desired, *secondary.desired),
        existing=primary.existing,
        actions=(*primary.actions, *secondary.actions),
        warnings=(*primary.warnings, *secondary.warnings),
    )


def build_ensure_latest_plan(
    config: FleetConfig,
    *,
    runtime_image_ref: str,
    workers: int | None,
    existing: Sequence[ExistingContainer],
    leases: Sequence[ActiveLease],
    demands: Sequence[QueueDemand],
) -> FleetPlan:
    normalized_ref = normalize_runtime_image_ref(runtime_image_ref)
    desired: list[DesiredDeployment] = []
    warnings: list[str] = []
    actions: list[FleetAction] = []
    for host in selected_hosts(config, None):
        worker_count = workers if workers is not None else host.max_workers
        if worker_count < 1:
            raise ValueError("--workers must be at least 1")
        if worker_count > host.max_workers:
            raise ValueError(
                f"--workers {worker_count} exceeds {host.name} max_workers={host.max_workers}"
            )
        desired.append(
            build_desired_deployment(
                host=host,
                key=DeploymentKey(
                    host=host.name,
                    profile_id=None,
                    runtime_image_ref=normalized_ref,
                    run_target=host.run_target,
                ),
                workers=worker_count,
                pending_count=0,
                running_count=0,
            )
        )

    desired_by_name = {item.name: item for item in desired}
    existing_by_name = {item.name: item for item in existing}
    for item in desired:
        host = config.hosts[item.key.host]
        current = existing_by_name.get(item.name)
        if current is None:
            actions.append(
                FleetAction(
                    kind="start",
                    host=host.name,
                    container=item.name,
                    reason="latest image baseline for active fleet host",
                    commands=start_commands(host, item),
                )
            )
            continue
        if current.state.lower() != "running":
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} is not running but still owns an active lease")
                continue
            actions.append(
                FleetAction(
                    kind="restart",
                    host=host.name,
                    container=item.name,
                    reason=f"latest image container state is {current.state or 'unknown'}",
                    commands=restart_commands(host, item),
                )
            )
            continue
        if current.labels.get(CONFIG_HASH_LABEL) != item.config_hash:
            if container_has_active_lease(current, leases):
                warnings.append(f"{current.name} config changed but active lease prevents restart")
                continue
            actions.append(
                FleetAction(
                    kind="recreate",
                    host=host.name,
                    container=item.name,
                    reason="latest image container config changed",
                    commands=restart_commands(host, item),
                )
            )
            continue
        actions.append(
            FleetAction(
                kind="keep",
                host=host.name,
                container=item.name,
                reason="latest image runner already matches desired state",
            )
        )

    for current in existing:
        key = current.key
        if key is not None and key.worker_kind != WORKER_KIND_TRAIN:
            continue
        if current.name in desired_by_name:
            continue
        matching_demand = matching_demand_for_container(current, demands)
        if matching_demand is not None:
            warnings.append(
                f"{current.name} still has matching pending/running demand; leaving it alone"
            )
            continue
        if container_has_active_lease(current, leases):
            warnings.append(f"{current.name} is not latest but still owns an active lease")
            continue
        actions.append(
            FleetAction(
                kind="remove",
                host=current.host,
                container=current.name,
                reason="not latest baseline and no matching pending/running jobs",
                commands=(remove_command(config.hosts[current.host], current.name),),
            )
        )

    return FleetPlan(
        desired=tuple(desired),
        existing=tuple(existing),
        actions=tuple(actions),
        warnings=tuple(warnings),
    )


QUEUE_DEMAND_SQL = """
SELECT
  profile_id,
  runtime_image_ref,
  run_target,
  COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
  COUNT(*) FILTER (WHERE status = 'running') AS running_count,
  MIN(id) AS oldest_job_id
FROM train_jobs
WHERE runtime_image_ref IS NOT NULL
  AND cancel_requested = FALSE
  AND status IN ('pending', 'running')
GROUP BY profile_id, runtime_image_ref, run_target
ORDER BY oldest_job_id ASC
"""


ACTIVE_LEASE_SQL = """
SELECT
  lease_owner,
  profile_id,
  runtime_image_ref,
  run_target,
  COUNT(*) AS running_count
FROM train_jobs
WHERE status = 'running'
  AND lease_owner IS NOT NULL
  AND runtime_image_ref IS NOT NULL
GROUP BY lease_owner, profile_id, runtime_image_ref, run_target
ORDER BY lease_owner
"""


RUNNING_JOBS_SQL = """
SELECT
  id,
  lease_owner,
  profile_id,
  runtime_image_ref,
  run_target,
  run_name,
  started_at,
  heartbeat_at
FROM train_jobs
WHERE status = 'running'
  AND lease_owner IS NOT NULL
  AND runtime_image_ref IS NOT NULL
ORDER BY id
"""


def queue_demands(conn) -> list[QueueDemand]:
    with conn.cursor() as cur:
        cur.execute(QUEUE_DEMAND_SQL)
        rows = cur.fetchall()
    demands = []
    for row in rows:
        demands.append(
            QueueDemand(
                profile_id=str(row["profile_id"]) if row["profile_id"] else None,
                runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
                run_target=str(row["run_target"]) if row["run_target"] else None,
                pending_count=int(row["pending_count"]),
                running_count=int(row["running_count"]),
                oldest_job_id=int(row["oldest_job_id"]),
            )
        )
    return demands


def active_leases(conn) -> list[ActiveLease]:
    with conn.cursor() as cur:
        cur.execute(ACTIVE_LEASE_SQL)
        rows = cur.fetchall()
    return [
        ActiveLease(
            lease_owner=str(row["lease_owner"]),
            profile_id=str(row["profile_id"]) if row["profile_id"] else None,
            runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
            run_target=str(row["run_target"]) if row["run_target"] else None,
            running_count=int(row["running_count"]),
            worker_kind=WORKER_KIND_TRAIN,
        )
        for row in rows
    ]


def active_worker_leases(conn) -> tuple[ActiveLease, ...]:
    return tuple(active_leases(conn))


def running_jobs(conn) -> list[RunningJob]:
    with conn.cursor() as cur:
        cur.execute(RUNNING_JOBS_SQL)
        rows = cur.fetchall()
    return [
        RunningJob(
            id=int(row["id"]),
            lease_owner=str(row["lease_owner"]),
            profile_id=str(row["profile_id"]) if row["profile_id"] else None,
            runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
            run_target=str(row["run_target"]) if row["run_target"] else None,
            run_name=str(row["run_name"]) if row["run_name"] else None,
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
        )
        for row in rows
    ]


def parse_label_string(value: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        labels[key.strip()] = raw_value.strip()
    return labels


def parse_docker_ps_json_lines(host: str, output: str) -> list[ExistingContainer]:
    containers: list[ExistingContainer] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        labels = payload.get("Labels")
        if isinstance(labels, str):
            parsed_labels = parse_label_string(labels)
        elif isinstance(labels, dict):
            parsed_labels = {str(key): str(value) for key, value in labels.items()}
        else:
            parsed_labels = {}
        containers.append(
            ExistingContainer(
                host=host,
                name=str(payload.get("Names") or payload.get("Name") or "").strip("/"),
                state=str(payload.get("State") or ""),
                status=str(payload.get("Status") or ""),
                image=str(payload.get("Image") or ""),
                labels=parsed_labels,
            )
        )
    return containers


def host_command(host: HostConfig, remote_args: Sequence[str]) -> list[str]:
    return ["ssh", *host.ssh_options, host.ssh_target, shell_join(remote_args)]


def run_host_script(
    host: HostConfig,
    script: str,
    *,
    local: bool = False,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["bash", "-lc", script] if local else host_command(host, ["bash", "-lc", script])
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def list_managed_containers(host: HostConfig, *, local: bool = False) -> list[ExistingContainer]:
    result = run_host_script(
        host,
        shell_join(
            docker_command(
                host,
                ["ps", "-a", "--filter", "label=rlab.managed=true", "--format", "{{json .}}"],
            )
        ),
        local=local,
        capture=True,
    )
    if result.returncode != 0:
        output = (result.stdout or "").strip()
        raise RuntimeError(f"failed to list managed containers on {host.name}: {output}")
    return parse_docker_ps_json_lines(host.name, result.stdout or "")


def collect_existing_containers(
    config: FleetConfig,
    *,
    host_filter: str | None = None,
    local: bool = False,
) -> tuple[list[ExistingContainer], list[str]]:
    existing: list[ExistingContainer] = []
    warnings: list[str] = []
    for host in selected_hosts(config, host_filter):
        try:
            existing.extend(list_managed_containers(host, local=local))
        except Exception as exc:
            warnings.append(str(exc))
    return existing, warnings


def run_action(config: FleetConfig, action: FleetAction, *, local: bool = False) -> int:
    return run_action_result(config, action, local=local, capture=False).exit_code


def run_action_result(
    config: FleetConfig,
    action: FleetAction,
    *,
    local: bool = False,
    capture: bool = False,
) -> ActionResult:
    if not action.commands:
        return ActionResult(
            kind=action.kind,
            host=action.host,
            container=action.container,
            exit_code=0,
        )
    host = config.hosts[action.host]
    script = "set -euo pipefail\n" + "\n".join(action.commands)
    result = run_host_script(host, script, local=local, capture=capture)
    return ActionResult(
        kind=action.kind,
        host=action.host,
        container=action.container,
        exit_code=int(result.returncode),
        output=(result.stdout or "").strip(),
    )


def selected_hosts(config: FleetConfig, host_filter: str | None) -> list[HostConfig]:
    if host_filter:
        if host_filter not in config.hosts:
            known = ", ".join(sorted(config.hosts))
            raise ValueError(f"unknown fleet host {host_filter!r}; known hosts: {known}")
        return [config.hosts[host_filter]]
    return [config.hosts[name] for name in config.hosts]


def setup_host_script(host: HostConfig, *, runtime_image_ref: str | None = None) -> str:
    docker_info = shell_join(docker_command(host, ["info"]))
    gpu_test = shell_join(
        docker_command(host, ["run", "--rm", "--gpus", "all", host.gpu_test_image, "nvidia-smi"])
    )
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(host.base_dir)}",
        f"mkdir -p {shlex.quote(host.runs_dir)} {shlex.quote(host.logs_dir)} "
        f"{shlex.quote(host.rom_dir)} {shlex.quote(host.state_dir)}",
        "if ! command -v docker >/dev/null 2>&1; then",
        "  if command -v apt-get >/dev/null 2>&1; then",
        "    sudo -n apt-get update",
        "    sudo -n apt-get install -y docker.io",
        "  else",
        "    echo 'docker is missing and apt-get is unavailable' >&2",
        "    exit 1",
        "  fi",
        "fi",
        "sudo -n systemctl enable --now docker >/dev/null 2>&1 || true",
        f"{docker_info} >/dev/null",
        "if ! command -v nvidia-smi >/dev/null 2>&1; then",
        "  echo 'warning: nvidia-smi is not on PATH' >&2",
        "else",
        "  nvidia-smi >/dev/null",
        "fi",
        "if ! command -v nvidia-ctk >/dev/null 2>&1; then",
        "  if command -v apt-get >/dev/null 2>&1; then",
        "    sudo -n apt-get install -y --no-install-recommends ca-certificates curl gnupg2",
        "    sudo -n install -d -m 0755 /usr/share/keyrings",
        "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
        "sudo -n gpg --batch --yes --dearmor "
        "-o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
        "    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/"
        "nvidia-container-toolkit.list | sed 's#deb https://#deb "
        "[signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
        "sudo -n tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null",
        "    sudo -n apt-get update",
        "    sudo -n apt-get install -y nvidia-container-toolkit",
        "  else",
        "    echo 'nvidia-ctk is missing and apt-get is unavailable' >&2",
        "    exit 1",
        "  fi",
        "fi",
        "if command -v nvidia-ctk >/dev/null 2>&1; then",
        "  sudo -n nvidia-ctk runtime configure --runtime=docker",
        "  sudo -n systemctl restart docker || true",
        "fi",
        f"if ! {gpu_test} >/dev/null; then",
        f"  {gpu_test} >/dev/null",
        "fi",
        f"if [ ! -f {shlex.quote(host.env_file)} ]; then",
        f"  umask 077; cat > {shlex.quote(host.env_file)} <<'EOF'",
        "# rlab runner secrets live here; fill values on the host.",
        "TRAIN_QUEUE_DATABASE_URL=",
        "WANDB_API_KEY=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_S3_ENDPOINT_URL=",
        "AWS_REGION=",
        "CHECKPOINT_BUCKET_URI=",
        "EOF",
        "fi",
        f"test -f {shlex.quote(host.env_file)}",
    ]
    if runtime_image_ref:
        image = docker_image_ref(runtime_image_ref)
        lines.extend(
            [
                shell_join(docker_command(host, ["pull", image])),
                shell_join(
                    docker_command(
                        host,
                        [
                            "run",
                            "--rm",
                            "--gpus",
                            "all",
                            "--env-file",
                            host.env_file,
                            "-e",
                            f"RLAB_ROM_DIR={host.container_rom_dir}",
                            "-v",
                            f"{host.rom_dir}:{host.container_rom_dir}:ro",
                            image,
                            "rlab-container-entrypoint",
                            "rlab-container-smoke",
                        ],
                    )
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def image_ref_from_args(args: argparse.Namespace, *, default_latest: bool = False) -> str | None:
    image = str(getattr(args, "image", "") or "").strip()
    image_file = getattr(args, "image_file", None)
    if image_file:
        return runtime_image_ref_from_file(image_file)
    if image:
        if image == "latest":
            return latest_runtime_image_ref(
                workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
                branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
                artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
            )
        return normalize_runtime_image_ref(image)
    has_explicit_ref = bool(getattr(args, "runtime_image_ref", None))
    has_ref_file = bool(getattr(args, "runtime_image_ref_file", None))
    use_latest = bool(getattr(args, "latest_image", False)) or (
        default_latest and not has_explicit_ref and not has_ref_file
    )
    if use_latest:
        return latest_runtime_image_ref(
            workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
            branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
            artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
        )
    if getattr(args, "runtime_image_ref_file", None):
        return runtime_image_ref_from_file(args.runtime_image_ref_file)
    value = getattr(args, "runtime_image_ref", None)
    return normalize_runtime_image_ref(value) if value else None


def args_selects_latest_image(args: argparse.Namespace, *, default_latest: bool = False) -> bool:
    if getattr(args, "image_file", None) or getattr(args, "runtime_image_ref_file", None):
        return False
    image = str(getattr(args, "image", "") or "").strip()
    if image:
        return image == "latest"
    if getattr(args, "runtime_image_ref", None):
        return False
    return bool(getattr(args, "latest_image", False)) or default_latest


def recent_images_from_args(args: argparse.Namespace, *, limit: int = 3) -> tuple[RuntimeImageInfo, ...]:
    return recent_runtime_images(
        workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
        branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
        artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
        limit=limit,
    )


def runtime_image_context_from_args(
    args: argparse.Namespace,
    *,
    default_latest: bool = False,
) -> tuple[str | None, tuple[RuntimeImageInfo, ...], tuple[str, ...]]:
    recent_images: tuple[RuntimeImageInfo, ...] = ()
    warnings: list[str] = []
    selects_latest = args_selects_latest_image(args, default_latest=default_latest)
    try:
        recent_images = recent_images_from_args(args, limit=3)
    except Exception as exc:
        warnings.append(f"failed to list recent train images: {exc}")
    if selects_latest and recent_images:
        return recent_images[0].runtime_image_ref, recent_images, tuple(warnings)
    return image_ref_from_args(args, default_latest=default_latest), recent_images, tuple(warnings)


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(getattr(args, "direct", False)))


def _load_config_from_args(args: argparse.Namespace) -> FleetConfig:
    return load_fleet_config(
        repo_root_from_args(args),
        instances_path=getattr(args, "instances", DEFAULT_INSTANCES_CONFIG),
        machines_path=getattr(args, "machines", DEFAULT_MACHINE_REGISTRY),
    )


def repo_root_from_args(args: argparse.Namespace) -> Path:
    repo_root = getattr(args, "repo_root", None)
    if repo_root:
        return Path(repo_root).expanduser().resolve()
    return default_repo_root()


def watch_latest_lock_path(args: argparse.Namespace) -> Path:
    return repo_root_from_args(args) / "runs" / "fleet" / "watch.lock"


def acquire_watch_latest_lock(args: argparse.Namespace) -> WatchLatestLock:
    path = watch_latest_lock_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise WatchLatestLockBusy(path, owner) from exc
        raise
    handle.seek(0)
    handle.truncate()
    owner = {
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repo_root": str(repo_root_from_args(args)),
        "host": getattr(args, "host", None) or "all",
        "mode": "execute" if getattr(args, "execute", True) else "dry-run",
        "interval": getattr(args, "interval", DEFAULT_WATCH_LATEST_INTERVAL_SECONDS),
    }
    handle.write(json.dumps(owner, sort_keys=True) + "\n")
    handle.flush()
    return WatchLatestLock(path=path, handle=handle)


def release_watch_latest_lock(lock: WatchLatestLock) -> None:
    try:
        fcntl.flock(lock.handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock.handle.close()


def stale_lease_owner_prefix_for_host(host: HostConfig) -> str:
    return f"rlab-{sanitize_slug(host.name, limit=16)}-"


def stale_train_job_from_row(
    host: HostConfig,
    row: Mapping[str, Any],
    *,
    execute: bool,
) -> StaleTrainJob:
    return StaleTrainJob(
        host=host.name,
        id=int(row["id"]),
        profile_id=row.get("profile_id"),
        runtime_image_ref=row.get("runtime_image_ref"),
        run_target=row.get("run_target"),
        run_name=row.get("run_name"),
        lease_owner=row.get("stale_lease_owner"),
        heartbeat_at=row.get("stale_heartbeat_at"),
        execute=execute,
    )


def stale_train_job_error_for_host(host: HostConfig) -> str:
    return f"worker_lost: stale train job marked failed by rlab fleet watch host={host.name}"


def stale_train_jobs_for_watch(
    conn,
    config: FleetConfig,
    *,
    execute: bool,
    older_than_seconds: int,
    limit: int,
) -> tuple[StaleTrainJob, ...]:
    stale_jobs: list[StaleTrainJob] = []
    for host in sorted(config.hosts.values(), key=lambda item: item.name):
        common = {
            "run_target": host.run_target,
            "lease_owner_prefix": stale_lease_owner_prefix_for_host(host),
            "older_than_seconds": older_than_seconds,
            "limit": limit,
        }
        if execute:
            rows = mark_stale_train_jobs_failed(
                conn,
                **common,
                error=stale_train_job_error_for_host(host),
            )
        else:
            rows = list_stale_train_jobs(conn, **common)
        stale_jobs.extend(stale_train_job_from_row(host, row, execute=execute) for row in rows)
    return tuple(stale_jobs)


def running_job_device_key(job: RunningJob) -> str:
    return infer_device_key(
        "train",
        job.profile_id or "",
        job.lease_owner,
        {},
        run_target=job.run_target,
    )


def config_device_keys(config: FleetConfig) -> set[str]:
    keys: set[str] = set()
    for host in config.hosts.values():
        key = device_key_from_run_target(host.run_target) or host.run_target
        if key:
            keys.add(key)
    return keys


def active_watch_device_keys(config: FleetConfig, jobs: Sequence[RunningJob]) -> tuple[str, ...]:
    configured_keys = config_device_keys(config)
    keys: list[str] = []
    for job in jobs:
        key = running_job_device_key(job)
        if configured_keys and key not in configured_keys:
            continue
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def watch_monitor_jobs(
    jobs: Sequence[RunningJob],
    active_keys: Sequence[str],
) -> list[dict[str, Any]]:
    active_key_set = set(active_keys)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        key = running_job_device_key(job)
        if key not in active_key_set:
            continue
        rows.append(
            {
                "id": str(job.id),
                "state": "running",
                "device_key": key,
                "attention": "",
            }
        )
    return rows


def collect_active_watch_devices(
    repo_root: Path,
    config: FleetConfig,
    jobs: Sequence[RunningJob],
    *,
    probes: Mapping[str, DeviceProbe] | None = None,
) -> tuple[dict[str, Any], ...]:
    active_keys = active_watch_device_keys(config, jobs)
    if not active_keys:
        return ()
    live_probes = probes if probes is not None else live_device_probes(list(active_keys))
    rows = watch_monitor_jobs(jobs, active_keys)
    devices = devices_from_jobs(repo_root, rows, live_probes)
    active_key_set = set(active_keys)
    return tuple(
        device
        for device in devices
        if str(device.get("id")) in active_key_set and device.get("current_jobs")
    )


def build_live_plan(
    args: argparse.Namespace,
    *,
    local: bool = False,
) -> FleetPlan:
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    conn = _connect_from_args(args)
    try:
        demands = queue_demands(conn)
        leases = active_leases(conn)
    finally:
        conn.close()
    existing, container_warnings = collect_existing_containers(
        config,
        host_filter=None,
        local=local,
    )
    plan = build_fleet_plan(config, demands, existing, leases)
    return FleetPlan(
        desired=plan.desired,
        existing=plan.existing,
        actions=plan.actions,
        warnings=(*container_warnings, *plan.warnings),
    )


def format_demands(demands: Sequence[QueueDemand]) -> str:
    if not demands:
        return "queue demand: none"
    lines = ["queue demand:"]
    for demand in demands:
        lines.append(
            "  "
            f"profile={demand.profile_id or 'any'} target={demand.run_target or 'any'} "
            f"pending={demand.pending_count} running={demand.running_count} "
            f"digest={runtime_image_digest_slug(demand.runtime_image_ref)}"
        )
    return "\n".join(lines)


def format_capacity_policy(policy: Mapping[str, Any]) -> str:
    lines = [
        f"capacity_policy schema={policy.get('schema_version', 'unknown')} updated={policy.get('updated_at', 'unknown')}",
        f"purpose={policy.get('purpose', '')}",
    ]
    defaults = policy.get("defaults")
    if isinstance(defaults, Mapping):
        lines.append("defaults:")
        for key, value in sorted(defaults.items()):
            lines.append(f"  {key}={value}")
    lanes = policy.get("lanes")
    if isinstance(lanes, Sequence) and not isinstance(lanes, str):
        lines.append("lanes:")
        for lane in lanes:
            if not isinstance(lane, Mapping):
                continue
            lines.append(
                "  "
                f"{lane.get('name')} target={lane.get('target')} "
                f"manager={lane.get('manager')} max_runner_workers={lane.get('max_runner_workers')} "
                f"env_threads={lane.get('env_threads')}"
            )
    checks = policy.get("policy_checks")
    if isinstance(checks, Sequence) and not isinstance(checks, str):
        lines.append("policy_checks:")
        lines.extend(f"  {check}" for check in checks)
    return "\n".join(lines)


def format_plan(plan: FleetPlan) -> str:
    lines = [
        f"desired_deployments={len(plan.desired)}",
        f"existing_containers={len(plan.existing)}",
        f"actions={len([action for action in plan.actions if action.kind != 'keep'])}",
    ]
    if plan.desired:
        lines.append("desired:")
        for item in plan.desired:
            lines.append(
                "  "
                f"{item.name} host={item.key.host} kind={item.key.worker_kind} workers={item.workers} "
                f"profile={item.key.profile_id or 'any'} target={item.key.run_target or 'any'} "
                f"digest={runtime_image_digest_slug(item.key.runtime_image_ref)}"
            )
    if plan.actions:
        lines.append("actions:")
        for action in plan.actions:
            lines.append(
                "  "
                f"{action.kind} host={action.host} container={action.container} "
                f"reason={action.reason}"
            )
            for command in action.commands:
                lines.append(f"    $ {command}")
    if plan.warnings:
        lines.append("warnings:")
        lines.extend(f"  {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def format_elapsed_since(value: Any, *, now: datetime | None = None) -> str:
    if not value:
        return "unknown"
    timestamp: datetime
    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value).strip()
        if not text:
            return "unknown"
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    seconds = max(0, int((current - timestamp).total_seconds()))
    if seconds < 60:
        return f"{seconds}s_ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m_ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h_ago"
    return f"{hours // 24}d_ago"


def format_elapsed_duration_since(value: Any, *, now: datetime | None = None) -> str:
    elapsed = format_elapsed_since(value, now=now)
    return elapsed.removesuffix("_ago")


def format_utc_minute(value: Any) -> str:
    if not value:
        return "unknown"
    timestamp: datetime
    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value).strip()
        if not text:
            return "unknown"
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).strftime("%Y-%m-%d %H:%MZ")


def format_containers(
    containers: Sequence[ExistingContainer],
    jobs: Sequence[RunningJob] = (),
    warnings: Sequence[str] = (),
) -> str:
    lines = []
    matched_job_ids: set[int] = set()
    if containers:
        lines.append("managed containers:")
        for container in sorted(containers, key=lambda item: (item.host, item.name)):
            key = container.key
            profile = key.profile_id if key else None
            target = key.run_target if key else None
            worker_kind = key.worker_kind if key else "unknown"
            digest = container.labels.get(f"{LABEL_PREFIX}runtime-digest")
            worker_prefix = container.labels.get(f"{LABEL_PREFIX}worker-prefix")
            prefix = worker_prefix or container.name
            lines.append(
                "  "
                f"host={container.host} name={container.name} "
                f"state={container.state or 'unknown'} status={container.status or 'unknown'} "
                f"kind={worker_kind} "
                f"profile={profile or 'any'} target={target or 'any'} "
                f"digest={digest or 'unknown'} worker_prefix={prefix}"
            )
            owned_jobs = [job for job in jobs if job.lease_owner.startswith(prefix)]
            for job in owned_jobs:
                matched_job_ids.add(job.id)
                heartbeat = format_elapsed_since(job.heartbeat_at)
                started = job.started_at.isoformat() if hasattr(job.started_at, "isoformat") else job.started_at
                worker = job.lease_owner.removeprefix(f"{prefix}-")
                fields = [
                    f"job={job.id}",
                    f"run={job.run_name or 'unknown'}",
                    f"worker={worker}",
                    f"profile={job.profile_id or 'any'}",
                ]
                if job.run_target != target:
                    fields.append(f"target={job.run_target or 'any'}")
                fields.extend(
                    [
                        f"started={started or 'unknown'}",
                        f"heartbeat={heartbeat or 'unknown'}",
                    ]
                )
                lines.append(f"    {' '.join(fields)}")
    else:
        lines.append("managed containers: none")
    unmatched_jobs = [job for job in jobs if job.id not in matched_job_ids]
    if unmatched_jobs:
        lines.append("unmatched running jobs:")
        for job in unmatched_jobs:
            lines.append(
                "  "
                f"job={job.id} run={job.run_name or 'unknown'} owner={job.lease_owner} "
                f"profile={job.profile_id or 'any'} target={job.run_target or 'any'}"
            )
    if warnings:
        lines.append("warnings:")
        lines.extend(f"  {warning}" for warning in warnings)
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        demands = queue_demands(conn)
        leases = active_leases(conn)
    finally:
        conn.close()
    print(format_demands(demands))
    if leases:
        print("active leases:")
        for lease in leases:
            print(
                "  "
                f"owner={lease.lease_owner} running={lease.running_count} "
                f"profile={lease.profile_id or 'any'} target={lease.run_target or 'any'}"
            )
    else:
        print("active leases: none")
    return 0


@dataclass(frozen=True)
class JobContainer:
    machine: str
    name: str
    state: str
    status: str
    labels: dict[str, str]

    @property
    def launch_id(self) -> str | None:
        return self.labels.get(LAUNCH_ID_LABEL)

    @property
    def job_kind(self) -> str | None:
        return self.labels.get(JOB_KIND_LABEL)


def load_registry_from_args(args: argparse.Namespace) -> MachineRegistry:
    return load_machine_registry(args.machines)


def sanitize_container_part(value: str, *, limit: int = 32) -> str:
    return sanitize_slug(value, limit=limit)


def job_container_name(machine: MachineConfig, *, job_kind: str, launch_id: str) -> str:
    return (
        f"rlab-job-{sanitize_container_part(machine.name, limit=16)}-"
        f"{sanitize_container_part(job_kind, limit=8)}-"
        f"{sanitize_container_part(launch_id, limit=48)}"
    )[:120].strip("-")


def launch_payload_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.payloads_dir.rstrip('/')}/{launch_id}.json"


def launch_output_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.outputs_dir.rstrip('/')}/{launch_id}"


def container_payload_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.container_payloads_dir.rstrip('/')}/{launch_id}.json"


def container_output_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.container_outputs_dir.rstrip('/')}/{launch_id}"


def machine_docker_command(machine: MachineConfig, args: Sequence[str]) -> list[str]:
    return [*machine.docker_command, *args]


def job_container_run_command(
    machine: MachineConfig,
    *,
    job_kind: str,
    job_id: int,
    launch_id: str,
    runtime_image_ref: str,
    container_name: str,
) -> list[str]:
    image = docker_image_ref(runtime_image_ref)
    cmd = machine_docker_command(
        machine,
        [
            "run",
            "-d",
            "--name",
            container_name,
            "--restart",
            "no",
            "--gpus",
            "all",
            "--env-file",
            machine.paths.env_file,
            "-v",
            f"{machine.paths.payloads_dir}:{machine.paths.container_payloads_dir}:ro",
            "-v",
            f"{machine.paths.outputs_dir}:{machine.paths.container_outputs_dir}",
            "-v",
            f"{machine.paths.roms_dir}:{machine.paths.container_roms_dir}:ro",
            "-e",
            f"RLAB_ROM_DIR={machine.paths.container_roms_dir}",
        ],
    )
    labels = {
        MANAGED_LABEL: "true",
        JOB_CONTAINER_LABEL: "true",
        MACHINE_LABEL: machine.name,
        JOB_KIND_LABEL: job_kind,
        JOB_ID_LABEL: str(job_id),
        LAUNCH_ID_LABEL: launch_id,
        OUTPUT_URI_LABEL: launch_output_path(machine, launch_id),
        f"{LABEL_PREFIX}runtime-image-ref": runtime_image_ref,
    }
    for key, value in sorted(labels.items()):
        cmd.extend(["--label", f"{key}={value}"])
    cmd.extend(
        [
            image,
            "rlab-container-entrypoint",
            "rlab",
            "run-job",
            "--payload",
            container_payload_path(machine, launch_id),
            "--output-dir",
            container_output_path(machine, launch_id),
        ]
    )
    return cmd


def machine_ssh_prefix(machine: MachineConfig) -> list[str]:
    if machine.backend != "docker_ssh":
        return []
    return ["ssh", *machine.ssh_options, machine.ssh_target]


def run_machine_shell(
    machine: MachineConfig,
    script: str,
    *,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if machine.backend == "local_docker":
        return subprocess.run(
            ["sh", "-lc", script],
            input=input_text,
            capture_output=capture,
            text=True,
            check=False,
        )
    return subprocess.run(
        [*machine_ssh_prefix(machine), "sh", "-lc", shlex.quote(script)],
        input=input_text,
        capture_output=capture,
        text=True,
        check=False,
    )


def run_machine_docker(
    machine: MachineConfig,
    docker_args: Sequence[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = shell_join(machine_docker_command(machine, docker_args))
    return run_machine_shell(machine, command, capture=capture)


def write_remote_payload(machine: MachineConfig, path: str, payload: Mapping[str, Any]) -> None:
    payload_text = json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n"
    script = f"mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path)}"
    result = run_machine_shell(machine, script, input_text=payload_text, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to write payload {path}: {result.stderr or result.stdout}")


def parse_docker_labels(value: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in str(value or "").split(","):
        if "=" not in part:
            continue
        key, label_value = part.split("=", 1)
        labels[key.strip()] = label_value.strip()
    return labels


def parse_job_containers(machine: MachineConfig, output: str) -> list[JobContainer]:
    containers: list[JobContainer] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        labels = parse_docker_labels(str(row.get("Labels") or ""))
        if labels.get(JOB_CONTAINER_LABEL) != "true":
            continue
        containers.append(
            JobContainer(
                machine=machine.name,
                name=str(row.get("Names") or row.get("Name") or ""),
                state=str(row.get("State") or "").lower(),
                status=str(row.get("Status") or ""),
                labels=labels,
            )
        )
    return containers


def list_job_containers(machine: MachineConfig) -> list[JobContainer]:
    result = run_machine_docker(
        machine,
        [
            "ps",
            "-a",
            "--filter",
            f"label={JOB_CONTAINER_LABEL}=true",
            "--format",
            "{{json .}}",
        ],
        capture=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker ps failed on {machine.name}: {result.stderr or result.stdout}")
    return parse_job_containers(machine, result.stdout)


def read_remote_result(machine: MachineConfig, output_uri: str) -> dict[str, Any] | None:
    result_path = f"{str(output_uri).rstrip('/')}/result.json"
    result = run_machine_shell(machine, f"cat {shlex.quote(result_path)}", capture=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"{result_path} did not contain a JSON object")
    return payload


def remote_result_exists(machine: MachineConfig, output_uri: str) -> bool:
    result_path = f"{str(output_uri).rstrip('/')}/result.json"
    result = run_machine_shell(machine, f"test -s {shlex.quote(result_path)}", capture=True)
    return result.returncode == 0


def machine_queue_counts(conn) -> dict[str, dict[str, int]]:
    counts = {"train": {}}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            GROUP BY status
            ORDER BY status
            """
        )
        counts["train"] = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
    return counts


def build_machine_watch_snapshot(args: argparse.Namespace) -> MachineWatchSnapshot:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    warnings: list[str] = []
    containers = tuple(sorted(list_job_containers(machine), key=lambda item: item.name))
    conn = _connect_from_args(args)
    try:
        launches = tuple(
            active_job_launches(
                conn,
                machine=machine.name,
                states=("launching", "running"),
            )
        )
        queue_counts = machine_queue_counts(conn)
    finally:
        conn.close()

    containers_by_launch = {container.launch_id: container for container in containers if container.launch_id}
    result_present: dict[str, bool] = {}
    for launch in launches:
        launch_id = str(launch["launch_id"])
        container = containers_by_launch.get(launch_id)
        if container is None or container.state != "running":
            try:
                result_present[launch_id] = remote_result_exists(machine, str(launch["output_uri"]))
            except Exception as exc:
                result_present[launch_id] = False
                warnings.append(f"result check failed launch_id={launch_id}: {exc}")
    for container in containers:
        launch_id = container.launch_id
        output_uri = container.labels.get(OUTPUT_URI_LABEL)
        if not launch_id or not output_uri or container.state == "running":
            continue
        if launch_id in result_present:
            continue
        try:
            result_present[launch_id] = remote_result_exists(machine, output_uri)
        except Exception as exc:
            result_present[launch_id] = False
            warnings.append(f"result check failed launch_id={launch_id}: {exc}")

    return MachineWatchSnapshot(
        captured_at=datetime.now(UTC),
        machine=machine,
        containers=containers,
        launches=launches,
        queue_counts=queue_counts,
        result_present=result_present,
        warnings=tuple(warnings),
    )


def _job_container_active(container: JobContainer) -> bool:
    return container.state in {"running", "created", "restarting"}


def _watch_hint(
    *,
    launch: Mapping[str, Any] | None,
    container: JobContainer | None,
    result_present: bool,
) -> str:
    if launch is None:
        return "orphaned_container"
    if container is None:
        if result_present:
            return "needs_shepherd_finalize"
        if launch.get("state") == "launching":
            return "needs_shepherd_release"
        return "needs_shepherd_fail_or_retry"
    if container.state == "running":
        return "ok"
    if result_present:
        return "needs_shepherd_finalize"
    return "needs_shepherd_mark_failed"


def render_machine_watch_dashboard(snapshot: MachineWatchSnapshot, *, color: bool = False) -> str:
    del color
    machine = snapshot.machine
    active_containers = [container for container in snapshot.containers if _job_container_active(container)]
    active_by_kind: dict[str, int] = {"train": 0}
    for container in active_containers:
        if container.job_kind in active_by_kind:
            active_by_kind[str(container.job_kind)] += 1
    launches_by_id = {str(launch["launch_id"]): launch for launch in snapshot.launches}
    containers_by_launch = {container.launch_id: container for container in snapshot.containers if container.launch_id}
    train_counts = snapshot.queue_counts.get("train", {})
    lines = [
        f"rlab fleet watch machine={machine.name} mode=read-only captured={snapshot.captured_at.isoformat()}",
        (
            "capacity "
            f"total={len(active_containers)}/{machine.limits.max_parallel_containers} "
            f"train={active_by_kind['train']}/{machine.max_containers_for_kind('train')}"
        ),
        (
            "queue "
            f"train_pending={int(train_counts.get('pending', 0))} "
            f"train_launching={int(train_counts.get('launching', 0))} "
            f"train_running={int(train_counts.get('running', 0))}"
        ),
        "launches:",
    ]
    if not snapshot.launches:
        lines.append("  none")
    for launch in snapshot.launches:
        launch_id = str(launch["launch_id"])
        container = containers_by_launch.get(launch_id)
        result_present = bool(snapshot.result_present.get(launch_id, False))
        hint = _watch_hint(launch=launch, container=container, result_present=result_present)
        lines.append(
            "  "
            f"launch_id={launch_id} "
            f"job={launch['job_kind']}/{launch['job_id']} "
            f"launch_state={launch['state']} "
            f"container={container.name if container else 'missing'} "
            f"container_state={container.state if container else 'missing'} "
            f"result={'yes' if result_present else 'no'} "
            f"hint={hint}"
        )
    orphaned = [
        container
        for container in snapshot.containers
        if container.launch_id and container.launch_id not in launches_by_id
    ]
    if orphaned:
        lines.append("orphaned_containers:")
        for container in orphaned:
            result_present = bool(snapshot.result_present.get(str(container.launch_id), False))
            lines.append(
                "  "
                f"container={container.name} "
                f"launch_id={container.launch_id} "
                f"job={container.job_kind or 'unknown'}/{container.labels.get(JOB_ID_LABEL, 'unknown')} "
                f"state={container.state} "
                f"result={'yes' if result_present else 'no'} "
                "hint=orphaned_container"
            )
    if snapshot.warnings:
        lines.append("warnings:")
        lines.extend(f"  {warning}" for warning in snapshot.warnings)
    return "\n".join(lines)


def docker_pull_for_job(machine: MachineConfig, runtime_image_ref: str) -> int:
    if machine.pull_policy == "never":
        return 0
    result = run_machine_docker(machine, ["pull", docker_image_ref(runtime_image_ref)], capture=False)
    return int(result.returncode)


def launch_claimed_job_container(
    conn,
    *,
    machine: MachineConfig,
    job_kind: str,
    job_id: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    launch_id = new_launch_id(job_kind, job_id)
    claimed = claim_job_launch(
        conn,
        job_kind=job_kind,
        machine=machine.name,
        backend=machine.backend,
        job_id=job_id,
        launch_id=launch_id,
        output_uri=launch_output_path(machine, launch_id),
    )
    if claimed is None:
        return None
    job, launch = claimed
    runtime_image_ref = str(job["runtime_image_ref"])
    container_name = job_container_name(machine, job_kind=job_kind, launch_id=launch_id)
    payload = job_payload_for_launch(job, {**launch, "container_name": container_name})
    try:
        write_remote_payload(machine, launch_payload_path(machine, launch_id), payload)
        pull_status = docker_pull_for_job(machine, runtime_image_ref)
        if pull_status != 0:
            release_job_launch(conn, launch_id=launch_id, error=f"docker pull failed {pull_status}")
            raise RuntimeError(f"docker pull failed exit={pull_status}")
        run_command = job_container_run_command(
            machine,
            job_kind=job_kind,
            job_id=int(job["id"]),
            launch_id=launch_id,
            runtime_image_ref=runtime_image_ref,
            container_name=container_name,
        )
        result = run_machine_shell(machine, shell_join(run_command), capture=True)
        if result.returncode != 0:
            release_job_launch(
                conn,
                launch_id=launch_id,
                error=f"docker run failed {result.returncode}: {result.stderr or result.stdout}",
            )
            raise RuntimeError(f"docker run failed exit={result.returncode}: {result.stderr or result.stdout}")
        provider_run_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else container_name
        mark_job_launch_running(
            conn,
            launch_id=launch_id,
            container_name=container_name,
            provider_run_id=provider_run_id,
        )
        print(
            f"machine={machine.name} action=launch result=started "
            f"job_kind={job_kind} job_id={job['id']} "
            f"launch_id={launch_id} container={container_name}"
        )
        return job, launch
    except Exception:
        raise


def reconcile_machine_launches(conn, machine: MachineConfig) -> int:
    launches = active_job_launches(conn, machine=machine.name)
    containers = {container.launch_id: container for container in list_job_containers(machine)}
    reconciled = 0
    for launch in launches:
        launch_id = str(launch["launch_id"])
        container = containers.get(launch_id)
        if container is None:
            result = read_remote_result(machine, str(launch["output_uri"]))
            if result is not None:
                finish_job_launch_from_result(conn, launch_id=launch_id, result=result)
                print(f"finalized launch_id={launch_id} from result.json")
            elif launch["state"] == "launching":
                release_job_launch(conn, launch_id=launch_id, error="launching container missing")
                print(f"released launch_id={launch_id}: launching container missing")
            else:
                synthetic = {
                    "job_kind": launch["job_kind"],
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                    "status": "failed",
                    "exit_code": None,
                    "error": "running container missing and result.json not found",
                }
                finish_job_launch_from_result(conn, launch_id=launch_id, result=synthetic)
                print(f"failed launch_id={launch_id}: running container missing")
            reconciled += 1
            continue
        if container.state == "running":
            mark_job_launch_running(
                conn,
                launch_id=launch_id,
                container_name=container.name,
                provider_run_id=container.name,
            )
            reconciled += 1
            continue
        result = read_remote_result(machine, str(launch["output_uri"]))
        if result is not None:
            finish_job_launch_from_result(conn, launch_id=launch_id, result=result)
            print(f"finalized launch_id={launch_id} exit_state={container.state}")
        else:
            synthetic = {
                "job_kind": launch["job_kind"],
                "job_id": launch["job_id"],
                "launch_id": launch_id,
                "status": "failed",
                "exit_code": None,
                "error": f"container exited without result.json: {container.status}",
            }
            finish_job_launch_from_result(conn, launch_id=launch_id, result=synthetic)
            print(f"failed launch_id={launch_id}: missing result.json")
        reconciled += 1
    return reconciled


def job_container_slot_usage(machine: MachineConfig, *, job_kind: str) -> tuple[int, int, int]:
    containers = list_job_containers(machine)
    active = [
        container
        for container in containers
        if container.job_kind == job_kind and container.state in {"running", "created", "restarting"}
    ]
    capacity = machine.max_containers_for_kind(job_kind)
    return len(active), capacity, max(0, capacity - len(active))


def machine_available_slots(machine: MachineConfig, *, job_kind: str) -> int:
    _, _, available = job_container_slot_usage(machine, job_kind=job_kind)
    return available


def launch_next_jobs(
    conn,
    *,
    machine: MachineConfig,
    job_kind: str,
    limit: int,
    reconcile: bool = True,
) -> int:
    if reconcile:
        reconcile_machine_launches(conn, machine)
    launched = 0
    active, capacity, slots = job_container_slot_usage(machine, job_kind=job_kind)
    if slots <= 0:
        print(
            f"machine={machine.name} action=skip reason=no_available_slots "
            f"job_kind={job_kind} used={active} max={capacity}"
        )
        return 0
    for _ in range(min(int(limit), slots)):
        claimed = launch_claimed_job_container(
            conn,
            machine=machine,
            job_kind=job_kind,
        )
        if claimed is None:
            break
        launched += 1
    return launched


def cmd_container_launch(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    if not args.execute:
        print(
            f"dry_run: would claim and launch {args.job_kind}_job_id={args.job_id} "
            f"machine={machine.name}"
        )
        return 0
    conn = _connect_from_args(args)
    try:
        launched = launch_claimed_job_container(
            conn,
            machine=machine,
            job_kind=args.job_kind,
            job_id=args.job_id,
        )
    finally:
        conn.close()
    if launched is None:
        print("launch_claimed=0")
    return 0


def cmd_container_launch_next(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    if not args.execute:
        slots = machine_available_slots(machine, job_kind=args.job_kind)
        planned = min(int(args.limit), slots)
        print(f"dry_run: would claim up to {planned} {args.job_kind} job(s) machine={machine.name}")
        return 0
    conn = _connect_from_args(args)
    try:
        launched = launch_next_jobs(
            conn,
            machine=machine,
            job_kind=args.job_kind,
            limit=int(args.limit),
            reconcile=True,
        )
    finally:
        conn.close()
    print(f"launched={launched} machine={machine.name} job_kind={args.job_kind}")
    return 0


def cmd_container_reconcile(args: argparse.Namespace) -> int:
    if not args.execute:
        target = args.machine or "all"
        print(f"dry_run: would reconcile job-container launches for machine={target}")
        return 0
    registry = load_registry_from_args(args)
    machines = [resolve_machine(registry, args.machine)] if args.machine else list(registry.machines.values())
    conn = _connect_from_args(args)
    try:
        total = 0
        for machine in machines:
            total += reconcile_machine_launches(conn, machine)
    finally:
        conn.close()
    print(f"reconciled={total}")
    return 0


def shepherd_lock_key(machine_name: str) -> str:
    return f"rlab-fleet-shepherd:{machine_name}"


def acquire_shepherd_lock(conn, machine_name: str) -> ShepherdLock:
    key = shepherd_lock_key(machine_name)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": key},
        )
        row = cur.fetchone()
    if not row or not row.get("acquired"):
        raise ShepherdLockBusy(machine_name)
    return ShepherdLock(machine=machine_name, key=key)


def release_shepherd_lock(conn, lock: ShepherdLock) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0)) AS released",
            {"key": lock.key},
        )


def cmd_container_shepherd(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    if not args.execute:
        while True:
            print(f"machine={machine.name} action=reconcile mode=dry-run")
            status = cmd_container_reconcile(args)
            if status != 0:
                return status
            print(f"machine={machine.name} action=launch-next mode=dry-run limit={args.limit}")
            status = cmd_container_launch_next(args)
            if status != 0:
                return status
            if args.once:
                return 0
            time.sleep(args.interval)

    conn = _connect_from_args(args)
    lock: ShepherdLock | None = None
    try:
        try:
            lock = acquire_shepherd_lock(conn, machine.name)
        except ShepherdLockBusy as exc:
            print(f"machine={exc.machine} action=lock result=busy")
            return 2
        while True:
            try:
                print(f"machine={machine.name} action=reconcile result=start")
                reconciled = reconcile_machine_launches(conn, machine)
                print(f"machine={machine.name} action=reconcile result=ok reconciled={reconciled}")
                launched = launch_next_jobs(
                    conn,
                    machine=machine,
                    job_kind=args.job_kind,
                    limit=int(args.limit),
                    reconcile=False,
                )
                print(
                    f"machine={machine.name} action=launch-next result=ok "
                    f"job_kind={args.job_kind} launched={launched}"
                )
                if args.once:
                    return 0
            except KeyboardInterrupt:
                print(f"machine={machine.name} action=stop reason=keyboard_interrupt")
                return 130
            except Exception as exc:
                print(f"machine={machine.name} action=error error={shlex.quote(str(exc))}")
                if args.once or getattr(args, "fail_fast", False):
                    return 1
            time.sleep(args.interval)
    finally:
        if lock is not None:
            release_shepherd_lock(conn, lock)
        conn.close()


def cmd_container_watch_dashboard(args: argparse.Namespace) -> int:
    while True:
        try:
            snapshot = build_machine_watch_snapshot(args)
            write_tui_frame(
                render_machine_watch_dashboard(snapshot, color=not args.no_color),
                enabled=not args.no_tui,
            )
            if args.once:
                return 0
        except KeyboardInterrupt:
            print("\nwatch stopped")
            return 130
        except Exception as exc:
            message = (
                f"rlab fleet watch machine={args.machine} mode=read-only\n"
                f"snapshot failed: {exc}\n\nCtrl-C to stop."
            )
            write_tui_frame(message, enabled=not args.no_tui)
            if args.once or getattr(args, "fail_fast", False):
                return 1
        time.sleep(args.interval)


def cmd_ps(args: argparse.Namespace) -> int:
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    existing, warnings = collect_existing_containers(config, host_filter=None, local=False)
    jobs: list[RunningJob] = []
    conn = None
    try:
        conn = _connect_from_args(args)
        jobs = running_jobs(conn)
    except Exception as exc:
        warnings.append(f"failed to list running jobs: {exc}")
    finally:
        if conn is not None:
            conn.close()
    print(format_containers(existing, jobs, warnings))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    print(format_plan(build_live_plan(args)))
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    config = _load_config_from_args(args)
    policy = load_capacity_policy(repo_root, args.policy)
    validate_capacity_policy(policy, config)
    print(format_capacity_policy(policy))
    return 0


def _run_reconcile_once(args: argparse.Namespace, *, local: bool = False) -> int:
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    plan = build_live_plan(args, local=local)
    print(format_plan(plan))
    if not args.execute:
        print("dry_run: rerun without --dry-run to apply the plan")
        return 0
    status = 0
    for action in plan.actions:
        if action.kind == "keep":
            continue
        result = run_action(config, action, local=local)
        if result != 0:
            status = result
            print(
                f"action_failed host={action.host} container={action.container} "
                f"kind={action.kind} exit={result}",
                file=sys.stderr,
            )
            break
    return status


def cmd_reconcile(args: argparse.Namespace) -> int:
    if getattr(args, "machine", None):
        return cmd_container_reconcile(args)
    while True:
        status = _run_reconcile_once(args, local=False)
        if status != 0 or not args.watch:
            return status
        time.sleep(args.interval)


def cmd_ensure_runner(args: argparse.Namespace) -> int:
    config = filter_config_to_host(_load_config_from_args(args), args.host)
    host = config.hosts[args.host]
    runtime_image_ref = image_ref_from_args(args, default_latest=True)
    if not runtime_image_ref:
        raise SystemExit("--image, --image-file, --runtime-image-ref, or --runtime-image-ref-file is required")
    profile_id = str(args.profile).strip() if args.profile else None
    print(f"runtime_image_ref={runtime_image_ref}")
    print(f"profile={profile_id or 'any'}")
    conn = _connect_from_args(args)
    try:
        leases = active_leases(conn)
    finally:
        conn.close()
    existing, container_warnings = collect_existing_containers(config, host_filter=None, local=False)
    plan = build_ensure_runner_plan(
        config,
        host_name=host.name,
        profile_id=profile_id,
        runtime_image_ref=runtime_image_ref,
        run_target=args.target,
        workers=args.workers,
        existing=existing,
        leases=leases,
    )
    plan = FleetPlan(
        desired=plan.desired,
        existing=plan.existing,
        actions=plan.actions,
        warnings=(*container_warnings, *plan.warnings),
    )
    print(format_plan(plan))
    if not args.execute:
        print("dry_run: rerun without --dry-run to apply the plan")
        return 0
    status = 0
    for action in plan.actions:
        if action.kind == "keep":
            continue
        result = run_action(config, action, local=False)
        if result != 0:
            status = result
            print(
                f"action_failed host={action.host} container={action.container} "
                f"kind={action.kind} exit={result}",
                file=sys.stderr,
            )
            break
    return status


def run_plan_actions(config: FleetConfig, plan: FleetPlan, *, local: bool = False) -> int:
    status = 0
    for action in plan.actions:
        if action.kind == "keep":
            continue
        result = run_action(config, action, local=local)
        if result != 0:
            status = result
            print(
                f"action_failed host={action.host} container={action.container} "
                f"kind={action.kind} exit={result}",
                file=sys.stderr,
            )
            break
    return status


def build_live_ensure_latest_plan(args: argparse.Namespace) -> tuple[FleetConfig, str, FleetPlan]:
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    runtime_image_ref = image_ref_from_args(args, default_latest=True)
    if not runtime_image_ref:
        raise SystemExit("--image, --image-file, --runtime-image-ref, or --runtime-image-ref-file is required")
    conn = _connect_from_args(args)
    try:
        demands = queue_demands(conn)
        leases = active_worker_leases(conn)
    finally:
        conn.close()
    existing, container_warnings = collect_existing_containers(config, host_filter=None, local=False)
    plan = build_ensure_latest_plan(
        config,
        runtime_image_ref=runtime_image_ref,
        workers=args.workers,
        existing=existing,
        leases=leases,
        demands=demands,
    )
    return (
        config,
        runtime_image_ref,
        FleetPlan(
            desired=plan.desired,
            existing=plan.existing,
            actions=plan.actions,
            warnings=(*container_warnings, *plan.warnings),
        ),
    )


def build_latest_watch_snapshot(
    args: argparse.Namespace,
    *,
    action_results: Sequence[ActionResult] = (),
    image_context: RuntimeImageContext | None = None,
    image_resolver: RuntimeImageResolver | None = None,
) -> LatestWatchSnapshot:
    repo_root = repo_root_from_args(args)
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    image_context = image_context or (
        image_resolver or RuntimeImageResolver(args, default_latest=True)
    ).resolve()
    runtime_image_ref = image_context.runtime_image_ref
    if not runtime_image_ref:
        raise SystemExit("--image, --image-file, --runtime-image-ref, or --runtime-image-ref-file is required")
    conn = _connect_from_args(args)
    try:
        stale_train_jobs = (
            stale_train_jobs_for_watch(
                conn,
                config,
                execute=bool(args.execute),
                older_than_seconds=int(
                    getattr(args, "stale_older_than_seconds", DEFAULT_WATCH_STALE_OLDER_THAN_SECONDS)
                ),
                limit=int(getattr(args, "stale_limit", DEFAULT_WATCH_STALE_LIMIT)),
            )
            if getattr(args, "claim_stale_jobs", True)
            else ()
        )
        demands = tuple(queue_demands(conn))
        leases = tuple(active_worker_leases(conn))
        jobs = tuple(running_jobs(conn))
    finally:
        conn.close()
    devices = collect_active_watch_devices(repo_root, config, jobs)
    existing, container_warnings = collect_existing_containers(config, host_filter=None, local=False)
    down_hosts = tuple(sorted(warning_hosts(config, container_warnings)))
    train_plan = build_ensure_latest_plan(
        config,
        runtime_image_ref=runtime_image_ref,
        workers=args.workers,
        existing=existing,
        leases=leases,
        demands=demands,
    )
    plan = train_plan
    return LatestWatchSnapshot(
        captured_at=datetime.now(UTC),
        config=config,
        runtime_image_ref=runtime_image_ref,
        demands=demands,
        leases=leases,
        jobs=jobs,
        plan=FleetPlan(
            desired=plan.desired,
            existing=plan.existing,
            actions=tuple(action for action in plan.actions if action.host not in down_hosts),
            warnings=(*image_context.warnings, *plan.warnings),
        ),
        recent_images=image_context.recent_images,
        devices=devices,
        stale_train_jobs=stale_train_jobs,
        down_hosts=down_hosts,
        action_results=tuple(action_results),
        execute=bool(args.execute),
        interval=float(args.interval),
    )


def run_latest_watch_actions(config: FleetConfig, plan: FleetPlan) -> tuple[ActionResult, ...]:
    results: list[ActionResult] = []
    failed_hosts: set[str] = set()
    for action in plan.actions:
        if action.kind == "keep" or action.host in failed_hosts:
            continue
        result = run_action_result(config, action, local=False, capture=True)
        results.append(result)
        if result.exit_code != 0:
            failed_hosts.add(action.host)
    return tuple(results)


ANSI_STYLES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "red": "\033[31m",
    "bright_red": "\033[1;31m",
    "green": "\033[32m",
    "bright_green": "\033[1;32m",
    "yellow": "\033[33m",
    "bright_yellow": "\033[1;33m",
    "blue": "\033[34m",
    "bright_blue": "\033[1;34m",
    "magenta": "\033[35m",
    "bright_magenta": "\033[1;35m",
    "cyan": "\033[36m",
    "bright_cyan": "\033[1;36m",
    "white": "\033[37m",
    "orange": "\033[38;5;208m",
}


def colorize(text: str, style: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI_STYLES[style]}{text}{ANSI_STYLES['reset']}"


def dashboard_divider(width: int, *, color: bool) -> str:
    return colorize("-" * min(width, 120), "gray", enabled=color)


def dashboard_chip(label: str, value: str, style: str, *, color: bool) -> str:
    return f"{label}={colorize(value, style, enabled=color)}"


def section_label(text: str, style: str, *, color: bool) -> str:
    return colorize(text, style, enabled=color)


def numbered_section(number: int, text: str, style: str, *, color: bool) -> str:
    prefix = colorize(f"{number}", "bright_red", enabled=color)
    return f"{prefix}{section_label(text, style, color=color)}"


def heat_style(ratio: float) -> str:
    if ratio >= 0.9:
        return "bright_red"
    if ratio >= 0.75:
        return "orange"
    if ratio >= 0.55:
        return "bright_yellow"
    return "bright_green"


def percent_ratio(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", value)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1)) / 100.0))


def used_total_ratio(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    total = float(match.group(2))
    if total <= 0:
        return None
    return max(0.0, min(1.0, float(match.group(1)) / total))


def usage_meter(value: str, *, ratio: float | None = None, color: bool, width: int = 10) -> str:
    if not value or value == "unknown":
        return colorize("unknown", "dim", enabled=color)
    if ratio is None:
        ratio = percent_ratio(value) or used_total_ratio(value)
    if ratio is None:
        return highlight_dashboard_text(value, color=color)
    filled = max(0, min(width, round(ratio * width)))
    empty = max(0, width - filled)
    style = heat_style(ratio)
    if color:
        bar = f"[{colorize('#' * filled, style, enabled=True)}{colorize('-' * empty, 'dim', enabled=True)}]"
        return f"{bar} {colorize(value, style, enabled=True)}"
    return f"[{'#' * filled}{'-' * empty}] {value}"


def highlight_dashboard_text(text: str, *, color: bool) -> str:
    if not color:
        return text
    styles = [
        (r"\bwould_fail\b", "bright_yellow"),
        (r"\bfailed\b", "bright_red"),
        (r"\bexit=\d+\b", "bright_red"),
        (r"\bdown\b", "bright_red"),
        (r"\bmissing\b", "yellow"),
        (r"\bbusy\b", "bright_green"),
        (r"\bwarning\b", "bright_yellow"),
        (r"\boffline\b", "bright_red"),
        (r"\bunreachable\b", "bright_red"),
        (r"\breachable\b", "bright_green"),
        (r"\blive\b", "bright_green"),
        (r"\bok\b", "bright_green"),
        (r"\bsteady\b", "bright_green"),
        (r"\bstart\b", "bright_cyan"),
        (r"\brestart\b", "bright_yellow"),
        (r"\bremove\b", "bright_yellow"),
        (r"\bplanned\b", "bright_cyan"),
        (r"\bnone\b", "dim"),
        (r"\bunknown\b", "dim"),
    ]
    highlighted = text
    for pattern, style in styles:
        highlighted = re.sub(
            pattern,
            lambda match, style=style: colorize(match.group(0), style, enabled=True),
            highlighted,
        )
    highlighted = re.sub(
        r"\[[#-]{1,10}\]\s+(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?/\d+(?:\.\d+)?\s+[A-Za-z]+)",
        lambda match: colorize(
            match.group(0),
            heat_style(percent_ratio(match.group(0)) or used_total_ratio(match.group(0)) or 0.0),
            enabled=True,
        ),
        highlighted,
    )
    return re.sub(
        r"\b[0-9a-f]{12}\b",
        lambda match: colorize(match.group(0), "cyan", enabled=True),
        highlighted,
    )


def style_table(table: str, *, color: bool) -> str:
    if not color or not table:
        return table
    lines = table.splitlines()
    if lines:
        lines[0] = colorize(lines[0], "white", enabled=True)
    if len(lines) > 1:
        lines[1] = colorize(lines[1], "dim", enabled=True)
    for index in range(2, len(lines)):
        lines[index] = highlight_dashboard_text(lines[index], color=True)
    return "\n".join(lines)


def compact_ref(runtime_image_ref: str) -> str:
    return runtime_image_digest_slug(runtime_image_ref)


def truncate_cell(value: Any, width: int) -> str:
    text = str(value)
    if width < 4 or len(text) <= width:
        return text
    return f"{text[: width - 3]}..."


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_width: int,
) -> str:
    if not headers:
        return ""
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(
            len(str(headers[index])),
            *(len(row[index]) for row in string_rows),
        )
        for index in range(len(headers))
    ]
    min_widths = [min(len(str(header)), 10) for header in headers]
    while sum(widths) + (3 * (len(widths) - 1)) > max_width and max(widths) > 10:
        widest = max(range(len(widths)), key=lambda index: widths[index])
        if widths[widest] <= min_widths[widest]:
            break
        widths[widest] -= 1
    line_parts = [str(header).ljust(widths[index]) for index, header in enumerate(headers)]
    lines = [" | ".join(line_parts)]
    lines.append("-+-".join("-" * width for width in widths))
    for row in string_rows:
        lines.append(
            " | ".join(
                truncate_cell(row[index], widths[index]).ljust(widths[index])
                for index in range(len(headers))
            )
        )
    return "\n".join(lines)


def warning_hosts(config: FleetConfig, warnings: Sequence[str]) -> set[str]:
    hosts = set()
    for host_name in config.hosts:
        if any(host_name in warning for warning in warnings):
            hosts.add(host_name)
    return hosts


def jobs_for_prefix(jobs: Sequence[RunningJob], prefix: str) -> list[RunningJob]:
    return [job for job in jobs if job.lease_owner.startswith(prefix)]


def short_hash(value: str | None, *, length: int = 12) -> str:
    text = str(value or "").strip()
    return text[:length] if text else "unknown"


def recent_image_dashboard_rows(images: Sequence[RuntimeImageInfo], *, limit: int = 3) -> list[list[str]]:
    return [
        [
            compact_ref(image.runtime_image_ref),
            short_hash(image.source_sha),
            format_utc_minute(image.published_at),
            image.commit_message or "unknown",
        ]
        for image in list(images)[:limit]
    ]


def device_detail(device: Mapping[str, Any], key: str) -> str:
    details = device.get("details")
    if not isinstance(details, Mapping):
        return "unknown"
    value = details.get(key)
    if value is None or value == "":
        return "unknown"
    return str(value)


def active_device_dashboard_rows(devices: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for device in devices:
        current_jobs = device.get("current_jobs")
        if isinstance(current_jobs, Sequence) and not isinstance(current_jobs, str):
            job_count = len(current_jobs)
        else:
            job_count = 1 if device.get("current_job") else 0
        cpu = device_detail(device, "cpu")
        memory = device_detail(device, "memory")
        gpu = device_detail(device, "gpu")
        vram = device_detail(device, "vram")
        rows.append(
            [
                str(device.get("device") or device.get("id") or "unknown"),
                str(device.get("target") or "unknown").removeprefix("docker/"),
                str(device.get("state") or "unknown"),
                str(job_count),
                usage_meter(cpu, color=False, width=4),
                usage_meter(memory, color=False, width=4),
                usage_meter(gpu, color=False, width=4),
                usage_meter(vram, color=False, width=4),
                str(device.get("last_check") or "unknown"),
            ]
        )
    return rows


def host_dashboard_rows(snapshot: LatestWatchSnapshot) -> list[list[str]]:
    down_hosts = set(snapshot.down_hosts) | warning_hosts(snapshot.config, snapshot.plan.warnings)
    desired_by_host = {
        item.key.host: item
        for item in snapshot.plan.desired
        if item.key.worker_kind == WORKER_KIND_TRAIN
    }
    actions_by_host: dict[str, list[FleetAction]] = {}
    for action in snapshot.plan.actions:
        actions_by_host.setdefault(action.host, []).append(action)
    results_by_host: dict[str, list[ActionResult]] = {}
    for result in snapshot.action_results:
        results_by_host.setdefault(result.host, []).append(result)
    rows: list[list[str]] = []
    for host_name, host in sorted(snapshot.config.hosts.items()):
        desired = desired_by_host.get(host_name)
        containers = [container for container in snapshot.plan.existing if container.host == host_name]
        latest = next(
            (
                container
                for container in containers
                if desired is not None and container.name == desired.name
            ),
            None,
        )
        if latest is not None:
            prefix = latest.labels.get(f"{LABEL_PREFIX}worker-prefix") or latest.name
            job_count = len(jobs_for_prefix(snapshot.jobs, prefix))
            runner = latest.status or latest.state or "present"
            digest = latest.labels.get(f"{LABEL_PREFIX}runtime-digest") or "unknown"
        else:
            job_count = 0
            runner = "missing"
            digest = compact_ref(snapshot.runtime_image_ref)
        old_count = sum(1 for container in containers if desired is None or container.name != desired.name)
        non_keep_actions = [action.kind for action in actions_by_host.get(host_name, ()) if action.kind != "keep"]
        failed = [result for result in results_by_host.get(host_name, ()) if result.exit_code != 0]
        live = "down" if host_name in down_hosts else "live"
        if live == "down":
            action_text = "down"
        elif failed:
            action_text = f"failed:{failed[-1].kind}"
        elif non_keep_actions:
            action_text = ",".join(non_keep_actions)
        else:
            action_text = "ok"
        rows.append(
            [
                host.name,
                host.run_target,
                live,
                runner,
                digest,
                f"{job_count}/{host.max_workers}",
                str(old_count),
                action_text,
            ]
        )
    return rows


def demand_dashboard_rows(demands: Sequence[QueueDemand]) -> list[list[str]]:
    return [
        [
            demand.profile_id or "any",
            demand.run_target or "any",
            str(demand.pending_count),
            str(demand.running_count),
            compact_ref(demand.runtime_image_ref),
        ]
        for demand in demands
    ]


def action_dashboard_rows(plan: FleetPlan, results: Sequence[ActionResult]) -> list[list[str]]:
    result_by_key = {
        (result.host, result.container, result.kind): result
        for result in results
    }
    rows = []
    for action in plan.actions:
        if action.kind == "keep":
            continue
        result = result_by_key.get((action.host, action.container, action.kind))
        if result is None:
            status = "planned"
        elif result.exit_code == 0:
            status = "ok"
        else:
            status = f"exit={result.exit_code}"
        rows.append([action.host, action.kind, status, action.container, action.reason])
    return rows


def running_job_dashboard_rows(
    jobs: Sequence[RunningJob],
    *,
    limit: int = 8,
    now: datetime | None = None,
) -> list[list[str]]:
    rows = []
    for job in list(jobs)[:limit]:
        rows.append(
            [
                str(job.id),
                job.run_target or "any",
                compact_ref(job.runtime_image_ref),
                job.run_name or "unknown",
                format_elapsed_duration_since(job.started_at, now=now),
                format_elapsed_since(job.heartbeat_at, now=now),
            ]
        )
    return rows


def stale_train_job_dashboard_rows(jobs: Sequence[StaleTrainJob], *, limit: int = 8) -> list[list[str]]:
    rows = []
    for job in list(jobs)[:limit]:
        owner = job.lease_owner or "unknown"
        host_prefix = f"rlab-{job.host}-"
        if owner.startswith(host_prefix):
            owner = owner.removeprefix(host_prefix)
        rows.append(
            [
                job.host,
                "failed" if job.execute else "would_fail",
                str(job.id),
                job.run_target or "any",
                owner,
                format_elapsed_since(job.heartbeat_at),
                job.run_name or "unknown",
            ]
        )
    return rows


def rich_available() -> bool:
    return all(
        item is not None
        for item in (
            rich_box,
            RichColumns,
            RichConsole,
            RichGroup,
            RichPanel,
            RichTable,
            RichText,
        )
    )


def rich_heat_style(ratio: float) -> str:
    if ratio >= 0.9:
        return "bright_red"
    if ratio >= 0.75:
        return "yellow"
    if ratio >= 0.55:
        return "bright_yellow"
    return "green"


def rich_text(value: Any, *, base_style: str = "") -> Any:
    if RichText is None:
        return str(value)
    text = RichText(str(value), style=base_style)
    styles = [
        (r"\bwould_fail\b", "bright_yellow"),
        (r"\bfailed\b", "bright_red"),
        (r"\bexit=\d+\b", "bright_red"),
        (r"\bdown\b", "bright_red"),
        (r"\bmissing\b", "yellow"),
        (r"\bbusy\b", "bright_green"),
        (r"\bwarning\b", "bright_yellow"),
        (r"\boffline\b", "bright_red"),
        (r"\bunreachable\b", "bright_red"),
        (r"\breachable\b", "bright_green"),
        (r"\blive\b", "bright_green"),
        (r"\bok\b", "bright_green"),
        (r"\bsteady\b", "bright_green"),
        (r"\bstart\b", "bright_cyan"),
        (r"\brestart\b", "bright_yellow"),
        (r"\bremove\b", "bright_yellow"),
        (r"\bplanned\b", "bright_cyan"),
        (r"\bnone\b", "dim"),
        (r"\bunknown\b", "dim"),
        (r"\b[0-9a-f]{12}\b", "cyan"),
    ]
    plain = text.plain
    for pattern, style in styles:
        for match in re.finditer(pattern, plain):
            text.stylize(style, match.start(), match.end())
    for match in re.finditer(
        r"\[[#-]{1,10}\]\s+(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?/\d+(?:\.\d+)?\s+[A-Za-z]+)",
        plain,
    ):
        ratio = percent_ratio(match.group(0)) or used_total_ratio(match.group(0)) or 0.0
        text.stylize(rich_heat_style(ratio), match.start(), match.end())
    return text


def rich_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    expand: bool = True,
) -> Any:
    if RichTable is None or rich_box is None:
        return format_table(headers, rows, max_width=120)
    table = RichTable(
        box=rich_box.SIMPLE,
        expand=expand,
        header_style="bold white",
        border_style="dim",
        show_edge=False,
        pad_edge=False,
    )
    for header in headers:
        table.add_column(header, overflow="ellipsis", no_wrap=True)
    for row in rows:
        table.add_row(*(rich_text(cell) for cell in row))
    return table


def rich_panel(number: int, title: str, body: Any, *, border_style: str) -> Any:
    if RichPanel is None or rich_box is None:
        return body
    return RichPanel(
        body,
        title=f"[bright_red]{number}[/] [{border_style}]{title}:[/]",
        title_align="left",
        border_style=border_style,
        box=rich_box.ROUNDED,
        padding=(0, 1),
        expand=True,
    )


def render_latest_watch_dashboard_rich(snapshot: LatestWatchSnapshot, *, max_width: int | None = None) -> str:
    if not rich_available():
        return render_latest_watch_dashboard_plain(snapshot, color=True, max_width=max_width)
    width = max_width or shutil.get_terminal_size((120, 30)).columns
    width = max(width, 72)
    console = RichConsole(
        width=width,
        force_terminal=True,
        color_system="truecolor",
        file=sys.stdout,
        _environ={**os.environ, "COLUMNS": str(width)},
    )
    action_count = len([action for action in snapshot.plan.actions if action.kind != "keep"])
    stale_count = len(snapshot.stale_train_jobs)
    failed_results = [result for result in snapshot.action_results if result.exit_code != 0]
    if failed_results or snapshot.plan.warnings or snapshot.down_hosts:
        status_style = "yellow"
        status = "attention"
    elif action_count or stale_count:
        status_style = "cyan"
        status = "applying" if snapshot.execute else "planned"
    else:
        status_style = "green"
        status = "steady"
    mode = "execute" if snapshot.execute else "dry-run"
    mode_style = "bright_yellow" if snapshot.execute else "blue"
    live_count = max(0, len(snapshot.config.hosts) - len(snapshot.down_hosts))
    down_count = len(snapshot.down_hosts)
    pending_count = sum(demand.pending_count for demand in snapshot.demands)
    running_count = sum(demand.running_count for demand in snapshot.demands)
    latest = compact_ref(snapshot.runtime_image_ref)
    summary = RichTable.grid(expand=True)
    summary.add_column(ratio=1)
    summary.add_column(justify="right")
    summary.add_row(
        rich_text(
            f"time={snapshot.captured_at.isoformat(timespec='seconds')} mode={mode} "
            f"interval={snapshot.interval:g}s latest={latest} status={status}"
        ),
        rich_text(snapshot.captured_at.strftime("%H:%M:%SZ"), base_style="bold white"),
    )
    summary.add_row(
        rich_text(
            f"hosts live={live_count} down={down_count} queue pending={pending_count} "
            f"running={running_count} actions={action_count} stale={stale_count}"
        ),
        rich_text(f"mode={mode}", base_style=mode_style),
    )
    header = RichPanel(
        summary,
        title="[bold bright_cyan]rlab fleet watch[/]",
        title_align="left",
        border_style=status_style,
        box=rich_box.ROUNDED,
        padding=(0, 1),
        expand=True,
    )

    image_rows = recent_image_dashboard_rows(snapshot.recent_images)
    image_body = rich_text("none")
    if image_rows:
        latest_commit = (
            [rich_text(f"latest_commit={snapshot.recent_images[0].commit_message}")]
            if snapshot.recent_images
            else []
        )
        image_body = RichGroup(
            *latest_commit,
            rich_table(["digest", "hash", "published", "commit"], image_rows),
        )
    images = rich_panel(
        1,
        "recent images",
        image_body,
        border_style="magenta",
    )
    hosts = rich_panel(
        2,
        "hosts",
        rich_table(
            ["host", "target", "live", "runner", "digest", "jobs/cap", "old", "action"],
            host_dashboard_rows(snapshot),
        ),
        border_style="cyan",
    )
    device_rows = active_device_dashboard_rows(snapshot.devices)
    device_summaries = [
        rich_text(
            f"{device.get('device') or device.get('id') or 'unknown'} "
            f"cpu={device_detail(device, 'cpu')} ram={device_detail(device, 'memory')} "
            f"gpu={device_detail(device, 'gpu')} vram={device_detail(device, 'vram')}"
        )
        for device in snapshot.devices
    ]
    devices = rich_panel(
        3,
        "active devices",
        RichGroup(
            *device_summaries,
            rich_table(
                ["host", "target", "state", "jobs", "cpu", "ram", "gpu", "vram", "health"],
                device_rows,
            ),
        )
        if device_rows
        else rich_text("none"),
        border_style="green",
    )
    demand_rows = demand_dashboard_rows(snapshot.demands)
    demand = rich_panel(
        4,
        "queue demand",
        rich_table(["profile", "target", "pending", "running", "digest"], demand_rows)
        if demand_rows
        else rich_text("none"),
        border_style="blue",
    )
    action_rows = action_dashboard_rows(snapshot.plan, snapshot.action_results)
    actions = rich_panel(
        5,
        "actions",
        rich_table(["host", "kind", "status", "container", "reason"], action_rows)
        if action_rows
        else rich_text("none"),
        border_style="cyan",
    )
    panels: list[Any] = [
        header,
        images,
        hosts,
        devices,
        RichColumns([demand, actions], equal=True, expand=True),
    ]
    stale_rows = stale_train_job_dashboard_rows(snapshot.stale_train_jobs)
    if stale_rows:
        panels.append(
            rich_panel(
                6,
                "stale train jobs",
                rich_table(["host", "action", "id", "target", "owner", "heartbeat", "run"], stale_rows),
                border_style="yellow",
            )
        )
    job_rows = running_job_dashboard_rows(snapshot.jobs, now=snapshot.captured_at)
    panels.append(
        rich_panel(
            7,
            "running jobs",
            rich_table(["id", "target", "digest", "run", "runtime", "heartbeat"], job_rows)
            if job_rows
            else rich_text("none"),
            border_style="green",
        )
    )
    if snapshot.plan.warnings:
        panels.append(
            rich_panel(
                8,
                "warnings",
                RichGroup(*(rich_text(f"  {warning}") for warning in snapshot.plan.warnings[:8])),
                border_style="yellow",
            )
        )
    if failed_results:
        panels.append(
            rich_panel(
                9,
                "failed actions",
                RichGroup(
                    *(
                        rich_text(
                            f"host={result.host} kind={result.kind} container={result.container} "
                            f"exit={result.exit_code} "
                            f"{result.output.splitlines()[-1] if result.output else ''}"
                        )
                        for result in failed_results[:4]
                    )
                ),
                border_style="bright_red",
            )
        )
    panels.append(rich_text("Ctrl-C to stop.", base_style="dim"))
    with console.capture() as capture:
        console.print(RichGroup(*panels), end="")
    return capture.get()


def render_latest_watch_dashboard_plain(
    snapshot: LatestWatchSnapshot,
    *,
    color: bool = True,
    max_width: int | None = None,
) -> str:
    width = max_width or shutil.get_terminal_size((120, 30)).columns
    width = max(width, 72)
    action_count = len([action for action in snapshot.plan.actions if action.kind != "keep"])
    stale_count = len(snapshot.stale_train_jobs)
    failed_results = [result for result in snapshot.action_results if result.exit_code != 0]
    if failed_results or snapshot.plan.warnings or snapshot.down_hosts:
        status_style = "yellow"
        status = "attention"
    elif action_count or stale_count:
        status_style = "cyan"
        status = "applying" if snapshot.execute else "planned"
    else:
        status_style = "green"
        status = "steady"
    mode = "execute" if snapshot.execute else "dry-run"
    mode_style = "bright_yellow" if snapshot.execute else "blue"
    live_count = max(0, len(snapshot.config.hosts) - len(snapshot.down_hosts))
    down_count = len(snapshot.down_hosts)
    pending_count = sum(demand.pending_count for demand in snapshot.demands)
    running_count = sum(demand.running_count for demand in snapshot.demands)
    title = colorize("rlab fleet watch", "bright_cyan", enabled=color)
    latest = colorize(compact_ref(snapshot.runtime_image_ref), "cyan", enabled=color)
    clock = colorize(snapshot.captured_at.strftime("%H:%M:%SZ"), "white", enabled=color)
    header = [
        f"{title} {clock}",
        (
            f"time={snapshot.captured_at.isoformat(timespec='seconds')} "
            f"{dashboard_chip('mode', mode, mode_style, color=color)} "
            f"interval={snapshot.interval:g}s latest={latest} "
            f"{dashboard_chip('status', status, status_style, color=color)}"
        ),
        (
            f"hosts "
            f"{dashboard_chip('live', str(live_count), 'bright_green', color=color)} "
            f"{dashboard_chip('down', str(down_count), 'bright_red' if down_count else 'dim', color=color)} "
            f"queue "
            f"{dashboard_chip('pending', str(pending_count), 'bright_cyan' if pending_count else 'dim', color=color)} "
            f"{dashboard_chip('running', str(running_count), 'bright_green' if running_count else 'dim', color=color)} "
            f"actions={colorize(str(action_count), 'bright_yellow' if action_count else 'dim', enabled=color)} "
            f"stale={colorize(str(stale_count), 'bright_yellow' if stale_count else 'dim', enabled=color)}"
        ),
        dashboard_divider(width, color=color),
    ]
    sections = ["\n".join(header)]
    image_rows = recent_image_dashboard_rows(snapshot.recent_images)
    sections.append(
        numbered_section(1, " recent images:", "magenta", color=color)
        + "\n"
        + (
            style_table(
                format_table(["digest", "hash", "published", "commit"], image_rows, max_width=width),
                color=color,
            )
            if image_rows
            else highlight_dashboard_text("none", color=color)
        )
    )
    sections.append(
        style_table(
            format_table(
                ["host", "target", "live", "runner", "digest", "jobs/cap", "old", "action"],
                host_dashboard_rows(snapshot),
                max_width=width,
            ),
            color=color,
        )
    )
    device_rows = active_device_dashboard_rows(snapshot.devices)
    sections.append(
        numbered_section(2, " active devices:", "green", color=color)
        + "\n"
        + (
            style_table(
                format_table(
                    ["host", "target", "state", "jobs", "cpu", "ram", "gpu", "vram", "health"],
                    device_rows,
                    max_width=width,
                ),
                color=color,
            )
            if device_rows
            else highlight_dashboard_text("none", color=color)
        )
    )
    demand_rows = demand_dashboard_rows(snapshot.demands)
    sections.append(
        numbered_section(3, " queue demand:", "blue", color=color)
        + "\n"
        + (
            style_table(
                format_table(["profile", "target", "pending", "running", "digest"], demand_rows, max_width=width),
                color=color,
            )
            if demand_rows
            else highlight_dashboard_text("none", color=color)
        )
    )
    action_rows = action_dashboard_rows(snapshot.plan, snapshot.action_results)
    sections.append(
        numbered_section(4, " actions:", "cyan", color=color)
        + "\n"
        + (
            style_table(
                format_table(["host", "kind", "status", "container", "reason"], action_rows, max_width=width),
                color=color,
            )
            if action_rows
            else highlight_dashboard_text("none", color=color)
        )
    )
    stale_rows = stale_train_job_dashboard_rows(snapshot.stale_train_jobs)
    if stale_rows:
        sections.append(
            numbered_section(5, " stale train jobs:", "yellow", color=color)
            + "\n"
            + style_table(
                format_table(
                    ["host", "action", "id", "target", "owner", "heartbeat", "run"],
                    stale_rows,
                    max_width=width,
                ),
                color=color,
            )
        )
    job_rows = running_job_dashboard_rows(snapshot.jobs, now=snapshot.captured_at)
    sections.append(
        numbered_section(6, " running jobs:", "green", color=color)
        + "\n"
        + (
            style_table(
                format_table(["id", "target", "digest", "run", "runtime", "heartbeat"], job_rows, max_width=width),
                color=color,
            )
            if job_rows
            else highlight_dashboard_text("none", color=color)
        )
    )
    if snapshot.plan.warnings:
        sections.append(
            numbered_section(7, " warnings:", "yellow", color=color)
            + "\n"
            + "\n".join(highlight_dashboard_text(f"  {warning}", color=color) for warning in snapshot.plan.warnings[:8])
        )
    if failed_results:
        lines = [numbered_section(8, " failed actions:", "bright_red", color=color)]
        for result in failed_results[:4]:
            tail = result.output.splitlines()[-1] if result.output else ""
            lines.append(
                highlight_dashboard_text(
                    f"  host={result.host} kind={result.kind} container={result.container} "
                    f"exit={result.exit_code} {tail}",
                    color=color,
                )
            )
        sections.append("\n".join(lines))
    sections.append(colorize("Ctrl-C to stop.", "dim", enabled=color))
    return "\n\n".join(sections)


def render_latest_watch_dashboard(
    snapshot: LatestWatchSnapshot,
    *,
    color: bool = True,
    max_width: int | None = None,
) -> str:
    if color and rich_available():
        return render_latest_watch_dashboard_rich(snapshot, max_width=max_width)
    return render_latest_watch_dashboard_plain(snapshot, color=color, max_width=max_width)


def requested_image_label(args: argparse.Namespace) -> str:
    image = str(getattr(args, "image", "") or "").strip()
    if image:
        if image == "latest":
            return "latest successful train image"
        try:
            return compact_ref(image)
        except ValueError:
            return image
    image_file = getattr(args, "image_file", None)
    if image_file:
        return str(image_file)
    value = getattr(args, "runtime_image_ref", None)
    if value:
        try:
            return compact_ref(value)
        except ValueError:
            return str(value)
    ref_file = getattr(args, "runtime_image_ref_file", None)
    if ref_file:
        return str(ref_file)
    return "latest successful train image"


def render_latest_watch_starting_dashboard(
    args: argparse.Namespace,
    *,
    color: bool = True,
    max_width: int | None = None,
) -> str:
    width = max_width or shutil.get_terminal_size((120, 30)).columns
    width = max(width, 72)
    mode = "execute" if args.execute else "dry-run"
    mode_style = "bright_yellow" if args.execute else "blue"
    host = args.host or "all"
    title = colorize("rlab fleet watch", "bright_cyan", enabled=color)
    header = [
        title,
        (
            f"time={datetime.now(UTC).isoformat(timespec='seconds')} "
            f"{dashboard_chip('mode', mode, mode_style, color=color)} "
            f"interval={args.interval:g}s host={colorize(host, 'cyan', enabled=color)} "
            f"latest={colorize(requested_image_label(args), 'cyan', enabled=color)} "
            f"{dashboard_chip('status', 'starting', 'bright_cyan', color=color)}"
        ),
        dashboard_divider(width, color=color),
    ]
    body = [
        colorize("polling now...", "bright_cyan", enabled=color),
        colorize("resolving image, reading queue state, and checking SSH/Docker hosts", "dim", enabled=color),
        "",
        colorize("Ctrl-C to stop.", "dim", enabled=color),
    ]
    return "\n".join([*header, *body])


def render_watch_latest_lock_busy(
    error: WatchLatestLockBusy,
    *,
    color: bool = True,
    max_width: int | None = None,
) -> str:
    width = max_width or shutil.get_terminal_size((120, 30)).columns
    width = max(width, 72)
    lines = [
        colorize("rlab fleet watch", "bright_cyan", enabled=color),
        dashboard_divider(width, color=color),
        colorize("another watch session already owns the lock", "bright_yellow", enabled=color),
        f"lock={error.path}",
    ]
    if error.owner:
        lines.extend(["owner:", f"  {error.owner}"])
    lines.append("Stop the existing session before starting another one.")
    return "\n".join(lines)


def write_tui_frame(text: str, *, enabled: bool) -> None:
    if enabled and sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
    print(text, flush=True)


def cmd_ensure_latest(args: argparse.Namespace) -> int:
    while True:
        config, runtime_image_ref, plan = build_live_ensure_latest_plan(args)
        print(f"runtime_image_ref={runtime_image_ref}")
        print(f"hosts={','.join(sorted(config.hosts))}")
        print(format_plan(plan))
        if not args.execute:
            print("dry_run: rerun without --dry-run to apply the plan")
            return 0
        status = run_plan_actions(config, plan, local=False)
        if status != 0 or not args.watch:
            return status
        time.sleep(args.interval)


def cmd_watch_latest(args: argparse.Namespace) -> int:
    if getattr(args, "machine", None):
        return cmd_container_watch_dashboard(args)
    color = not args.no_color
    tui = not args.no_tui
    try:
        lock = acquire_watch_latest_lock(args)
    except WatchLatestLockBusy as exc:
        write_tui_frame(
            render_watch_latest_lock_busy(exc, color=color, max_width=args.width),
            enabled=tui,
        )
        return 2
    try:
        write_tui_frame(
            render_latest_watch_starting_dashboard(args, color=color, max_width=args.width),
            enabled=tui,
        )
        image_resolver = RuntimeImageResolver(args, default_latest=True)
        while True:
            try:
                snapshot = build_latest_watch_snapshot(
                    args,
                    image_resolver=image_resolver,
                )
                action_results: tuple[ActionResult, ...] = ()
                if args.execute:
                    action_results = run_latest_watch_actions(snapshot.config, snapshot.plan)
                    snapshot = replace(snapshot, action_results=action_results)
                write_tui_frame(
                    render_latest_watch_dashboard(snapshot, color=color, max_width=args.width),
                    enabled=tui,
                )
                exit_code = max((result.exit_code for result in action_results), default=0)
                if args.once:
                    return exit_code
                if exit_code != 0 and args.fail_fast:
                    return exit_code
            except KeyboardInterrupt:
                print("\nwatch stopped")
                return 130
            except Exception as exc:
                width = args.width or shutil.get_terminal_size((120, 30)).columns
                message = (
                    colorize("rlab fleet watch", "bright_cyan", enabled=color)
                    + "\n"
                    + dashboard_divider(width, color=color)
                    + "\n"
                    + colorize(f"snapshot failed: {exc}", "bright_red", enabled=color)
                    + "\n\nCtrl-C to stop."
                )
                write_tui_frame(message, enabled=tui)
                if args.once:
                    return 1
                if args.fail_fast:
                    return 1
            time.sleep(args.interval)
    finally:
        release_watch_latest_lock(lock)


def cmd_setup_host(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    runtime_image_ref = image_ref_from_args(args)
    status = 0
    for host in selected_hosts(config, args.host):
        script = setup_host_script(host, runtime_image_ref=runtime_image_ref)
        print(f"host: {host.name}")
        print(script.rstrip())
        if not args.execute:
            print("dry_run: rerun without --dry-run to run setup over SSH")
            continue
        result = run_host_script(host, script)
        if result.returncode != 0:
            status = int(result.returncode)
    return status


def print_stale_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    execute: bool,
) -> None:
    action = "failed" if execute else "would_fail"
    print(f"stale_train_jobs_{action}={len(rows)}")
    for row in rows:
        print(
            "  "
            f"train_job_id={row['id']} "
            f"profile={row.get('profile_id') or 'any'} "
            f"target={row.get('run_target') or 'any'} "
            f"owner={row.get('stale_lease_owner') or 'unknown'} "
            f"heartbeat={row.get('stale_heartbeat_at') or 'unknown'} "
            f"run={row.get('run_name') or ''}"
        )


def cmd_mark_stale_failed(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    if args.host not in config.hosts:
        known = ", ".join(sorted(config.hosts))
        raise SystemExit(f"unknown fleet host {args.host!r}; known hosts: {known}")
    host = config.hosts[args.host]
    lease_owner_prefix = args.lease_owner_prefix or stale_lease_owner_prefix_for_host(host)
    conn = _connect_from_args(args)
    try:
        func = mark_stale_train_jobs_failed if args.execute else list_stale_train_jobs
        if args.execute:
            rows = func(
                conn,
                job_ids=args.job_id,
                run_target=host.run_target,
                lease_owner_prefix=lease_owner_prefix,
                older_than_seconds=args.older_than_seconds,
                limit=args.limit,
                error=args.error,
            )
        else:
            rows = func(
                conn,
                job_ids=args.job_id,
                run_target=host.run_target,
                lease_owner_prefix=lease_owner_prefix,
                older_than_seconds=args.older_than_seconds,
                limit=args.limit,
            )
    finally:
        conn.close()
    print(
        f"host={host.name} target={host.run_target} "
        f"lease_owner_prefix={lease_owner_prefix} mode={'execute' if args.execute else 'dry-run'}"
    )
    print_stale_train_rows(rows, execute=args.execute)
    if not args.execute:
        print("dry_run: rerun without --dry-run to mark these stale train jobs failed")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES_CONFIG)
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Apply planned changes; this is the default.",
    )
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Preview planned changes without applying them.",
    )


def add_reconcile_args(parser: argparse.ArgumentParser, *, host_required: bool = False) -> None:
    parser.add_argument(
        "--host",
        required=host_required,
        help="Limit reconciliation to one fleet host.",
    )
    add_dry_run_arg(parser)
    parser.add_argument("--watch", action="store_true", help="Run repeatedly.")
    parser.add_argument("--interval", type=float, default=30.0, help="Watch interval in seconds.")


def add_runtime_image_args(
    parser: argparse.ArgumentParser,
    *,
    allow_latest: bool = False,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-image-ref")
    group.add_argument("--runtime-image-ref-file", type=Path)
    if allow_latest:
        group.add_argument(
            "--latest-image",
            action="store_true",
            help="Use the latest successful train-image CI artifact; default for ensure-runner.",
        )
        parser.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
        parser.add_argument("--image-branch", default=DEFAULT_IMAGE_BRANCH)
        parser.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)


def add_ensure_image_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--image",
        help="'latest' or an immutable docker:...@sha256:... runtime image ref. Defaults to latest.",
    )
    group.add_argument(
        "--image-file",
        type=Path,
        help="JSON artifact or plain-text file containing the immutable runtime image ref.",
    )
    group.add_argument("--runtime-image-ref", help=argparse.SUPPRESS)
    group.add_argument("--runtime-image-ref-file", type=Path, help=argparse.SUPPRESS)
    group.add_argument("--latest-image", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    parser.add_argument("--image-branch", default=DEFAULT_IMAGE_BRANCH)
    parser.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage rlab runner containers from queue state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print digest-pinned train queue demand.")
    add_common_args(status)
    status.set_defaults(func=cmd_status)

    ps = subparsers.add_parser("ps", help="List managed runner containers across fleet hosts.")
    add_common_args(ps)
    ps.add_argument("--host", help="Limit listing to one fleet host.")
    ps.set_defaults(func=cmd_ps)

    plan = subparsers.add_parser("plan", help="Plan runner container changes.")
    add_common_args(plan)
    plan.add_argument("--host", help="Limit planning to one fleet host.")
    plan.set_defaults(func=cmd_plan)

    policy = subparsers.add_parser("policy", help="Print the repo capacity policy.")
    add_common_args(policy)
    policy.add_argument("--policy", type=Path, default=DEFAULT_CAPACITY_POLICY)
    policy.set_defaults(func=cmd_policy)

    reconcile = subparsers.add_parser("reconcile", help="Reconcile remote Docker hosts over SSH.")
    add_common_args(reconcile)
    add_reconcile_args(reconcile)
    reconcile.add_argument(
        "--machine",
        help="Use one-job-container reconciliation for this machine.",
    )
    reconcile.set_defaults(func=cmd_reconcile)

    launch = subparsers.add_parser("launch", help="Launch one claimed job as one container.")
    add_common_args(launch)
    launch.add_argument("--machine", required=True)
    launch.add_argument("--job-id", type=int, required=True)
    launch.add_argument("--job-kind", choices=("train",), default="train")
    add_dry_run_arg(launch)
    launch.set_defaults(func=cmd_container_launch)

    launch_next = subparsers.add_parser(
        "launch-next",
        help="Fill available machine container slots with pending jobs.",
    )
    add_common_args(launch_next)
    launch_next.add_argument("--machine", required=True)
    launch_next.add_argument("--job-kind", choices=("train",), default="train")
    launch_next.add_argument("--limit", type=int, default=1)
    add_dry_run_arg(launch_next)
    launch_next.set_defaults(func=cmd_container_launch_next)

    shepherd = subparsers.add_parser(
        "shepherd",
        help="Run the mutating one-job-container orchestration loop for one machine.",
    )
    add_common_args(shepherd)
    shepherd.add_argument("--machine", required=True)
    shepherd.add_argument("--job-kind", choices=("train",), default="train")
    shepherd.add_argument("--limit", type=int, default=1)
    shepherd.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds.")
    shepherd.add_argument("--once", action="store_true", help="Run one reconcile/fill pass and exit.")
    shepherd.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit when a poll or action fails instead of retrying forever.",
    )
    add_dry_run_arg(shepherd)
    shepherd.set_defaults(func=cmd_container_shepherd)

    ensure = subparsers.add_parser(
        "ensure-runner",
        help="Ensure one managed runner exists for a host/profile/image.",
    )
    add_common_args(ensure)
    ensure.add_argument("--host", required=True, help="Fleet host to run the container on.")
    ensure.add_argument("--profile", help="Optional exact train_jobs.profile_id to claim.")
    ensure.add_argument("--target", help="Run target; defaults to the host canonical target.")
    ensure.add_argument("--workers", type=int, help="Workers inside the runner; defaults to host capacity.")
    add_dry_run_arg(ensure)
    add_ensure_image_args(ensure)
    ensure.set_defaults(func=cmd_ensure_runner)

    ensure_latest = subparsers.add_parser(
        "ensure-latest",
        help="Ensure selected fleet hosts run the latest image and remove idle old-image runners.",
    )
    add_common_args(ensure_latest)
    ensure_latest.add_argument("--host", help="Limit rollout to one fleet host.")
    ensure_latest.add_argument("--workers", type=int, help="Workers inside each latest runner; defaults to host capacity.")
    add_dry_run_arg(ensure_latest)
    ensure_latest.add_argument("--watch", action="store_true", help="Run repeatedly.")
    ensure_latest.add_argument("--interval", type=float, default=30.0, help="Watch interval in seconds.")
    add_ensure_image_args(ensure_latest)
    ensure_latest.set_defaults(func=cmd_ensure_latest)

    watch_latest = subparsers.add_parser(
        "watch",
        help="Run a live dashboard; machine mode is read-only.",
    )
    add_common_args(watch_latest)
    watch_latest.add_argument("--host", help="Limit monitoring to one fleet host.")
    watch_latest.add_argument(
        "--machine",
        help="Use recoverable one-job-container orchestration for this machine.",
    )
    watch_latest.add_argument("--job-kind", choices=("train",), default="train")
    watch_latest.add_argument("--limit", type=int, default=1, help="Max jobs to launch per poll.")
    watch_latest.add_argument(
        "--workers",
        type=int,
        help="Workers inside each latest runner; defaults to host capacity.",
    )
    add_dry_run_arg(watch_latest)
    watch_latest.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_WATCH_LATEST_INTERVAL_SECONDS,
        help="Polling interval in seconds; defaults to 15.",
    )
    watch_latest.add_argument("--once", action="store_true", help="Render/apply one poll and exit.")
    watch_latest.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit when a poll or action fails instead of retrying forever.",
    )
    watch_latest.add_argument(
        "--no-claim-stale-jobs",
        dest="claim_stale_jobs",
        action="store_false",
        help="Disable stale running train job detection and failure marking.",
    )
    watch_latest.add_argument(
        "--stale-older-than-seconds",
        type=int,
        default=DEFAULT_WATCH_STALE_OLDER_THAN_SECONDS,
        help="Treat running train jobs with no recent heartbeat as stale after this many seconds.",
    )
    watch_latest.add_argument(
        "--stale-limit",
        type=int,
        default=DEFAULT_WATCH_STALE_LIMIT,
        help="Maximum stale train jobs to inspect or fail per host; 0 means no limit.",
    )
    watch_latest.add_argument("--no-tui", action="store_true", help="Do not clear/redraw the terminal.")
    watch_latest.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    watch_latest.add_argument("--width", type=int, help="Override dashboard width.")
    add_ensure_image_args(watch_latest)
    watch_latest.set_defaults(func=cmd_watch_latest)

    mark_stale = subparsers.add_parser(
        "mark-stale-failed",
        help="Mark stale running train jobs for one fleet host failed.",
    )
    add_common_args(mark_stale)
    mark_stale.add_argument("--host", required=True, help="Fleet host whose lost workers owned the jobs.")
    mark_stale.add_argument("--job-id", type=int, action="append", default=[])
    mark_stale.add_argument(
        "--lease-owner-prefix",
        help="Override the derived host worker-prefix filter.",
    )
    mark_stale.add_argument("--older-than-seconds", type=int, default=300)
    mark_stale.add_argument("--limit", type=int, default=50, help="Maximum rows to affect; 0 means no limit.")
    mark_stale.add_argument("--error", help="Failure message to store on job/result rows.")
    add_dry_run_arg(mark_stale)
    mark_stale.set_defaults(func=cmd_mark_stale_failed)

    setup = subparsers.add_parser("setup-host", help="Prepare SSH Docker hosts for runners.")
    add_common_args(setup)
    setup.add_argument("--host", required=True, help="Fleet host to set up.")
    add_dry_run_arg(setup)
    add_runtime_image_args(setup)
    setup.set_defaults(func=cmd_setup_host)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
