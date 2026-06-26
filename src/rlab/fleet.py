from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.campaign import connect, database_url
from rlab.compute_targets import instance_defaults, load_instance_config
from rlab.json_utils import json_safe
from rlab.runtime_refs import (
    normalize_runtime_image_ref,
    runtime_image_digest_slug,
    runtime_image_ref_from_file,
)


DEFAULT_FLEET_CONFIG = Path("experiments/fleet.json")
DEFAULT_INSTANCES_CONFIG = Path("experiments/instances.json")
DEFAULT_IMAGE_WORKFLOW = "rlab train image"
DEFAULT_IMAGE_BRANCH = "main"
DEFAULT_IMAGE_ARTIFACT = "rlab-train-image"
DEFAULT_IMAGE_ARTIFACT_FILE = "rlab-train-image.json"
LABEL_PREFIX = "rlab."
MANAGED_LABEL = f"{LABEL_PREFIX}managed"
CONFIG_HASH_LABEL = f"{LABEL_PREFIX}config-hash"


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
    profile_id: str
    runtime_image_ref: str
    run_target: str | None
    pending_count: int
    running_count: int
    max_priority: int
    oldest_job_id: int

    @property
    def total(self) -> int:
        return self.pending_count + self.running_count


@dataclass(frozen=True)
class ActiveLease:
    lease_owner: str
    profile_id: str
    runtime_image_ref: str
    run_target: str | None
    running_count: int


@dataclass(frozen=True)
class DeploymentKey:
    host: str
    profile_id: str | None
    runtime_image_ref: str
    run_target: str | None


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
        return DeploymentKey(
            host=self.host,
            profile_id=profile_id,
            runtime_image_ref=runtime_image_ref,
            run_target=run_target,
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


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected string or list, got {type(value).__name__}")


def _host_config_from_raw(
    *,
    name: str,
    raw: Mapping[str, Any],
    instances: Mapping[str, Any],
) -> HostConfig:
    run_target = str(raw.get("run_target") or name).strip()
    instance = instance_defaults(dict(instances), run_target)
    max_workers = int(
        raw.get("max_workers")
        or instance.get("max_children")
        or instance.get("children")
        or 1
    )
    ssh_target = str(raw.get("ssh_target") or "").strip()
    if not ssh_target:
        raise ValueError(f"fleet host {name!r} must define ssh_target")
    return HostConfig(
        name=name,
        ssh_target=ssh_target,
        ssh_options=_tuple(raw.get("ssh_options", ())),
        run_target=str(instance.get("name", run_target)),
        max_workers=max_workers,
        base_dir=str(raw.get("base_dir") or raw.get("repo_dir") or "/home/tsilva/rlab"),
        env_file=str(raw.get("env_file") or "/home/tsilva/rlab/.env.runner"),
        runs_dir=str(raw.get("runs_dir") or "/home/tsilva/rlab/runs"),
        logs_dir=str(raw.get("logs_dir") or "/home/tsilva/rlab/logs"),
        rom_dir=str(raw.get("rom_dir") or "/home/tsilva/roms"),
        state_dir=str(raw.get("state_dir") or "/home/tsilva/rlab/fleet"),
        container_runs_dir=str(raw.get("container_runs_dir") or "/root/rlab/runs"),
        container_logs_dir=str(raw.get("container_logs_dir") or "/root/rlab/logs"),
        container_rom_dir=str(raw.get("container_rom_dir") or "/roms"),
        log_dir_in_container=str(raw.get("log_dir_in_container") or "/root/rlab/logs/train_runner"),
        gpu_test_image=str(raw.get("gpu_test_image") or "nvidia/cuda:12.9.1-base-ubuntu22.04"),
        docker_command=_tuple(raw.get("docker_command") or ("docker",)),
        docker_network=str(raw.get("docker_network") or "").strip() or None,
        pull_policy=str(raw.get("pull_policy") or "always"),
        extra_env=_tuple(raw.get("extra_env", ())),
    )


def load_fleet_config(
    repo_root: Path,
    *,
    fleet_path: Path | None = None,
    instances_path: Path | None = None,
) -> FleetConfig:
    fleet_data = load_json_file(resolve_repo_path(repo_root, fleet_path, DEFAULT_FLEET_CONFIG))
    instances = load_instance_config(
        repo_root,
        resolve_repo_path(repo_root, instances_path, DEFAULT_INSTANCES_CONFIG),
    )
    hosts_raw = fleet_data.get("hosts")
    if not isinstance(hosts_raw, dict) or not hosts_raw:
        raise ValueError("fleet config must define hosts")
    hosts = {
        str(name): _host_config_from_raw(name=str(name), raw=raw, instances=instances)
        for name, raw in hosts_raw.items()
        if isinstance(raw, dict)
    }
    policies_raw = fleet_data.get("profile_policies", [{"profile_id": "*", "hosts": list(hosts)}])
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


def resolve_repo_path(repo_root: Path, path: Path | None, default: Path) -> Path:
    candidate = path or default
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


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
    profile = sanitize_slug(key.profile_id or "any-profile", limit=44)
    target = sanitize_slug(key.run_target or "any", limit=16)
    host = sanitize_slug(key.host, limit=16)
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
    cmd.extend([image, "rlab-container-entrypoint", "rlab-train-runner"])
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
        "--worker-id",
        worker_prefix,
        "--log-dir",
        host.log_dir_in_container,
    ]
    if key.profile_id:
        command = ["--profile", key.profile_id, *command]
    hash_input = {
        "host": host.name,
        "profile_id": key.profile_id,
        "runtime_image_ref": key.runtime_image_ref,
        "run_target": key.run_target,
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
        f"{LABEL_PREFIX}worker-prefix": worker_prefix,
    }
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
    names = _matching_policy_hosts(config, demand.profile_id)
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
            -item.max_priority,
            item.oldest_job_id,
            item.profile_id,
            item.runtime_image_ref,
            item.run_target or "",
        ),
    )
    for demand in sorted_demands:
        hosts = eligible_hosts(config, demand)
        if not hosts:
            warnings.append(
                "no eligible host for "
                f"profile={demand.profile_id} target={demand.run_target or 'any'}"
            )
            continue
        chosen = next((host for host in hosts if remaining[host.name] > 0), None)
        if chosen is None:
            warnings.append(
                "capacity exhausted for "
                f"profile={demand.profile_id} target={demand.run_target or 'any'}"
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
                f"{demand.profile_id} on {chosen.name}"
            )
    return tuple(desired), tuple(warnings)


def demand_index(demands: Sequence[QueueDemand]) -> dict[tuple[str, str, str | None], QueueDemand]:
    return {
        (demand.profile_id, demand.runtime_image_ref, demand.run_target): demand
        for demand in demands
    }


def demand_matches_key(demand: QueueDemand, key: DeploymentKey) -> bool:
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
        and key.profile_id is None
        and key.runtime_image_ref == desired.key.runtime_image_ref
        and key.run_target == desired.key.run_target
    )


def active_prefixes(leases: Sequence[ActiveLease]) -> tuple[str, ...]:
    return tuple(sorted({lease.lease_owner for lease in leases if lease.lease_owner}))


def container_has_active_lease(container: ExistingContainer, leases: Sequence[ActiveLease]) -> bool:
    prefix = container.labels.get(f"{LABEL_PREFIX}worker-prefix") or container.name
    return any(lease.lease_owner.startswith(prefix) for lease in leases)


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
    demand_by_key = demand_index(demands)
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
        if current.name in desired_by_name:
            continue
        key = current.key
        matching_demand = (
            demand_by_key.get((key.profile_id, key.runtime_image_ref, key.run_target))
            if key is not None
            else None
        )
        if matching_demand is None and key is not None and key.profile_id is None:
            matching_demand = next(
                (demand for demand in demands if demand_matches_key(demand, key) and demand.total > 0),
                None,
            )
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


QUEUE_DEMAND_SQL = """
SELECT
  profile_id,
  runtime_image_ref,
  run_target,
  COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
  COUNT(*) FILTER (WHERE status = 'running') AS running_count,
  COALESCE(MAX(priority), 0) AS max_priority,
  MIN(id) AS oldest_job_id
FROM train_jobs
WHERE runtime_image_ref IS NOT NULL
  AND cancel_requested = FALSE
  AND status IN ('pending', 'running')
GROUP BY profile_id, runtime_image_ref, run_target
ORDER BY max_priority DESC, oldest_job_id ASC
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


def queue_demands(conn) -> list[QueueDemand]:
    with conn.cursor() as cur:
        cur.execute(QUEUE_DEMAND_SQL)
        rows = cur.fetchall()
    demands = []
    for row in rows:
        demands.append(
            QueueDemand(
                profile_id=str(row["profile_id"]),
                runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
                run_target=str(row["run_target"]) if row["run_target"] else None,
                pending_count=int(row["pending_count"]),
                running_count=int(row["running_count"]),
                max_priority=int(row["max_priority"]),
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
            profile_id=str(row["profile_id"]),
            runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
            run_target=str(row["run_target"]) if row["run_target"] else None,
            running_count=int(row["running_count"]),
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
    if not action.commands:
        return 0
    host = config.hosts[action.host]
    script = "set -euo pipefail\n" + "\n".join(action.commands)
    result = run_host_script(host, script, local=local)
    return int(result.returncode)


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


def _run_gh_json(command: Sequence[str]) -> Any:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to resolve --latest-image") from exc
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(f"gh command failed: {shell_join(command)}\n{output}")
    return json.loads(result.stdout or "null")


def latest_runtime_image_ref(
    *,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str = DEFAULT_IMAGE_BRANCH,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
) -> str:
    runs = _run_gh_json(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "databaseId",
        ]
    )
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"no successful {workflow!r} runs found on branch {branch!r}")
    run_id = str(runs[0].get("databaseId") or "").strip()
    if not run_id:
        raise RuntimeError(f"latest {workflow!r} run did not include a databaseId")
    with tempfile.TemporaryDirectory(prefix="rlab-train-image-") as tmp:
        result = subprocess.run(
            [
                "gh",
                "run",
                "download",
                run_id,
                "--name",
                artifact_name,
                "--dir",
                tmp,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
            raise RuntimeError(f"failed to download artifact {artifact_name!r} from run {run_id}\n{output}")
        return runtime_image_ref_from_file(Path(tmp) / DEFAULT_IMAGE_ARTIFACT_FILE)


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


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(getattr(args, "direct", False)))


def _load_config_from_args(args: argparse.Namespace) -> FleetConfig:
    return load_fleet_config(
        repo_root_from_args(args),
        fleet_path=args.fleet_config,
        instances_path=args.instances,
    )


def repo_root_from_args(args: argparse.Namespace) -> Path:
    return Path(args.repo_root).expanduser().resolve()


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
            f"profile={demand.profile_id} target={demand.run_target or 'any'} "
            f"pending={demand.pending_count} running={demand.running_count} "
            f"digest={runtime_image_digest_slug(demand.runtime_image_ref)}"
        )
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
                f"{item.name} host={item.key.host} workers={item.workers} "
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
                f"profile={lease.profile_id} target={lease.run_target or 'any'}"
            )
    else:
        print("active leases: none")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    print(format_plan(build_live_plan(args)))
    return 0


def _run_reconcile_once(args: argparse.Namespace, *, local: bool = False) -> int:
    config = filter_config_to_host(_load_config_from_args(args), getattr(args, "host", None))
    plan = build_live_plan(args, local=local)
    print(format_plan(plan))
    if not args.execute:
        print("dry_run: pass --execute to apply the plan")
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
        print("dry_run: pass --execute to apply the plan")
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


def cmd_setup_host(args: argparse.Namespace) -> int:
    config = _load_config_from_args(args)
    runtime_image_ref = image_ref_from_args(args)
    status = 0
    for host in selected_hosts(config, args.host):
        script = setup_host_script(host, runtime_image_ref=runtime_image_ref)
        print(f"host: {host.name}")
        print(script.rstrip())
        if not args.execute:
            print("dry_run: pass --execute to run setup over SSH")
            continue
        result = run_host_script(host, script)
        if result.returncode != 0:
            status = int(result.returncode)
    return status


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fleet-config", type=Path, default=DEFAULT_FLEET_CONFIG)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES_CONFIG)
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")


def add_reconcile_args(parser: argparse.ArgumentParser, *, host_required: bool = False) -> None:
    parser.add_argument(
        "--host",
        required=host_required,
        help="Limit reconciliation to one fleet host.",
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes instead of dry-run.")
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

    plan = subparsers.add_parser("plan", help="Plan runner container changes.")
    add_common_args(plan)
    plan.add_argument("--host", help="Limit planning to one fleet host.")
    plan.set_defaults(func=cmd_plan)

    reconcile = subparsers.add_parser("reconcile", help="Reconcile remote Docker hosts over SSH.")
    add_common_args(reconcile)
    add_reconcile_args(reconcile)
    reconcile.set_defaults(func=cmd_reconcile)

    ensure = subparsers.add_parser(
        "ensure-runner",
        help="Ensure one managed runner exists for a host/profile/image.",
    )
    add_common_args(ensure)
    ensure.add_argument("--host", required=True, help="Fleet host to run the container on.")
    ensure.add_argument("--profile", help="Optional exact train_jobs.profile_id to claim.")
    ensure.add_argument("--target", help="Run target; defaults to the host canonical target.")
    ensure.add_argument("--workers", type=int, help="Workers inside the runner; defaults to host capacity.")
    ensure.add_argument("--execute", action="store_true")
    add_ensure_image_args(ensure)
    ensure.set_defaults(func=cmd_ensure_runner)

    setup = subparsers.add_parser("setup-host", help="Prepare SSH Docker hosts for runners.")
    add_common_args(setup)
    setup.add_argument("--host", required=True, help="Fleet host to set up.")
    setup.add_argument("--execute", action="store_true")
    add_runtime_image_args(setup)
    setup.set_defaults(func=cmd_setup_host)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
