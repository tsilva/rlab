from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlab.cli_args import add_direct_database_arg, add_dry_run_arg
from rlab.docker_host import (
    DockerRunnerHost,
    JobContainer,
    run_wandb_publisher_recovery_container,
    setup_docker_host,
)
from rlab.job_queue import (
    QueueDemand,
    TRAIN_JOB_KIND,
    active_job_launches,
    claim_job_launch,
    claim_live_publication_recovery,
    connect,
    database_url,
    finish_job_launch_from_result,
    finish_live_publication_recovery,
    job_payload_for_launch,
    machine_control,
    mark_job_launch_running,
    mark_train_job_ready,
    new_train_launch_id,
    next_pending_train_job,
    queue_demands,
    record_job_launch_error,
    set_machine_control,
)
from rlab.telemetry_mailbox import (
    ATTEMPT_ID_ENV,
    ATTEMPT_TOKEN_ENV,
    MAILBOX_DATABASE_ENV,
    TRANSPORT_NAME,
    issue_worker_attempt_token,
)
from rlab.json_utils import json_safe
from rlab.machines import (
    DEFAULT_MACHINE_REGISTRY,
    MachineConfig,
    MachineRegistry,
    load_machine_registry,
    resolve_machine,
)
from rlab.runtime_refs import (
    docker_image_ref,
    recent_runtime_images,
    runtime_image_ref_from_args,
    normalize_runtime_image_ref,
)
from rlab.fleet_labels import (
    JOB_CONTAINER_LABEL,
    JOB_ID_LABEL,
    JOB_KIND_LABEL,
    DEFAULT_RUNTIME_IMAGE_REPOSITORIES,
    LABEL_PREFIX,
    LAUNCH_ID_LABEL,
    MACHINE_LABEL,
    MANAGED_LABEL,
    OUTPUT_URI_LABEL,
)

DEFAULT_SHARED_RUNNER_ENV_FILE = Path(".env")
SHARED_RUNNER_ENV_KEYS = (
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT_URL",
    "AWS_REGION",
    "CHECKPOINT_BUCKET_URI",
)
MAILBOX_RUNNER_ENV_KEYS = (
    MAILBOX_DATABASE_ENV,
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT_URL",
    "AWS_REGION",
    "CHECKPOINT_BUCKET_URI",
)


@dataclass(frozen=True)
class MachineMutationLock:
    machine: str
    key: str


class MachineLockBusy(RuntimeError):
    def __init__(self, machine: str) -> None:
        super().__init__(f"another reconciler is already running for machine={machine}")
        self.machine = machine


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


def sanitize_slug(value: str, *, limit: int = 40) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "value"
    return slug[:limit].strip("-") or "value"


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(getattr(args, "direct", False)))


def load_shared_runner_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"shared runner env file is missing: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in SHARED_RUNNER_ENV_KEYS:
            continue
        if key in values:
            raise RuntimeError(f"shared runner env file defines {key} more than once: {path}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise RuntimeError(f"shared runner env value is empty: {key} in {path}")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise RuntimeError(
                f"shared runner env value contains an invalid control character: {key}"
            )
        values[key] = value
    missing = [key for key in SHARED_RUNNER_ENV_KEYS if key not in values]
    if missing:
        raise RuntimeError(
            f"shared runner env file is missing required key(s): {', '.join(missing)} in {path}"
        )
    return values


def load_mailbox_runner_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"runner env file is missing: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in MAILBOX_RUNNER_ENV_KEYS:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value or any(char in value for char in ("\x00", "\n", "\r")):
            raise RuntimeError(f"runner env value is invalid: {key} in {path}")
        values[key] = value
    missing = [key for key in MAILBOX_RUNNER_ENV_KEYS if key not in values]
    if missing:
        raise RuntimeError(
            f"runner env file is missing mailbox key(s): {', '.join(missing)} in {path}"
        )
    return values


def load_registry_from_args(args: argparse.Namespace) -> MachineRegistry:
    return load_machine_registry(args.machines)


def sanitize_container_part(value: str, *, limit: int = 32) -> str:
    return sanitize_slug(value, limit=limit)


def job_container_name(machine: MachineConfig, *, launch_id: str) -> str:
    return (
        f"rlab-job-{sanitize_container_part(machine.name, limit=16)}-"
        f"{TRAIN_JOB_KIND}-"
        f"{sanitize_container_part(launch_id, limit=48)}"
    )[:120].strip("-")


def runtime_image_repository(runtime_image_ref: str) -> str | None:
    try:
        image = docker_image_ref(runtime_image_ref)
    except ValueError:
        return None
    if "@" not in image:
        return None
    repository, _ = image.split("@", 1)
    return repository or None


def protected_runtime_image_refs(
    *,
    machine: MachineConfig,
    demands: Sequence[QueueDemand],
    containers: Sequence[JobContainer],
) -> set[str]:
    protected: set[str] = set()
    for demand in demands:
        if demand.total <= 0:
            continue
        if demand.machine != machine.name:
            continue
        protected.add(normalize_runtime_image_ref(demand.runtime_image_ref))
    for container in containers:
        if container.state in {"removing", "dead"}:
            continue
        runtime_image_ref = container.labels.get(f"{LABEL_PREFIX}runtime-image-ref")
        if not runtime_image_ref:
            continue
        try:
            protected.add(normalize_runtime_image_ref(runtime_image_ref))
        except ValueError:
            continue
    return protected


def repositories_for_runtime_images(protected_refs: set[str]) -> tuple[str, ...]:
    repositories = set(DEFAULT_RUNTIME_IMAGE_REPOSITORIES)
    for runtime_image_ref in protected_refs:
        repository = runtime_image_repository(runtime_image_ref)
        if repository:
            repositories.add(repository)
    return tuple(sorted(repositories))


def prune_stale_runtime_images(
    conn,
    host: DockerRunnerHost,
    *,
    extra_protected_refs: Sequence[str] = (),
) -> int:
    machine = host.machine
    demands = queue_demands(conn)
    containers = host.list_runtime_image_containers()
    protected = protected_runtime_image_refs(
        machine=machine,
        demands=demands,
        containers=containers,
    )
    for runtime_image_ref in extra_protected_refs:
        protected.add(normalize_runtime_image_ref(runtime_image_ref))
    images = host.list_runtime_images(
        repositories_for_runtime_images(protected),
    )
    stale_images = tuple(image for image in images if image.runtime_image_ref not in protected)
    pruned = 0
    for image in stale_images:
        if host.remove_runtime_image(image.image_ref).ok:
            pruned += 1
    return pruned


def prune_inactive_job_containers(conn, host: DockerRunnerHost) -> int:
    machine = host.machine
    active_launch_ids = {
        str(launch["launch_id"]) for launch in active_job_launches(conn, machine=machine.name)
    }
    removed = 0
    for container in host.list_job_containers():
        if container.state not in {"exited", "dead"}:
            continue
        if container.launch_id and container.launch_id in active_launch_ids:
            continue
        if host.remove_container(container.name).ok:
            removed += 1
    return removed


def prewarm_latest_runtime(
    host: DockerRunnerHost,
    *,
    state_path: Path,
    release: Any | None = None,
) -> tuple[dict[str, Any], str]:
    release = release or recent_runtime_images(limit=1)[0]
    runtime_image_ref = release.runtime_image_ref
    expected_state = {
        "runtime_image_ref": runtime_image_ref,
        "runtime_input_sha256": release.runtime_input_sha256,
        "runtime_build_source_sha": release.runtime_build_source_sha or release.source_sha,
    }
    try:
        current_state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError, json.JSONDecodeError, OSError:
        current_state = {}
    present = host.runtime_image_present(runtime_image_ref)
    pulled = False
    if not present:
        if not host.ensure_runtime_image(runtime_image_ref):
            raise RuntimeError(f"failed to pull {runtime_image_ref}")
        pulled = True
    probed = current_state != expected_state or pulled
    if probed:
        host.probe_runtime_image(
            runtime_image_ref=runtime_image_ref,
            expected_source_sha=release.runtime_build_source_sha or release.source_sha,
            expected_runtime_input_sha256=release.runtime_input_sha256,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_name(f".{state_path.name}.tmp")
        temporary.write_text(
            json.dumps(expected_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, state_path)
    return (
        {
            "status": "ready",
            "runtime_image_ref": runtime_image_ref,
            "pulled": pulled,
            "probed": probed,
        },
        runtime_image_ref,
    )


def _load_train_job(conn, job_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM train_jobs WHERE id = %(job_id)s", {"job_id": int(job_id)})
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"train job not found: {job_id}")
    return dict(row)


def _terminal_result(
    launch: Mapping[str, Any],
    *,
    status: str,
    error: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_kind": launch["job_kind"],
        "job_id": int(launch["job_id"]),
        "launch_id": str(launch["launch_id"]),
        "machine": str(launch["machine"]),
        "runtime_image_ref": str(launch["runtime_image_ref"]),
        "status": status,
        "exit_code": exit_code,
        "error": error,
    }


def _record_launch_error(conn, launch_id: str, error: str) -> None:
    record_job_launch_error(conn, launch_id=launch_id, error=error, retry_after_seconds=30.0)


def _start_or_resume_launch(
    conn,
    host: DockerRunnerHost,
    *,
    launch: Mapping[str, Any],
    known_container: JobContainer | None = None,
    image_ready: bool = False,
) -> bool:
    machine = host.machine
    launch_id = str(launch["launch_id"])
    container_name = str(launch["container_name"])
    job = _load_train_job(conn, int(launch["job_id"]))
    try:
        if not image_ready and not host.ensure_runtime_image(str(launch["runtime_image_ref"])):
            _record_launch_error(conn, launch_id, "runtime image is unavailable")
            return False
        telemetry_transport = str(
            job.get("telemetry_transport")
            or (job.get("train_config") or {}).get("telemetry_transport")
            or "legacy_local"
        )
        if (
            telemetry_transport == TRANSPORT_NAME
            and known_container is not None
            and known_container.state != "running"
        ):
            host.remove_container(known_container.name, force=True)
            known_container = None
        attempt_env_path = None
        if telemetry_transport == TRANSPORT_NAME and (
            known_container is None or known_container.state != "running"
        ):
            token = issue_worker_attempt_token(conn, attempt_id=launch_id)
            mailbox_env = load_mailbox_runner_env(
                default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE
            )
            mailbox_env[ATTEMPT_ID_ENV] = launch_id
            mailbox_env[ATTEMPT_TOKEN_ENV] = token
            attempt_env_path = host.write_attempt_env(launch_id, mailbox_env)
        host.write_payload(launch_id, job_payload_for_launch(job, launch))
        container = known_container
        if container is None:
            labels = {
                MANAGED_LABEL: "true",
                JOB_CONTAINER_LABEL: "true",
                MACHINE_LABEL: machine.name,
                JOB_KIND_LABEL: TRAIN_JOB_KIND,
                JOB_ID_LABEL: str(job["id"]),
                LAUNCH_ID_LABEL: launch_id,
                OUTPUT_URI_LABEL: host.output_host_path(launch_id),
                f"{LABEL_PREFIX}runtime-image-ref": str(launch["runtime_image_ref"]),
            }
            created = host.create_train_container(
                launch_id=launch_id,
                container_name=container_name,
                runtime_image_ref=str(launch["runtime_image_ref"]),
                labels=labels,
                attempt_env_path=attempt_env_path,
            )
            containers = {item.launch_id: item for item in host.list_job_containers()}
            container = containers.get(launch_id)
            if container is None:
                _record_launch_error(
                    conn,
                    launch_id,
                    created.detail or "docker create failed",
                )
                return False
        if container.state != "running":
            started = host.start_container(container_name)
            containers = {item.launch_id: item for item in host.list_job_containers()}
            container = containers.get(launch_id)
            if container is None or container.state != "running":
                _record_launch_error(
                    conn,
                    launch_id,
                    started.detail or "docker start failed",
                )
                return False
        mark_job_launch_running(
            conn,
            launch_id=launch_id,
            container_name=container_name,
            provider_run_id=container_name,
        )
        return True
    except Exception as exc:
        _record_launch_error(conn, launch_id, str(exc))
        return False


def launch_cancel_requested(conn, launch: Mapping[str, Any]) -> bool:
    if str(launch.get("job_kind") or "") != "train":
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cancel_requested
            FROM train_jobs
            WHERE id = %(job_id)s
            """,
            {"job_id": launch["job_id"]},
        )
        row = cur.fetchone()
    return bool(row and row.get("cancel_requested"))


def _remove_attempt_env(host: DockerRunnerHost, launch_id: str) -> None:
    remove = getattr(host, "remove_attempt_env", None)
    if callable(remove):
        remove(launch_id)


def cancel_running_job_launch(
    conn,
    host: DockerRunnerHost,
    *,
    launch: Mapping[str, Any],
    container: JobContainer,
) -> bool:
    launch_id = str(launch["launch_id"])
    result = (
        host.remove_container(container.name, force=True)
        if container.state == "created"
        else host.stop_container(container.name, grace_seconds=300)
    )
    if not result.ok:
        _record_launch_error(
            conn,
            launch_id,
            result.detail or "docker stop failed",
        )
        return False
    finish_job_launch_from_result(
        conn,
        launch_id=launch_id,
        result=_terminal_result(
            launch,
            status="canceled",
            exit_code=130,
            error="cancel requested",
        ),
    )
    _remove_attempt_env(host, launch_id)
    return True


def reconcile_machine_launches(conn, host: DockerRunnerHost) -> int:
    machine = host.machine
    launches = active_job_launches(conn, machine=machine.name)
    containers = {container.launch_id: container for container in host.list_job_containers()}
    reconciled = 0
    for launch in launches:
        launch_id = str(launch["launch_id"])
        retry_at = launch.get("next_retry_at")
        if retry_at is not None and retry_at > datetime.now(UTC):
            continue
        container = containers.get(launch_id)
        if container is None:
            observation = host.observe_result(str(launch["output_uri"]))
            if observation.state == "present":
                finish_job_launch_from_result(
                    conn, launch_id=launch_id, result=observation.payload or {}
                )
                _remove_attempt_env(host, launch_id)
                reconciled += 1
            elif observation.state == "error":
                _record_launch_error(
                    conn, launch_id, observation.error or "result observation failed"
                )
            elif launch_cancel_requested(conn, launch):
                finish_job_launch_from_result(
                    conn,
                    launch_id=launch_id,
                    result=_terminal_result(
                        launch,
                        status="canceled",
                        exit_code=130,
                        error="cancel requested; container authoritatively absent",
                    ),
                )
                _remove_attempt_env(host, launch_id)
                reconciled += 1
            elif launch["state"] == "launching":
                if _start_or_resume_launch(conn, host, launch=launch):
                    reconciled += 1
            else:
                finish_job_launch_from_result(
                    conn,
                    launch_id=launch_id,
                    result=_terminal_result(
                        launch,
                        status="failed",
                        exit_code=None,
                        error="running container authoritatively absent without result.json",
                    ),
                )
                _remove_attempt_env(host, launch_id)
                reconciled += 1
            continue
        if container.state in {"created", "restarting", "running"} and launch_cancel_requested(
            conn, launch
        ):
            if cancel_running_job_launch(
                conn,
                host,
                launch=launch,
                container=container,
            ):
                reconciled += 1
                continue
        if container.state == "created":
            if _start_or_resume_launch(conn, host, launch=launch, known_container=container):
                reconciled += 1
            continue
        if container.state == "running":
            if launch["state"] == "launching":
                mark_job_launch_running(
                    conn,
                    launch_id=launch_id,
                    container_name=container.name,
                    provider_run_id=container.name,
                )
                reconciled += 1
            readiness = host.observe_readiness(str(launch["output_uri"]))
            if readiness.state == "present":
                if mark_train_job_ready(
                    conn,
                    launch_id=launch_id,
                    readiness=readiness.payload or {},
                ):
                    reconciled += 1
            elif readiness.state == "error":
                _record_launch_error(
                    conn,
                    launch_id,
                    readiness.error or "readiness observation failed",
                )
            continue
        observation = host.observe_result(str(launch["output_uri"]))
        if observation.state == "present":
            finish_job_launch_from_result(
                conn,
                launch_id=launch_id,
                result=observation.payload or {},
            )
            _remove_attempt_env(host, launch_id)
            reconciled += 1
        elif observation.state == "error":
            _record_launch_error(conn, launch_id, observation.error or "result observation failed")
        else:
            finish_job_launch_from_result(
                conn,
                launch_id=launch_id,
                result=_terminal_result(
                    launch,
                    status="failed",
                    exit_code=None,
                    error=f"container exited without result.json: {container.status}",
                ),
            )
            _remove_attempt_env(host, launch_id)
            reconciled += 1
    return reconciled


def train_container_slot_usage(conn, host: DockerRunnerHost) -> tuple[int, int, int]:
    machine = host.machine
    containers = host.list_job_containers()
    launches = active_job_launches(conn, machine=machine.name)
    reserved = {str(launch["launch_id"]) for launch in launches}
    orphan_count = 0
    for container in containers:
        if container.job_kind != TRAIN_JOB_KIND or container.state not in {
            "running",
            "created",
            "restarting",
        }:
            continue
        if container.launch_id:
            reserved.add(container.launch_id)
        else:
            orphan_count += 1
    control = machine_control(conn, machine=machine.name)
    configured = machine.limits.max_parallel_containers
    requested = control.get("effective_capacity")
    capacity = min(configured, int(requested)) if requested is not None else configured
    used = len(reserved) + orphan_count
    return used, capacity, max(0, capacity - used)


def launch_next_jobs(
    conn,
    *,
    host: DockerRunnerHost,
) -> int:
    machine = host.machine
    launched = 0
    available_images: set[str] = set()
    control = machine_control(conn, machine=machine.name)
    if bool(control.get("drained")):
        return 0
    _used, _capacity, slots = train_container_slot_usage(conn, host)
    for _ in range(slots):
        pending = next_pending_train_job(conn, machine=machine.name)
        if pending is None:
            break
        runtime_image_ref = str(pending["runtime_image_ref"])
        if runtime_image_ref not in available_images:
            if not host.ensure_runtime_image(runtime_image_ref):
                break
            available_images.add(runtime_image_ref)
        job_id = int(pending["id"])
        launch_id = new_train_launch_id(job_id)
        container_name = job_container_name(machine, launch_id=launch_id)
        claimed = claim_job_launch(
            conn,
            machine=machine.name,
            backend=machine.backend,
            job_id=job_id,
            launch_id=launch_id,
            container_name=container_name,
            output_uri=host.output_host_path(launch_id),
        )
        if claimed is None:
            break
        _job, launch = claimed
        if _start_or_resume_launch(conn, host, launch=launch, image_ready=True):
            launched += 1
    return launched


def run_reconcile_fill_pass(
    conn,
    *,
    host: DockerRunnerHost,
    shared_env_file: Path | None = None,
) -> tuple[int, int]:
    machine = host.machine
    pending = next_pending_train_job(conn, machine=machine.name)
    launching = active_job_launches(conn, machine=machine.name, states=("launching",))
    if machine.backend == "docker_ssh" and (pending is not None or launching):
        host.sync_shared_env(
            load_shared_runner_env(
                shared_env_file or (default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE)
            )
        )
    reconciled = reconcile_machine_launches(conn, host)
    launched = launch_next_jobs(conn, host=host)
    return reconciled, launched


def recover_live_publication(conn, *, host: DockerRunnerHost) -> int:
    """Run one CPU-only durable-outbox recovery without taking a train slot."""

    recovery = claim_live_publication_recovery(conn, machine=host.machine.name)
    if recovery is None:
        return 0
    error = None
    try:
        run_wandb_publisher_recovery_container(
            host,
            launch_id=str(recovery["launch_id"]),
            run_name=str(recovery.get("run_name") or f"train_job_{recovery['id']}"),
            runtime_image_ref=str(recovery["runtime_image_ref"]),
        )
    except Exception as exc:
        error = repr(exc)
    finish_live_publication_recovery(
        conn,
        job_id=int(recovery["id"]),
        error=error,
    )
    return int(error is None)


@contextmanager
def machine_mutation_lock(conn, machine_name: str):
    lock = acquire_machine_lock(conn, machine_name)
    try:
        yield lock
    finally:
        release_machine_lock(conn, lock)


def machine_lock_key(machine_name: str) -> str:
    return f"rlab-fleet-reconciler:{machine_name}"


def acquire_machine_lock(conn, machine_name: str) -> MachineMutationLock:
    key = machine_lock_key(machine_name)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": key},
        )
        row = cur.fetchone()
    if not row or not row.get("acquired"):
        raise MachineLockBusy(machine_name)
    return MachineMutationLock(machine=machine_name, key=key)


def release_machine_lock(conn, lock: MachineMutationLock) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0)) AS released",
            {"key": lock.key},
        )


def _maintenance_due(path: Path, *, interval_seconds: float = 3600.0) -> bool:
    try:
        return time.time() - path.stat().st_mtime >= interval_seconds
    except FileNotFoundError:
        return True


def run_service_machine_pass(
    *,
    machine_name: str,
    machines_path: Path,
    repo_root: Path,
    deadline_monotonic: float,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"machine lane deadline expired before start: {machine_name}")
    machine = resolve_machine(load_machine_registry(machines_path), machine_name)
    host = DockerRunnerHost(machine, deadline_monotonic=deadline_monotonic)
    conn = None
    try:
        if progress:
            progress(
                "CHECKING MACHINE", f"Connecting to {machine.name} and acquiring its scheduler lock"
            )
        # Session-scoped advisory locks must bypass PgBouncer. Otherwise a
        # terminated Fleet process can return a still-locked backend session
        # to the pool and block an unrelated client indefinitely.
        conn = connect(database_url(use_direct=True))
        with machine_mutation_lock(conn, machine.name):
            if progress:
                progress(
                    "RECONCILING TRAIN", "Observing active containers and filling available slots"
                )
            reconciled, launched = run_reconcile_fill_pass(
                conn,
                host=host,
                shared_env_file=repo_root / DEFAULT_SHARED_RUNNER_ENV_FILE,
            )
            if progress:
                progress(
                    "FINALIZING PUBLICATION", "Recovering any durable train publication outbox"
                )
            recovered_publications = recover_live_publication(conn, host=host)
            prewarm: dict[str, Any] = {"status": "disabled"}
            prewarmed_ref = ""
            if machine.prewarm_latest_runtime:
                try:
                    if progress:
                        progress(
                            "PREWARMING RUNTIME",
                            "Pulling and probing the latest demanded runtime image",
                        )
                    release = recent_runtime_images(limit=1)[0]
                    prewarmed_ref = release.runtime_image_ref
                    prewarm, prewarmed_ref = prewarm_latest_runtime(
                        host,
                        state_path=(repo_root / "logs" / "fleet" / f"prewarm-{machine.name}.json"),
                        release=release,
                    )
                except Exception as exc:
                    prewarm = {"status": "error", "error": str(exc)}
            if time.monotonic() >= deadline_monotonic:
                return {
                    "reconciled": reconciled,
                    "launched": launched,
                    "recovered_publications": recovered_publications,
                    "removed_containers": 0,
                    "pruned_images": 0,
                    "prewarm": prewarm,
                    "maintenance_skipped": "machine lane deadline expired after reconciliation",
                }
            maintenance_marker = repo_root / "logs" / "fleet" / f"maintenance-{machine.name}.stamp"
            pruned = 0
            removed_containers = 0
            if reconciled or launched or _maintenance_due(maintenance_marker):
                if progress:
                    progress(
                        "REMOVING STALE RESOURCES",
                        "Pruning inactive containers and unused runtime images safely",
                    )
                removed_containers = prune_inactive_job_containers(conn, host)
                pruned = prune_stale_runtime_images(
                    conn,
                    host,
                    extra_protected_refs=(prewarmed_ref,) if prewarmed_ref else (),
                )
                maintenance_marker.parent.mkdir(parents=True, exist_ok=True)
                maintenance_marker.touch()
            return {
                "reconciled": reconciled,
                "launched": launched,
                "recovered_publications": recovered_publications,
                "removed_containers": removed_containers,
                "pruned_images": pruned,
                "prewarm": prewarm,
            }
    finally:
        if conn is not None:
            conn.close()


def _kick_after_machine_control(reason: str, machine: str) -> str:
    from rlab.fleet_service import kick_service

    try:
        return (
            "kicked"
            if kick_service(reason=reason, entity_kind="machine", entity_id=machine)
            else "degraded"
        )
    except Exception:
        return "degraded"


def cmd_drain(args: argparse.Namespace) -> int:
    resolve_machine(load_registry_from_args(args), args.machine)
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(
            conn,
            machine=args.machine,
            drained=True,
            reason=args.reason or "operator drain",
        )
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "control": json_safe(control),
                "dispatch": _kick_after_machine_control("machine_drain", args.machine),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    resolve_machine(load_registry_from_args(args), args.machine)
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(conn, machine=args.machine, drained=False, reason="resumed")
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "control": json_safe(control),
                "dispatch": _kick_after_machine_control("machine_resume", args.machine),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    if args.capacity is not None and args.capacity > machine.limits.max_parallel_containers:
        raise SystemExit(
            f"capacity {args.capacity} exceeds configured maximum "
            f"{machine.limits.max_parallel_containers} for {machine.name}"
        )
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(
            conn,
            machine=machine.name,
            effective_capacity=args.capacity,
            reset_capacity=bool(args.reset),
            reason="capacity reset" if args.reset else f"capacity set to {args.capacity}",
        )
    finally:
        conn.close()
    print(
        json.dumps(
            {
                "control": json_safe(control),
                "dispatch": _kick_after_machine_control("machine_capacity", machine.name),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_setup_host(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.host)
    runtime_image_ref = runtime_image_ref_from_args(args)
    script, _ = setup_docker_host(
        machine,
        runtime_image_ref,
        execute=False,
    )
    print(f"host: {machine.name}")
    print(script.rstrip())
    if not args.execute:
        print("dry_run: rerun without --dry-run to run setup over SSH")
        return 0
    _, returncode = setup_docker_host(
        machine,
        runtime_image_ref,
        execute=True,
    )
    return int(returncode or 0)


def add_machine_registry_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)


def add_database_arg(parser: argparse.ArgumentParser) -> None:
    add_direct_database_arg(parser)


def add_runtime_image_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-image-ref")
    group.add_argument("--runtime-image-ref-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab fleet",
        description="Manage one-job rlab containers from queue state.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    drain = subparsers.add_parser("drain", help="Block new claims for one machine.")
    add_machine_registry_arg(drain)
    add_database_arg(drain)
    drain.add_argument("--machine", required=True)
    drain.add_argument("--reason")
    drain.set_defaults(func=cmd_drain)

    resume = subparsers.add_parser("resume", help="Allow new claims for one machine.")
    add_machine_registry_arg(resume)
    add_database_arg(resume)
    resume.add_argument("--machine", required=True)
    resume.set_defaults(func=cmd_resume)

    capacity = subparsers.add_parser("capacity", help="Set temporary effective machine capacity.")
    add_machine_registry_arg(capacity)
    add_database_arg(capacity)
    capacity.add_argument("--machine", required=True)
    capacity_action = capacity.add_mutually_exclusive_group(required=True)
    capacity_action.add_argument("--set", dest="capacity", type=int)
    capacity_action.add_argument("--reset", action="store_true")
    capacity.set_defaults(func=cmd_capacity)

    setup = subparsers.add_parser("setup-host", help="Prepare SSH Docker hosts for job containers.")
    add_machine_registry_arg(setup)
    setup.add_argument("--host", required=True, help="Fleet host to set up.")
    add_dry_run_arg(setup)
    add_runtime_image_args(setup)
    setup.set_defaults(func=cmd_setup_host)

    from rlab.fleet_service import add_service_parser

    add_service_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
