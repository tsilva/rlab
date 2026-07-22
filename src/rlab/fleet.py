from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
    finalize_machine_drain,
    job_payload_for_launch,
    machine_control,
    mark_job_launch_running,
    mark_train_job_ready,
    new_train_launch_id,
    next_pending_train_job,
    queue_demands,
    record_job_launch_error,
    record_runtime_image_failure,
    record_machine_drain_zero_receipt,
    reset_runtime_image_retry,
    runtime_image_retry_state,
    set_machine_control,
)
from rlab.telemetry_mailbox import (
    ATTEMPT_ID_ENV,
    ATTEMPT_TOKEN_ENV,
    MAILBOX_DATABASE_ENV,
    TRANSPORT_NAME,
    issue_worker_attempt_token,
)
from rlab.workspace_contract import (
    build_workspace_manifest,
    cleanup_batch_from_mapping,
    workspace_manifest_from_mapping,
)
from rlab.workspace_gc import (
    WorkspaceNotReady,
    attest_launch_workspace_layout,
    authorize_cleanup_batch,
    authorize_workspace_cleanup_canary,
    authorize_workspace_enqueue,
    begin_recovery_container_generation,
    claim_cleanup_rows,
    complete_workspace_qualification_schedule,
    completed_journal_cleanup_due,
    complete_recovery_env_reservation,
    create_telemetry_obligation,
    create_workspace_qualification_schedule,
    drain_inventory_request,
    due_proof_reduction_launches,
    due_signed_cleanup_batches,
    finalize_host_deleted,
    finish_host_operation_lease,
    load_workspace_manifest,
    mark_container_generation_released,
    mark_container_member_state,
    mark_env_unlinked,
    make_terminal_env_eligible,
    mark_journal_cleaned,
    mark_prepare_receipt_cleaned,
    mark_recovery_env_intent_cleaned,
    mark_reservation_intent_cleaned,
    mark_reservation_recovered_for_retry,
    persist_reservation_receipt,
    persist_workspace_manifest,
    mark_terminal_credential_cleanup_failure,
    mark_terminal_member_absent,
    manifests_for_cleanup_batch,
    record_cleanup_delivery_intent,
    record_delete_prepare,
    reconcile_cleanup_result,
    reduce_cleanup_proof,
    record_workspace_qualification_receipt,
    record_workspace_promotion_receipt,
    record_proof_reducer_result,
    register_host_operation_lease,
    resolve_rollback_review,
    register_container_generation,
    workspace_protocol_mode,
    workspace_gc_status,
    set_machine_cleanup_control,
    set_workspace_rollout_control,
    terminal_credential_cleanup_targets,
)
from rlab.workspace_helper import HELPER_REVISION
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
from rlab.rom_assets import manifest_from_train_config
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
        if container.state in {"exited", "removing", "dead"}:
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
    failures: list[str] = []
    for image in stale_images:
        result = host.remove_runtime_image(image.image_ref)
        if result.ok:
            pruned += 1
        else:
            detail = str(result.detail or "remove failed").strip()
            failures.append(f"{image.image_ref}: {detail}")
    if failures:
        raise RuntimeError(
            f"failed to prune stale runtime images on {machine.name}: {'; '.join(failures)}"
        )
    return pruned


def prune_inactive_job_containers(conn, host: DockerRunnerHost) -> int:
    machine = host.machine
    active_launch_ids = {
        str(launch["launch_id"]) for launch in active_job_launches(conn, machine=machine.name)
    }
    containers = {container.name: container for container in host.list_job_containers()}
    for container in host.list_runtime_image_containers():
        if container.labels.get(MANAGED_LABEL) == "true":
            containers.setdefault(container.name, container)
    removed = 0
    for container in containers.values():
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


@contextmanager
def host_operation_lease_guard(
    conn,
    *,
    machine: str,
    operation_kind: str,
    resource_scope: str,
    launch_id: str | None = None,
    lifecycle_generation: int | None = None,
):
    operation_id = register_host_operation_lease(
        conn,
        machine=machine,
        operation_kind=operation_kind,
        resource_scope=resource_scope,
        source_revision=HELPER_REVISION,
        owner=f"fleet:{os.getpid()}",
        max_duration=timedelta(minutes=16),
        launch_id=launch_id,
        lifecycle_generation=lifecycle_generation,
    )
    try:
        yield operation_id
    except Exception as exc:
        finish_host_operation_lease(
            conn, operation_id=operation_id, error=str(exc), evidence={}
        )
        raise
    else:
        finish_host_operation_lease(
            conn,
            operation_id=operation_id,
            evidence={"completed_at": datetime.now(UTC).isoformat()},
        )


def _start_or_resume_launch_unleased(
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
            record_runtime_image_failure(
                conn,
                machine=machine.name,
                runtime_image_ref=str(launch["runtime_image_ref"]),
                error="runtime image is unavailable",
            )
            _record_launch_error(conn, launch_id, "runtime image is unavailable")
            return False
        reset_runtime_image_retry(
            conn,
            machine=machine.name,
            runtime_image_ref=str(launch["runtime_image_ref"]),
        )
        telemetry_transport = str(
            job.get("telemetry_transport")
            or (job.get("train_config") or {}).get("telemetry_transport")
            or "legacy_local"
        )
        layout_v1 = workspace_protocol_mode(conn) != "dormant"
        if layout_v1:
            generation = int(launch.get("lifecycle_generation") or 1)
            payload_document = job_payload_for_launch(job, launch)
            payload = (
                json.dumps(json_safe(payload_document), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            stored = load_workspace_manifest(
                conn, launch_id=launch_id, generation=generation
            )
            if stored is None:
                if known_container is not None:
                    raise RuntimeError(
                        "cannot adopt an existing container into workspace layout v1"
                    )
                manifest = build_workspace_manifest(
                    machine=machine.name,
                    launch_id=launch_id,
                    generation=generation,
                    reservation_nonce=secrets.token_urlsafe(24),
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    host_root=machine.paths.host_root,
                    payloads_root=machine.paths.payloads_dir,
                    outputs_root=machine.paths.outputs_dir,
                    protected_metadata_root=machine.paths.protected_metadata_dir,
                    logs_root=machine.paths.logs_dir,
                    shared_env_path=machine.paths.env_file,
                    rom_cache_root=machine.paths.rom_cache_dir,
                    container_payloads_root=machine.paths.container_payloads_dir,
                    container_outputs_root=machine.paths.container_outputs_dir,
                )
                persist_workspace_manifest(conn, manifest=manifest)
                stored = (
                    manifest,
                    {
                        "reservation_receipt": None,
                        "reservation_intent_state": "pending",
                    },
                )
            manifest, manifest_row = stored
            receipt = dict(manifest_row.get("reservation_receipt") or {})
            if receipt and str(manifest_row.get("reservation_intent_state")) != "cleaned":
                host.release_reservation_intent(
                    manifest, receipt_sha256=str(receipt["receipt_sha256"])
                )
                mark_reservation_intent_cleaned(
                    conn,
                    launch_id=launch_id,
                    generation=generation,
                    receipt_sha256=str(receipt["receipt_sha256"]),
                )
            if not receipt:
                if str(manifest_row.get("reservation_intent_state") or "pending") != "pending":
                    recovery = host.run_workspace_helper(
                        "recover-intent", {"manifest": manifest.as_dict()}
                    )
                    mark_reservation_recovered_for_retry(
                        conn,
                        launch_id=launch_id,
                        generation=generation,
                        recovery=recovery,
                    )
                if telemetry_transport == TRANSPORT_NAME:
                    token = issue_worker_attempt_token(conn, attempt_id=launch_id)
                    attempt_env = load_mailbox_runner_env(
                        default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE
                    )
                    attempt_env[ATTEMPT_ID_ENV] = launch_id
                    attempt_env[ATTEMPT_TOKEN_ENV] = token
                else:
                    attempt_env = load_shared_runner_env(
                        default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE
                    )
                receipt = host.reserve_workspace(
                    manifest, payload=payload, attempt_env=attempt_env
                )
                persist_reservation_receipt(
                    conn, launch_id=launch_id, generation=generation, receipt=receipt
                )
                host.release_reservation_intent(
                    manifest, receipt_sha256=str(receipt["receipt_sha256"])
                )
                mark_reservation_intent_cleaned(
                    conn,
                    launch_id=launch_id,
                    generation=generation,
                    receipt_sha256=str(receipt["receipt_sha256"]),
                )
            if not receipt:
                raise RuntimeError("workspace reservation receipt is not durable")
            members = []
            train_config = dict(job.get("train_config") or {})
            rom_asset_manifest = manifest_from_train_config(
                train_config,
                expected_game=str(train_config.get("game") or ""),
            )
            if rom_asset_manifest is not None:
                members.append(
                    {
                        "member_kind": "rom",
                        "container_name": f"rlab-rom-{launch_id}",
                        "runtime_image_ref": str(launch["runtime_image_ref"]),
                    }
                )
            members.append(
                {
                    "member_kind": "train",
                    "container_name": container_name,
                    "runtime_image_ref": str(launch["runtime_image_ref"]),
                }
            )
            register_container_generation(
                conn,
                launch_id=launch_id,
                workspace_generation=generation,
                container_generation=1,
                purpose="startup",
                env_path=manifest.env_path,
                env_identity=receipt["leaves"]["env"],
                env_fingerprint_sha256=str(receipt["env_fingerprint_sha256"]),
                members=members,
            )
            create_telemetry_obligation(
                conn,
                train_job_id=int(job["id"]),
                launch_id=launch_id,
                lifecycle_generation=generation,
                producer_identity=f"train:{launch_id}:1",
                sink=("wandb" if bool(train_config.get("wandb")) else "disabled"),
                configuration_revision=hashlib.sha256(
                    json.dumps(train_config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                expected_stream_id=(launch_id if telemetry_transport == TRANSPORT_NAME else None),
            )
            attempt_env_path = manifest.env_path
        else:
            generation = 1
        if (
            not layout_v1
            and
            telemetry_transport == TRANSPORT_NAME
            and known_container is not None
            and known_container.state != "running"
        ):
            host.remove_container(known_container.name, force=True)
            known_container = None
        attempt_env_path = attempt_env_path if layout_v1 else None
        if not layout_v1 and telemetry_transport == TRANSPORT_NAME and (
            known_container is None or known_container.state != "running"
        ):
            token = issue_worker_attempt_token(conn, attempt_id=launch_id)
            mailbox_env = load_mailbox_runner_env(
                default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE
            )
            mailbox_env[ATTEMPT_ID_ENV] = launch_id
            mailbox_env[ATTEMPT_TOKEN_ENV] = token
            attempt_env_path = host.write_attempt_env(launch_id, mailbox_env)
        if not layout_v1:
            host.write_payload(launch_id, job_payload_for_launch(job, launch))
        train_config = dict(job.get("train_config") or {})
        rom_asset_manifest = manifest_from_train_config(
            train_config,
            expected_game=str(train_config.get("game") or ""),
        )
        if rom_asset_manifest is not None and (
            known_container is None or known_container.state != "running"
        ):
            cached = host.ensure_rom_cache(
                launch_id=launch_id,
                runtime_image_ref=str(launch["runtime_image_ref"]),
                attempt_env_path=attempt_env_path,
                rom_asset_manifest=rom_asset_manifest,
                container_name=f"rlab-rom-{launch_id}",
            )
            if not cached.ok:
                _record_launch_error(
                    conn,
                    launch_id,
                    cached.detail or "ROM cache staging failed",
                )
                return False
            if layout_v1:
                digest = str(rom_asset_manifest["sha256"])
                mark_container_member_state(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    container_generation=1,
                    member_kind="rom",
                    state="absent",
                    mount_attestation={
                        "container_name": f"rlab-rom-{launch_id}",
                        "payload_source": manifest.payload_path,
                        "payload_destination": manifest.container_payload_path,
                        "cache_source": (
                            f"{machine.paths.rom_cache_dir.rstrip('/')}/sha256/{digest}"
                        ),
                        "cache_destination": (
                            f"{machine.paths.container_rom_cache_dir.rstrip('/')}/sha256/"
                            f"{digest}"
                        ),
                        "container_removed": True,
                    },
                )
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
                rom_asset_manifest=rom_asset_manifest,
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
            if layout_v1:
                attestation = host.attest_train_container(
                    container_name=container_name,
                    launch_id=launch_id,
                    runtime_image_ref=str(launch["runtime_image_ref"]),
                )
                mark_container_member_state(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    container_generation=1,
                    member_kind="train",
                    state="attested_unreleased",
                    container_id=str(attestation["container_id"]),
                    mount_attestation=attestation,
                )
        elif layout_v1:
            attestation = host.attest_train_container(
                container_name=container_name,
                launch_id=launch_id,
                runtime_image_ref=str(launch["runtime_image_ref"]),
            )
            mark_container_member_state(
                conn,
                launch_id=launch_id,
                workspace_generation=generation,
                container_generation=1,
                member_kind="train",
                state="running" if container.state == "running" else "attested_unreleased",
                container_id=str(attestation["container_id"]),
                mount_attestation=attestation,
            )
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
        if layout_v1:
            mark_container_member_state(
                conn,
                launch_id=launch_id,
                workspace_generation=generation,
                container_generation=1,
                member_kind="train",
                state="running",
            )
            mark_container_generation_released(
                conn,
                launch_id=launch_id,
                workspace_generation=generation,
                container_generation=1,
            )
            attest_launch_workspace_layout(
                conn,
                launch_id=launch_id,
                generation=generation,
                container_generation=1,
            )
            if attempt_env_path is not None:
                env_receipt = host.unlink_reserved_attempt_env(
                    manifest, expected_identity=receipt["leaves"]["env"]
                )
                mark_env_unlinked(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    container_generation=1,
                    receipt=env_receipt,
                )
        return True
    except Exception as exc:
        _record_launch_error(conn, launch_id, str(exc))
        return False


def _start_or_resume_launch(
    conn,
    host: DockerRunnerHost,
    *,
    launch: Mapping[str, Any],
    known_container: JobContainer | None = None,
    image_ready: bool = False,
) -> bool:
    if workspace_protocol_mode(conn) == "dormant":
        return _start_or_resume_launch_unleased(
            conn,
            host,
            launch=launch,
            known_container=known_container,
            image_ready=image_ready,
        )
    generation = int(launch.get("lifecycle_generation") or 1)
    with host_operation_lease_guard(
        conn,
        machine=host.machine.name,
        operation_kind="train_start_or_resume",
        resource_scope=f"launch:{launch['launch_id']}",
        launch_id=str(launch["launch_id"]),
        lifecycle_generation=generation,
    ):
        return _start_or_resume_launch_unleased(
            conn,
            host,
            launch=launch,
            known_container=known_container,
            image_ready=image_ready,
        )


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


def cleanup_terminal_credentials(conn, host: DockerRunnerHost, *, limit: int = 32) -> int:
    if workspace_protocol_mode(conn) == "dormant":
        return 0
    cleaned = 0
    for target in terminal_credential_cleanup_targets(
        conn, machine=host.machine.name, limit=limit
    ):
        launch_id = str(target["launch_id"])
        workspace_generation = int(target["workspace_generation"])
        container_generation = int(target["container_generation"])
        try:
            for member in target["members"]:
                if str(member["state"]) == "absent":
                    continue
                container_name = str(member["container_name"])
                if not host.container_is_absent(container_name):
                    result = host.remove_container(container_name, force=True)
                    if not result.ok and not host.container_is_absent(container_name):
                        raise RuntimeError(
                            result.detail or f"failed to remove {container_name}"
                        )
                if not host.container_is_absent(container_name):
                    raise RuntimeError(f"container still present after removal: {container_name}")
                mark_terminal_member_absent(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=workspace_generation,
                    container_generation=container_generation,
                    member_kind=str(member["member_kind"]),
                    ordinal=int(member["ordinal"]),
                    receipt={
                        "container_name": container_name,
                        "container_id": member.get("container_id"),
                        "verified_absent_at": datetime.now(UTC).isoformat(),
                        "machine": host.machine.name,
                    },
                )
            make_terminal_env_eligible(
                conn,
                launch_id=launch_id,
                workspace_generation=workspace_generation,
                container_generation=container_generation,
            )
            if str(target["env_cleanup_state"]) != "unlinked":
                manifest = workspace_manifest_from_mapping(target["manifest_json"])
                receipt = host.unlink_reserved_attempt_env(
                    manifest,
                    expected_identity={
                        "device": int(target["env_device"]),
                        "inode": int(target["env_inode"]),
                    },
                )
                mark_env_unlinked(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=workspace_generation,
                    container_generation=container_generation,
                    receipt=receipt,
                )
            if str(target.get("env_intent_cleanup_state") or "not_created") not in {
                "not_created",
                "cleaned",
            }:
                reservation = dict(target.get("env_reservation_receipt") or {})
                host.release_recovery_env_intent(
                    manifest,
                    container_generation=container_generation,
                    receipt_sha256=str(reservation["receipt_sha256"]),
                )
                mark_recovery_env_intent_cleaned(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=workspace_generation,
                    container_generation=container_generation,
                )
            cleaned += 1
        except Exception as exc:
            mark_terminal_credential_cleanup_failure(
                conn,
                launch_id=launch_id,
                workspace_generation=workspace_generation,
                container_generation=container_generation,
                error=str(exc),
            )
    return cleaned


def try_finalize_requested_drain(conn, host: DockerRunnerHost) -> bool:
    request = drain_inventory_request(conn, machine=host.machine.name)
    if request is None:
        return False
    manifests = tuple(
        workspace_manifest_from_mapping(value) for value in request["manifests"]
    )
    receipt = host.workspace_drain_zero_receipt(
        manifests=manifests,
        exact_container_names=request["container_names"],
        exact_protected_paths=request["protected_paths"],
        receipt_nonce=secrets.token_urlsafe(24),
    )
    if any(int(value) for value in dict(receipt.get("counts") or {}).values()):
        return False
    record_machine_drain_zero_receipt(
        conn,
        machine=host.machine.name,
        control_revision=int(request["control_revision"]),
        receipt=receipt,
    )
    finalize_machine_drain(
        conn,
        machine=host.machine.name,
        control_revision=int(request["control_revision"]),
    )
    return True


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
    unavailable_images: set[str] = set()
    control = machine_control(conn, machine=machine.name)
    if bool(control.get("drained")) or bool(control.get("drain_requested")) or bool(
        control.get("normal_admission_disabled")
    ):
        return 0
    _used, _capacity, slots = train_container_slot_usage(conn, host)
    while launched < slots:
        pending = next_pending_train_job(
            conn,
            machine=machine.name,
            exclude_runtime_image_refs=tuple(sorted(unavailable_images)),
        )
        if pending is None:
            break
        runtime_image_ref = str(pending["runtime_image_ref"])
        if runtime_image_ref in unavailable_images:
            break
        if runtime_image_ref not in available_images:
            if not host.ensure_runtime_image(runtime_image_ref):
                record_runtime_image_failure(
                    conn,
                    machine=machine.name,
                    runtime_image_ref=runtime_image_ref,
                    error="runtime image is unavailable",
                )
                unavailable_images.add(runtime_image_ref)
                continue
            reset_runtime_image_retry(
                conn,
                machine=machine.name,
                runtime_image_ref=runtime_image_ref,
            )
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


def run_workspace_gc_pass(conn, *, host: DockerRunnerHost) -> dict[str, int]:
    reduced = 0
    not_ready = 0
    for launch_id in due_proof_reduction_launches(conn, limit=8):
        try:
            reduce_cleanup_proof(conn, launch_id=launch_id)
            record_proof_reducer_result(conn, launch_id=launch_id, ready=True)
            reduced += 1
        except WorkspaceNotReady as exc:
            record_proof_reducer_result(
                conn, launch_id=launch_id, ready=False, error=str(exc)
            )
            not_ready += 1

    prepared_rows: list[dict[str, Any]] = []
    for row in claim_cleanup_rows(conn, machine=host.machine.name, limit=8):
        stored = load_workspace_manifest(
            conn,
            launch_id=str(row["launch_id"]),
            generation=int(row["lifecycle_generation"]),
        )
        if stored is None:
            raise RuntimeError("claimed cleanup row has no durable manifest")
        manifest, _ = stored
        with host_operation_lease_guard(
            conn,
            machine=host.machine.name,
            operation_kind="workspace_delete_prepare",
            resource_scope=f"workspace:{row['launch_id']}",
            launch_id=str(row["launch_id"]),
            lifecycle_generation=int(row["lifecycle_generation"]),
        ):
            receipt = host.delete_prepare(
                manifest,
                cleanup_row_id=int(row["id"]),
                cleanup_attempt_id=str(row["cleanup_attempt_id"]),
                control_revision=int(row["control_revision"]),
                cursor_sha256=str(row["cursor_sha256"]),
            )
        prepared_rows.append(
            record_delete_prepare(
                conn,
                cleanup_row_id=int(row["id"]),
                cleanup_attempt_id=str(row["cleanup_attempt_id"]),
                receipt=receipt,
            )
        )
    authorized = 0
    if prepared_rows:
        authorize_cleanup_batch(
            conn,
            machine=host.machine.name,
            cleanup_row_ids=[int(row["id"]) for row in prepared_rows],
            helper_revision=HELPER_REVISION,
            key_revision=str(os.environ.get("RLAB_WORKSPACE_KEY_REVISION") or "key-v1"),
        )
        authorized = len(prepared_rows)

    deleted = 0
    ambiguous = 0
    for batch in due_signed_cleanup_batches(conn, machine=host.machine.name, limit=1):
        batch_id = str(batch["batch_id"])
        envelope_json, signature = record_cleanup_delivery_intent(conn, batch_id=batch_id)
        envelope = cleanup_batch_from_mapping(envelope_json)
        manifests = manifests_for_cleanup_batch(conn, batch_id=batch_id)
        try:
            with host_operation_lease_guard(
                conn,
                machine=host.machine.name,
                operation_kind="workspace_delete_commit",
                resource_scope=f"cleanup-batch:{batch_id}",
            ):
                result = host.delete_commit(
                    envelope,
                    signature=signature,
                    manifests=manifests,
                )
        except Exception:
            reconcile_cleanup_result(conn, batch_id=batch_id, result=None, ambiguous=True)
            ambiguous += 1
            continue
        reconcile_cleanup_result(conn, batch_id=batch_id, result=result)
        deleted += len(result.get("completed") or [])
        for row in envelope.rows:
            manifest = manifests[row.row_id]
            with host_operation_lease_guard(
                conn,
                machine=host.machine.name,
                operation_kind="workspace_prepare_journal_cleanup",
                resource_scope=f"workspace:{manifest.launch_id}",
                launch_id=manifest.launch_id,
                lifecycle_generation=manifest.generation,
            ):
                host.remove_prepare_receipt(
                    manifest,
                    cleanup_attempt_id=row.cleanup_attempt_id,
                    prepare_receipt_sha256=row.prepare_receipt_sha256,
                )
            mark_prepare_receipt_cleaned(
                conn,
                cleanup_row_id=row.row_id,
                prepare_receipt_sha256=row.prepare_receipt_sha256,
            )

    finalized = finalize_host_deleted(conn, limit=100)
    journals_cleaned = 0
    for row in completed_journal_cleanup_due(conn, machine=host.machine.name, limit=8):
        manifest = workspace_manifest_from_mapping(row["manifest_json"])
        with host_operation_lease_guard(
            conn,
            machine=host.machine.name,
            operation_kind="workspace_host_deleted_journal_cleanup",
            resource_scope=f"workspace:{manifest.launch_id}",
            launch_id=manifest.launch_id,
            lifecycle_generation=manifest.generation,
        ):
            host.remove_host_deleted_journal(
                manifest,
                host_deleted_journal_sha256=str(row["host_deleted_journal_sha256"]),
            )
        mark_journal_cleaned(
            conn,
            cleanup_row_id=int(row["id"]),
            host_deleted_journal_sha256=str(row["host_deleted_journal_sha256"]),
        )
        journals_cleaned += 1
    return {
        "proofs_reduced": reduced,
        "proofs_not_ready": not_ready,
        "rows_prepared": len(prepared_rows),
        "rows_authorized": authorized,
        "rows_deleted": deleted,
        "ambiguous_batches": ambiguous,
        "rows_finalized": finalized,
        "journals_cleaned": journals_cleaned,
    }


def recover_live_publication(conn, *, host: DockerRunnerHost) -> int:
    """Run one CPU-only durable-outbox recovery without taking a train slot."""

    recovery = claim_live_publication_recovery(conn, machine=host.machine.name)
    if recovery is None:
        return 0
    error = None
    try:
        if workspace_protocol_mode(conn) == "dormant":
            run_wandb_publisher_recovery_container(
                host,
                launch_id=str(recovery["launch_id"]),
                run_name=str(recovery.get("run_name") or f"train_job_{recovery['id']}"),
                runtime_image_ref=str(recovery["runtime_image_ref"]),
            )
        else:
            with host_operation_lease_guard(
                conn,
                machine=host.machine.name,
                operation_kind="wandb_publication_recovery",
                resource_scope=f"workspace:{recovery['launch_id']}",
            ):
                launch_id = str(recovery["launch_id"])
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT lifecycle_generation FROM job_launches
                        WHERE launch_id=%(launch_id)s
                        """,
                        {"launch_id": launch_id},
                    )
                    launch = cur.fetchone()
                if not launch:
                    raise RuntimeError("W&B recovery launch manifest is absent")
                generation = int(launch["lifecycle_generation"])
                stored = load_workspace_manifest(
                    conn, launch_id=launch_id, generation=generation
                )
                if stored is None:
                    raise RuntimeError("W&B recovery requires a layout-v1 manifest")
                manifest, _manifest_row = stored
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(MAX(container_generation), 1) + 1 AS next_generation
                        FROM workspace_container_generations
                        WHERE launch_id=%(launch_id)s
                          AND workspace_generation=%(generation)s
                        """,
                        {"launch_id": launch_id, "generation": generation},
                    )
                    proposed_generation = max(2, int(cur.fetchone()["next_generation"]))
                container_name = (
                    f"rlab-wandb-recovery-{launch_id}-g{proposed_generation}"
                )
                container_generation = begin_recovery_container_generation(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    purpose="wandb_recovery",
                    env_path=manifest.env_path,
                    member_kind="wandb",
                    container_name=container_name,
                    runtime_image_ref=str(recovery["runtime_image_ref"]),
                )
                if container_generation != proposed_generation:
                    raise RuntimeError("W&B recovery generation allocation changed")
                recovery_env = load_shared_runner_env(
                    default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE
                )
                env_reservation = host.reserve_recovery_env(
                    manifest,
                    container_generation=container_generation,
                    attempt_env=recovery_env,
                )
                complete_recovery_env_reservation(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    container_generation=container_generation,
                    receipt=env_reservation,
                )
                host.release_recovery_env_intent(
                    manifest,
                    container_generation=container_generation,
                    receipt_sha256=str(env_reservation["receipt_sha256"]),
                )
                mark_recovery_env_intent_cleaned(
                    conn,
                    launch_id=launch_id,
                    workspace_generation=generation,
                    container_generation=container_generation,
                )

                def attested(receipt: Mapping[str, Any]) -> None:
                    mark_container_member_state(
                        conn,
                        launch_id=launch_id,
                        workspace_generation=generation,
                        container_generation=container_generation,
                        member_kind="wandb",
                        state="attested_unreleased",
                        container_id=str(receipt["container_id"]),
                        mount_attestation=receipt,
                    )
                    mark_container_generation_released(
                        conn,
                        launch_id=launch_id,
                        workspace_generation=generation,
                        container_generation=container_generation,
                    )
                    env_receipt = host.unlink_reserved_attempt_env(
                        manifest,
                        expected_identity=env_reservation["env_identity"],
                    )
                    mark_env_unlinked(
                        conn,
                        launch_id=launch_id,
                        workspace_generation=generation,
                        container_generation=container_generation,
                        receipt=env_receipt,
                    )

                def absent(receipt: Mapping[str, Any]) -> None:
                    mark_terminal_member_absent(
                        conn,
                        launch_id=launch_id,
                        workspace_generation=generation,
                        container_generation=container_generation,
                        member_kind="wandb",
                        ordinal=0,
                        receipt={
                            "container_name": container_name,
                            "container_id": receipt.get("container_id"),
                            "verified_absent_at": datetime.now(UTC).isoformat(),
                        },
                    )

                run_wandb_publisher_recovery_container(
                    host,
                    launch_id=launch_id,
                    run_name=str(
                        recovery.get("run_name") or f"train_job_{recovery['id']}"
                    ),
                    runtime_image_ref=str(recovery["runtime_image_ref"]),
                    attempt_env_path=manifest.env_path,
                    on_attested=attested,
                    on_absent=absent,
                    container_name=container_name,
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
            control = machine_control(conn, machine=machine.name)
            if bool(control.get("drained")) and not active_job_launches(conn, machine=machine.name):
                return {
                    "reconciled": 0,
                    "launched": 0,
                    "recovered_publications": 0,
                    "removed_containers": 0,
                    "pruned_images": 0,
                    "prewarm": {"status": "skipped_drained"},
                    "maintenance_skipped": "machine is drained with no active launches",
                }
            credentials_cleaned = cleanup_terminal_credentials(conn, host)
            workspace_gc: dict[str, Any]
            if deadline_monotonic - time.monotonic() >= 45.0:
                if progress:
                    progress(
                        "RECLAIMING WORKSPACES",
                        "Reducing durable proofs and running one bounded cleanup quantum",
                    )
                workspace_gc = run_workspace_gc_pass(conn, host=host)
            else:
                workspace_gc = {"status": "skipped_normal_work_reserve"}
            if progress:
                progress(
                    "RECONCILING TRAIN", "Observing active containers and filling available slots"
                )
            reconciled, launched = run_reconcile_fill_pass(
                conn,
                host=host,
                shared_env_file=repo_root / DEFAULT_SHARED_RUNNER_ENV_FILE,
            )
            if time.monotonic() < deadline_monotonic:
                credentials_cleaned += cleanup_terminal_credentials(conn, host)
            drain_finalized = False
            if bool(control.get("drain_requested")) and time.monotonic() < deadline_monotonic:
                drain_finalized = try_finalize_requested_drain(conn, host)
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
                    retry_state = runtime_image_retry_state(
                        conn,
                        machine=machine.name,
                        runtime_image_ref=prewarmed_ref,
                    )
                    next_retry_at = (retry_state or {}).get("next_retry_at")
                    if next_retry_at is not None and next_retry_at > datetime.now(UTC):
                        prewarm = {
                            "status": "backoff",
                            "runtime_image_ref": prewarmed_ref,
                            "retry_count": int((retry_state or {}).get("retry_count") or 0),
                            "next_retry_at": next_retry_at.isoformat(),
                            "error": str((retry_state or {}).get("last_error") or ""),
                        }
                    else:
                        prewarm, prewarmed_ref = prewarm_latest_runtime(
                            host,
                            state_path=(
                                repo_root
                                / "logs"
                                / "fleet"
                                / f"prewarm-{machine.name}.json"
                            ),
                            release=release,
                        )
                        reset_runtime_image_retry(
                            conn,
                            machine=machine.name,
                            runtime_image_ref=prewarmed_ref,
                        )
                except Exception as exc:
                    if prewarmed_ref:
                        record_runtime_image_failure(
                            conn,
                            machine=machine.name,
                            runtime_image_ref=prewarmed_ref,
                            error=str(exc),
                        )
                    prewarm = {"status": "error", "error": str(exc)}
            if time.monotonic() >= deadline_monotonic:
                return {
                    "reconciled": reconciled,
                    "launched": launched,
                    "recovered_publications": recovered_publications,
                    "removed_containers": 0,
                    "pruned_images": 0,
                    "prewarm": prewarm,
                    "workspace_gc": workspace_gc,
                    "terminal_credentials_cleaned": credentials_cleaned,
                    "drain_finalized": drain_finalized,
                    "maintenance_skipped": "machine lane deadline expired after reconciliation",
                }
            maintenance_marker = repo_root / "logs" / "fleet" / f"maintenance-{machine.name}.stamp"
            pruned = 0
            removed_containers = 0
            image_pruning_skipped = ""
            if reconciled or launched or _maintenance_due(maintenance_marker):
                if progress:
                    progress(
                        "REMOVING STALE RESOURCES",
                        "Pruning inactive containers and unused runtime images safely",
                    )
                removed_containers = prune_inactive_job_containers(conn, host)
                if prewarm.get("status") == "error":
                    image_pruning_skipped = "latest runtime prewarm failed"
                else:
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
                "image_pruning_skipped": image_pruning_skipped,
                "workspace_gc": workspace_gc,
                "terminal_credentials_cleaned": credentials_cleaned,
                "drain_finalized": drain_finalized,
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


def cmd_workspace_gc_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        status = workspace_gc_status(conn, machine=args.machine)
    finally:
        conn.close()
    print(json.dumps(json_safe(status), sort_keys=True))
    return 0


def cmd_workspace_gc_doctor(args: argparse.Namespace) -> int:
    registry = load_registry_from_args(args)
    selected = [args.machine] if args.machine else sorted(registry.machines)
    helpers: dict[str, Any] = {}
    for name in selected:
        machine = resolve_machine(registry, name)
        try:
            helpers[name] = DockerRunnerHost(machine).workspace_helper_doctor()
        except Exception as exc:
            helpers[name] = {"error": str(exc)}
    conn = _connect_from_args(args)
    try:
        status = workspace_gc_status(conn, machine=args.machine)
    finally:
        conn.close()
    healthy = all("error" not in value for value in helpers.values())
    print(
        json.dumps(
            json_safe({"healthy": healthy, "helpers": helpers, "status": status}),
            sort_keys=True,
        )
    )
    return 0 if healthy else 1


def cmd_workspace_gc_install_key(args: argparse.Namespace) -> int:
    public_key = args.public_key.expanduser().resolve().read_bytes()
    registry = load_registry_from_args(args)
    selected = [args.machine] if args.machine else sorted(registry.machines)
    receipts: dict[str, Any] = {}
    for name in selected:
        machine = resolve_machine(registry, name)
        receipts[name] = DockerRunnerHost(machine).install_workspace_key_policy(
            key_revision=args.key_revision,
            public_key=public_key,
        )
    print(json.dumps(json_safe(receipts), sort_keys=True))
    return 0


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON evidence must be an object: {path}")
    return value


def cmd_workspace_gc_create_schedule(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = create_workspace_qualification_schedule(
            conn,
            rollout_owner=args.owner,
            control_revision=args.control_revision,
            schedule=_load_json_object(args.schedule_file),
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_record_qualification(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = record_workspace_qualification_receipt(
            conn,
            schedule_id=args.schedule_id,
            machine=args.machine,
            machine_control_revision=args.machine_control_revision,
            evidence=_load_json_object(args.evidence_file),
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_complete_schedule(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = complete_workspace_qualification_schedule(
            conn, schedule_id=args.schedule_id, rollout_owner=args.owner
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_authorize_canary(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = authorize_workspace_cleanup_canary(
            conn,
            schedule_id=args.schedule_id,
            cleanup_row_id=args.cleanup_row_id,
            rollout_owner=args.owner,
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_authorize_enqueue(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = authorize_workspace_enqueue(
            conn,
            schedule_id=args.schedule_id,
            machine=args.machine,
            submission_key=args.submission_key,
            request_hash=args.request_hash,
            rollout_owner=args.owner,
            promotion_only=bool(args.promotion_only),
        )
    finally:
        conn.close()
    payload = json_safe(result)
    payload["usage"] = (
        f"RLAB_WORKSPACE_ENQUEUE_CAPABILITY_ID={result['capability_id']} "
        "<exact queue submission>"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def _set_gc_rollout(args: argparse.Namespace, *, mode: str, paused: bool, enabled: bool) -> int:
    conn = _connect_from_args(args)
    try:
        result = set_workspace_rollout_control(
            conn,
            expected_revision=args.expected_revision,
            rollout_owner=args.owner,
            protocol_mode=mode,
            work_creation_paused=paused,
            cleanup_globally_enabled=enabled,
            reason=args.reason,
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_qualify(args: argparse.Namespace) -> int:
    return _set_gc_rollout(args, mode="qualification", paused=True, enabled=True)


def cmd_workspace_gc_promote(args: argparse.Namespace) -> int:
    return _set_gc_rollout(
        args, mode="promotion_verifying", paused=True, enabled=True
    )


def cmd_workspace_gc_record_promotion(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        result = record_workspace_promotion_receipt(
            conn,
            launch_id=args.launch_id,
            rollout_control_revision=args.control_revision,
            evidence=_load_json_object(args.evidence_file),
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_activate(args: argparse.Namespace) -> int:
    return _set_gc_rollout(args, mode="active", paused=False, enabled=True)


def cmd_workspace_gc_disable(args: argparse.Namespace) -> int:
    return _set_gc_rollout(args, mode="active", paused=False, enabled=False)


def cmd_workspace_gc_rollback(args: argparse.Namespace) -> int:
    return _set_gc_rollout(args, mode="rollback", paused=True, enabled=False)


def cmd_workspace_gc_enable_machine(args: argparse.Namespace) -> int:
    resolve_machine(load_registry_from_args(args), args.machine)
    conn = _connect_from_args(args)
    try:
        result = set_machine_cleanup_control(
            conn,
            machine=args.machine,
            expected_revision=args.expected_machine_revision,
            enabled=not args.disable,
            evidence_sha256=args.evidence_sha256,
            rollout_owner=args.owner,
            reason=args.reason,
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


def cmd_workspace_gc_resolve_review(args: argparse.Namespace) -> int:
    evidence = json.loads(args.evidence_file.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping):
        raise SystemExit("review evidence file must contain one JSON object")
    conn = _connect_from_args(args)
    try:
        result = resolve_rollback_review(
            conn,
            cleanup_row_id=args.row_id,
            expected_control_revision=args.expected_control_revision,
            evidence=evidence,
            resolution=args.resolution,
        )
    finally:
        conn.close()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


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

    workspace_gc_parser = subparsers.add_parser(
        "workspace-gc", help="Inspect and control proof-gated successful workspace cleanup."
    )
    workspace_gc_commands = workspace_gc_parser.add_subparsers(
        dest="workspace_gc_command", required=True
    )
    gc_status = workspace_gc_commands.add_parser("status")
    add_database_arg(gc_status)
    gc_status.add_argument("--machine")
    gc_status.set_defaults(func=cmd_workspace_gc_status)

    gc_doctor = workspace_gc_commands.add_parser("doctor")
    add_database_arg(gc_doctor)
    add_machine_registry_arg(gc_doctor)
    gc_doctor.add_argument("--machine")
    gc_doctor.set_defaults(func=cmd_workspace_gc_doctor)

    gc_install_key = workspace_gc_commands.add_parser("install-key")
    add_machine_registry_arg(gc_install_key)
    gc_install_key.add_argument("--machine")
    gc_install_key.add_argument("--key-revision", required=True)
    gc_install_key.add_argument("--public-key", type=Path, required=True)
    gc_install_key.set_defaults(func=cmd_workspace_gc_install_key)

    gc_schedule = workspace_gc_commands.add_parser("create-schedule")
    add_database_arg(gc_schedule)
    gc_schedule.add_argument("--control-revision", type=int, required=True)
    gc_schedule.add_argument("--owner", required=True)
    gc_schedule.add_argument("--schedule-file", type=Path, required=True)
    gc_schedule.set_defaults(func=cmd_workspace_gc_create_schedule)

    gc_record = workspace_gc_commands.add_parser("record-qualification")
    add_database_arg(gc_record)
    gc_record.add_argument("--schedule-id", required=True)
    gc_record.add_argument("--machine", required=True)
    gc_record.add_argument("--machine-control-revision", type=int, required=True)
    gc_record.add_argument("--evidence-file", type=Path, required=True)
    gc_record.set_defaults(func=cmd_workspace_gc_record_qualification)

    gc_complete = workspace_gc_commands.add_parser("complete-schedule")
    add_database_arg(gc_complete)
    gc_complete.add_argument("--schedule-id", required=True)
    gc_complete.add_argument("--owner", required=True)
    gc_complete.set_defaults(func=cmd_workspace_gc_complete_schedule)

    gc_canary = workspace_gc_commands.add_parser("authorize-canary")
    add_database_arg(gc_canary)
    gc_canary.add_argument("--schedule-id", required=True)
    gc_canary.add_argument("--cleanup-row-id", type=int, required=True)
    gc_canary.add_argument("--owner", required=True)
    gc_canary.set_defaults(func=cmd_workspace_gc_authorize_canary)

    gc_enqueue = workspace_gc_commands.add_parser("authorize-enqueue")
    add_database_arg(gc_enqueue)
    gc_enqueue.add_argument("--schedule-id", required=True)
    gc_enqueue.add_argument("--machine", required=True)
    gc_enqueue.add_argument("--submission-key", required=True)
    gc_enqueue.add_argument("--request-hash", required=True)
    gc_enqueue.add_argument("--owner", required=True)
    gc_enqueue.add_argument("--promotion-only", action="store_true")
    gc_enqueue.set_defaults(func=cmd_workspace_gc_authorize_enqueue)

    def add_rollout_control_args(command: argparse.ArgumentParser) -> None:
        add_database_arg(command)
        command.add_argument("--expected-revision", type=int, required=True)
        command.add_argument("--owner", required=True)
        command.add_argument("--reason", required=True)

    gc_qualify = workspace_gc_commands.add_parser("qualify")
    add_rollout_control_args(gc_qualify)
    gc_qualify.set_defaults(func=cmd_workspace_gc_qualify)
    gc_promote = workspace_gc_commands.add_parser("promote")
    add_rollout_control_args(gc_promote)
    gc_promote.set_defaults(func=cmd_workspace_gc_promote)
    gc_record_promotion = workspace_gc_commands.add_parser("record-promotion")
    add_database_arg(gc_record_promotion)
    gc_record_promotion.add_argument("--launch-id", required=True)
    gc_record_promotion.add_argument("--control-revision", type=int, required=True)
    gc_record_promotion.add_argument("--evidence-file", type=Path, required=True)
    gc_record_promotion.set_defaults(func=cmd_workspace_gc_record_promotion)
    gc_activate = workspace_gc_commands.add_parser("activate")
    add_rollout_control_args(gc_activate)
    gc_activate.set_defaults(func=cmd_workspace_gc_activate)
    gc_disable = workspace_gc_commands.add_parser("disable")
    add_rollout_control_args(gc_disable)
    gc_disable.set_defaults(func=cmd_workspace_gc_disable)
    gc_rollback = workspace_gc_commands.add_parser("rollback")
    add_rollout_control_args(gc_rollback)
    gc_rollback.set_defaults(func=cmd_workspace_gc_rollback)

    gc_machine = workspace_gc_commands.add_parser("enable-machine")
    add_database_arg(gc_machine)
    add_machine_registry_arg(gc_machine)
    gc_machine.add_argument("--machine", required=True)
    gc_machine.add_argument("--expected-machine-revision", type=int, required=True)
    gc_machine.add_argument("--owner", required=True)
    gc_machine.add_argument("--reason", required=True)
    gc_machine.add_argument("--evidence-sha256")
    gc_machine.add_argument("--disable", action="store_true")
    gc_machine.set_defaults(func=cmd_workspace_gc_enable_machine)

    gc_review = workspace_gc_commands.add_parser("resolve-review")
    add_database_arg(gc_review)
    gc_review.add_argument("--row-id", type=int, required=True)
    gc_review.add_argument("--expected-control-revision", type=int, required=True)
    gc_review.add_argument("--evidence-file", type=Path, required=True)
    gc_review.add_argument("--resolution", choices=("defer", "resume"), required=True)
    gc_review.set_defaults(func=cmd_workspace_gc_resolve_review)

    from rlab.fleet_service import add_service_parser

    add_service_parser(subparsers)

    queue = subparsers.add_parser("queue", help="Maintain the PostgreSQL queue schema.")
    queue_commands = queue.add_subparsers(dest="queue_command", required=True)
    from rlab.job_queue import cmd_reset_schema, cmd_setup

    queue_setup = queue_commands.add_parser("setup", help="Create queue tables.")
    add_database_arg(queue_setup)
    queue_setup.add_argument(
        "--worker-mailbox-role",
        default=os.environ.get("WORKER_MAILBOX_ROLE"),
    )
    queue_setup.set_defaults(func=cmd_setup)

    queue_reset = queue_commands.add_parser(
        "reset-schema", help="Export, drop, and recreate the queue schema."
    )
    add_database_arg(queue_reset)
    queue_reset.add_argument("--export-dir", type=Path, default=None)
    add_dry_run_arg(queue_reset)
    queue_reset.set_defaults(func=cmd_reset_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
