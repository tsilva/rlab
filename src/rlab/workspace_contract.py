from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


WORKSPACE_LAYOUT_VERSION = 1
WORKSPACE_HELPER_PROTOCOL_VERSION = 1
WORKSPACE_BATCH_LIMIT = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LAUNCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


class WorkspaceContractError(ValueError):
    pass


class WorkspaceState(StrEnum):
    PENDING = "pending"
    DELETING = "deleting"
    HOST_DELETED = "host_deleted"
    COMPLETED = "completed"
    ROLLBACK_REVIEW = "rollback_review"


class DeletionProgress(StrEnum):
    CLAIMED = "claimed"
    PREPARE_PENDING = "prepare_pending"
    PREPARED = "prepared"
    MUTATION_PENDING = "mutation_pending"
    MUTATING = "mutating"
    PARTIAL = "partial"
    MUTATION_COMPLETE = "mutation_complete"


class BatchState(StrEnum):
    SIGNING_PENDING = "signing_pending"
    SIGNER_CLAIMED = "signer_claimed"
    TOKEN_ISSUED = "token_issued"
    DELIVERY_INTENT = "delivery_intent"
    DELIVERY_ACK = "delivery_ack"
    DELIVERY_AMBIGUOUS = "delivery_ambiguous"
    CONSUMED = "consumed"
    EXITED = "exited"
    RECONCILED = "reconciled"
    SUPERSEDED = "superseded"
    CANCELED = "canceled"
    SIGNER_FAILED = "signer_failed"
    EXPIRED = "expired"


class HostOperationState(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    FAILED = "failed"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _absolute_clean_path(value: str, *, label: str) -> str:
    raw = str(value or "")
    if not raw.startswith("/"):
        raise WorkspaceContractError(f"{label} must be an absolute POSIX path")
    if "\x00" in raw:
        raise WorkspaceContractError(f"{label} contains a NUL byte")
    clean = posixpath.normpath(raw)
    if clean != raw.rstrip("/"):
        raise WorkspaceContractError(f"{label} must already be normalized: {raw!r}")
    if clean == "/":
        raise WorkspaceContractError(f"{label} must not be the filesystem root")
    return clean


def _strict_child(path: str, parent: str) -> bool:
    return path != parent and path.startswith(parent.rstrip("/") + "/")


def _overlap(first: str, second: str) -> bool:
    return first == second or _strict_child(first, second) or _strict_child(second, first)


def validate_workspace_roots(
    *,
    host_root: str,
    payloads_root: str,
    outputs_root: str,
    protected_metadata_root: str,
    logs_root: str,
    shared_env_path: str,
    rom_cache_root: str,
    container_payloads_root: str,
    container_outputs_root: str,
) -> dict[str, str]:
    roots = {
        "host_root": _absolute_clean_path(host_root, label="host_root"),
        "payloads_root": _absolute_clean_path(payloads_root, label="payloads_root"),
        "outputs_root": _absolute_clean_path(outputs_root, label="outputs_root"),
        "protected_metadata_root": _absolute_clean_path(
            protected_metadata_root, label="protected_metadata_root"
        ),
        "logs_root": _absolute_clean_path(logs_root, label="logs_root"),
        "shared_env_path": _absolute_clean_path(shared_env_path, label="shared_env_path"),
        "rom_cache_root": _absolute_clean_path(rom_cache_root, label="rom_cache_root"),
        "container_payloads_root": _absolute_clean_path(
            container_payloads_root, label="container_payloads_root"
        ),
        "container_outputs_root": _absolute_clean_path(
            container_outputs_root, label="container_outputs_root"
        ),
    }
    for key in ("payloads_root", "outputs_root", "protected_metadata_root", "logs_root"):
        if not _strict_child(roots[key], roots["host_root"]):
            raise WorkspaceContractError(f"{key} must be a strict child of host_root")
    owned = (
        "payloads_root",
        "outputs_root",
        "protected_metadata_root",
        "logs_root",
        "shared_env_path",
        "rom_cache_root",
    )
    for offset, first_key in enumerate(owned):
        for second_key in owned[offset + 1 :]:
            if _overlap(roots[first_key], roots[second_key]):
                raise WorkspaceContractError(
                    f"workspace roots overlap: {first_key}={roots[first_key]!r}, "
                    f"{second_key}={roots[second_key]!r}"
                )
    if _overlap(roots["container_payloads_root"], roots["container_outputs_root"]):
        raise WorkspaceContractError("container payload and output roots must not overlap")
    return roots


@dataclass(frozen=True)
class WorkspaceManifest:
    protocol_version: int
    layout_version: int
    machine: str
    launch_id: str
    generation: int
    reservation_nonce: str
    payload_path: str
    env_path: str
    output_path: str
    ownership_marker_path: str
    reservation_receipt_path: str
    container_payload_path: str
    container_output_path: str
    payload_sha256: str
    roots: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())

    @property
    def deletion_targets(self) -> tuple[str, ...]:
        return (
            self.payload_path,
            self.env_path,
            self.output_path,
            self.ownership_marker_path,
            self.reservation_receipt_path,
        )


def build_workspace_manifest(
    *,
    machine: str,
    launch_id: str,
    generation: int,
    reservation_nonce: str,
    payload_sha256: str,
    host_root: str,
    payloads_root: str,
    outputs_root: str,
    protected_metadata_root: str,
    logs_root: str,
    shared_env_path: str,
    rom_cache_root: str,
    container_payloads_root: str,
    container_outputs_root: str,
) -> WorkspaceManifest:
    if not LAUNCH_ID_RE.fullmatch(str(launch_id or "")):
        raise WorkspaceContractError("launch_id contains unsupported characters")
    if int(generation) < 1:
        raise WorkspaceContractError("generation must be at least one")
    if not SHA256_RE.fullmatch(str(payload_sha256 or "")):
        raise WorkspaceContractError("payload_sha256 must be a lowercase SHA-256 digest")
    nonce = str(reservation_nonce or "")
    if len(nonce) < 32 or not re.fullmatch(r"[A-Za-z0-9_-]+", nonce):
        raise WorkspaceContractError("reservation_nonce must be at least 32 URL-safe characters")
    roots = validate_workspace_roots(
        host_root=host_root,
        payloads_root=payloads_root,
        outputs_root=outputs_root,
        protected_metadata_root=protected_metadata_root,
        logs_root=logs_root,
        shared_env_path=shared_env_path,
        rom_cache_root=rom_cache_root,
        container_payloads_root=container_payloads_root,
        container_outputs_root=container_outputs_root,
    )
    payload_path = f"{roots['payloads_root']}/{launch_id}.json"
    env_path = f"{roots['payloads_root']}/{launch_id}.env"
    output_path = f"{roots['outputs_root']}/{launch_id}"
    marker = f"{roots['protected_metadata_root']}/{launch_id}.ownership.json"
    receipt = f"{roots['protected_metadata_root']}/{launch_id}.reservation.json"
    container_payload = f"{roots['container_payloads_root']}/{launch_id}.json"
    container_output = f"{roots['container_outputs_root']}/{launch_id}"
    exact_paths = (payload_path, env_path, output_path, marker, receipt)
    if len(set(exact_paths)) != len(exact_paths):
        raise WorkspaceContractError("workspace target paths must be distinct")
    return WorkspaceManifest(
        protocol_version=WORKSPACE_HELPER_PROTOCOL_VERSION,
        layout_version=WORKSPACE_LAYOUT_VERSION,
        machine=str(machine),
        launch_id=str(launch_id),
        generation=int(generation),
        reservation_nonce=nonce,
        payload_path=payload_path,
        env_path=env_path,
        output_path=output_path,
        ownership_marker_path=marker,
        reservation_receipt_path=receipt,
        container_payload_path=container_payload,
        container_output_path=container_output,
        payload_sha256=str(payload_sha256),
        roots=roots,
    )


def workspace_manifest_from_mapping(value: Mapping[str, Any]) -> WorkspaceManifest:
    roots_value = value.get("roots")
    if not isinstance(roots_value, Mapping):
        raise WorkspaceContractError("workspace manifest roots must be an object")
    expected = build_workspace_manifest(
        machine=str(value.get("machine") or ""),
        launch_id=str(value.get("launch_id") or ""),
        generation=int(value.get("generation") or 0),
        reservation_nonce=str(value.get("reservation_nonce") or ""),
        payload_sha256=str(value.get("payload_sha256") or ""),
        host_root=str(roots_value.get("host_root") or ""),
        payloads_root=str(roots_value.get("payloads_root") or ""),
        outputs_root=str(roots_value.get("outputs_root") or ""),
        protected_metadata_root=str(roots_value.get("protected_metadata_root") or ""),
        logs_root=str(roots_value.get("logs_root") or ""),
        shared_env_path=str(roots_value.get("shared_env_path") or ""),
        rom_cache_root=str(roots_value.get("rom_cache_root") or ""),
        container_payloads_root=str(roots_value.get("container_payloads_root") or ""),
        container_outputs_root=str(roots_value.get("container_outputs_root") or ""),
    )
    supplied = dict(value)
    if supplied != expected.as_dict():
        raise WorkspaceContractError("workspace manifest fields do not match canonical paths")
    return expected


@dataclass(frozen=True)
class CleanupRowEnvelope:
    row_id: int
    cleanup_attempt_id: str
    generation: int
    manifest_sha256: str
    prepare_receipt_sha256: str
    starting_cursor_sha256: str

    def validate(self) -> None:
        if self.row_id < 1 or self.generation < 1:
            raise WorkspaceContractError("cleanup row identity must be positive")
        for label, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("prepare_receipt_sha256", self.prepare_receipt_sha256),
            ("starting_cursor_sha256", self.starting_cursor_sha256),
        ):
            if not SHA256_RE.fullmatch(digest):
                raise WorkspaceContractError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CleanupBatchEnvelope:
    protocol_version: int
    batch_id: str
    machine: str
    helper_revision: str
    key_revision: str
    control_revision: int
    epoch: int
    boot_id: str
    monotonic_deadline_ns: int
    rows: tuple[CleanupRowEnvelope, ...]

    def validate(self) -> None:
        if self.protocol_version != WORKSPACE_HELPER_PROTOCOL_VERSION:
            raise WorkspaceContractError("unsupported cleanup batch protocol")
        if self.control_revision < 1 or self.epoch < 1 or self.monotonic_deadline_ns < 1:
            raise WorkspaceContractError("cleanup batch revisions and deadline must be positive")
        if not 1 <= len(self.rows) <= WORKSPACE_BATCH_LIMIT:
            raise WorkspaceContractError(
                f"cleanup batch must contain between 1 and {WORKSPACE_BATCH_LIMIT} rows"
            )
        seen: set[int] = set()
        for row in self.rows:
            row.validate()
            if row.row_id in seen:
                raise WorkspaceContractError("cleanup batch contains a duplicate row")
            seen.add(row.row_id)

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())


def cleanup_batch_from_mapping(value: Mapping[str, Any]) -> CleanupBatchEnvelope:
    rows_value = value.get("rows")
    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise WorkspaceContractError("cleanup batch rows must be an array")
    rows = tuple(CleanupRowEnvelope(**dict(row)) for row in rows_value)
    envelope = CleanupBatchEnvelope(
        protocol_version=int(value.get("protocol_version") or 0),
        batch_id=str(value.get("batch_id") or ""),
        machine=str(value.get("machine") or ""),
        helper_revision=str(value.get("helper_revision") or ""),
        key_revision=str(value.get("key_revision") or ""),
        control_revision=int(value.get("control_revision") or 0),
        epoch=int(value.get("epoch") or 0),
        boot_id=str(value.get("boot_id") or ""),
        monotonic_deadline_ns=int(value.get("monotonic_deadline_ns") or 0),
        rows=rows,
    )
    envelope.validate()
    return envelope


def exact_bind_mount(source: str, destination: str, *, readonly: bool) -> str:
    source_path = _absolute_clean_path(source, label="mount source")
    destination_path = _absolute_clean_path(destination, label="mount destination")
    parts = [
        "type=bind",
        f"src={source_path}",
        f"dst={destination_path}",
        "bind-nonrecursive=true",
    ]
    if readonly:
        parts.append("readonly")
    return ",".join(parts)
