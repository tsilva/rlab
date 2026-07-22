from __future__ import annotations

import argparse
import base64
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from rlab.workspace_contract import (
    CleanupBatchEnvelope,
    WORKSPACE_HELPER_PROTOCOL_VERSION,
    WorkspaceContractError,
    WorkspaceManifest,
    canonical_json_bytes,
    cleanup_batch_from_mapping,
    sha256_json,
    workspace_manifest_from_mapping,
)


HELPER_REVISION = "workspace-helper-v1"
PREPARE_LIFETIME_NS = 30_000_000_000
MUTATION_LIFETIME_NS = 3_000_000_000


class WorkspaceHelperError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    kind: str

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        kind = (
            "directory"
            if stat.S_ISDIR(value.st_mode)
            else "regular"
            if stat.S_ISREG(value.st_mode)
            else "other"
        )
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=stat.S_IMODE(value.st_mode),
            uid=int(value.st_uid),
            gid=int(value.st_gid),
            kind=kind,
        )


def _identity(path: str, *, follow_symlinks: bool = False) -> FileIdentity:
    try:
        value = os.stat(path, follow_symlinks=follow_symlinks)
    except FileNotFoundError as exc:
        raise WorkspaceHelperError(f"required path is absent: {path}") from exc
    identity = FileIdentity.from_stat(value)
    if identity.kind == "other":
        raise WorkspaceHelperError(f"unsupported file type: {path}")
    return identity


def _fsync_directory(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: str, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=parent)
    try:
        os.fchmod(fd, mode)
        payload = canonical_json_bytes(dict(value)) + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_json(path: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        value = os.read(fd, 16 * 1024 * 1024)
        if os.read(fd, 1):
            raise WorkspaceHelperError(f"JSON receipt is too large: {path}")
    finally:
        os.close(fd)
    try:
        document = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceHelperError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(document, dict):
        raise WorkspaceHelperError(f"JSON receipt must be an object: {path}")
    return document


def _continuous_ns() -> int:
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    if platform.system() == "Darwin":
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        library.mach_continuous_time.restype = ctypes.c_uint64
        class MachTimebaseInfo(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

        info = MachTimebaseInfo()
        library.mach_timebase_info(ctypes.byref(info))
        return int(library.mach_continuous_time()) * int(info.numer) // int(info.denom)
    return time.monotonic_ns()


def _boot_id() -> str:
    linux = "/proc/sys/kernel/random/boot_id"
    if os.path.isfile(linux):
        return Path(linux).read_text(encoding="utf-8").strip()
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise WorkspaceHelperError("host boot identity is unavailable")


@contextmanager
def helper_lock(manifest: WorkspaceManifest) -> Iterator[None]:
    with protected_root_lock(manifest.roots["protected_metadata_root"]):
        yield


@contextmanager
def protected_root_lock(root: str) -> Iterator[None]:
    root = os.path.abspath(root)
    if root == "/" or not os.path.isabs(root):
        raise WorkspaceHelperError("protected metadata root must be absolute and non-root")
    os.makedirs(root, mode=0o700, exist_ok=True)
    path = os.path.join(root, ".workspace-helper.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _container_absent(container_name: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", container_name):
        raise WorkspaceHelperError("container name is not path-safe")
    result = subprocess.run(
        ["docker", "container", "inspect", container_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode == 0:
        return False
    detail = str(result.stderr or result.stdout or "").lower()
    if "no such" in detail or "not found" in detail:
        return True
    raise WorkspaceHelperError(f"cannot inventory container {container_name}: {detail.strip()}")


def drain_zero_receipt(
    *,
    protected_metadata_root: str,
    manifests: tuple[WorkspaceManifest, ...],
    exact_container_names: tuple[str, ...],
    exact_protected_paths: tuple[str, ...],
    receipt_nonce: str,
) -> dict[str, Any]:
    root = os.path.abspath(protected_metadata_root)
    if len(receipt_nonce) < 32:
        raise WorkspaceHelperError("drain receipt nonce is too short")
    if any(manifest.roots["protected_metadata_root"] != root for manifest in manifests):
        raise WorkspaceHelperError("drain inventory manifests cross protected roots")
    root_prefix = f"{root.rstrip(os.sep)}{os.sep}"
    if any(
        not os.path.abspath(path).startswith(root_prefix) for path in exact_protected_paths
    ):
        raise WorkspaceHelperError("drain inventory path escaped protected metadata root")
    with protected_root_lock(root):
        discovered_journals = tuple(
            entry.path
            for entry in os.scandir(root)
            if entry.is_file(follow_symlinks=False)
            and (
                entry.name.endswith(".reservation-intent.json")
                or ".env-intent." in entry.name
                or ".delete-prepare." in entry.name
                or ".host-deleted." in entry.name
            )
        )
        counts = {
            "credential_envs": sum(os.path.lexists(manifest.env_path) for manifest in manifests),
            "reservation_intents": sum(
                os.path.lexists(reservation_intent_path(manifest)) for manifest in manifests
            ),
            "protected_journals": sum(
                os.path.lexists(path) for path in exact_protected_paths
            ),
            "discovered_protected_journals": len(discovered_journals),
            "credential_containers": sum(
                not _container_absent(name) for name in exact_container_names
            ),
        }
        targets = tuple(
            target for manifest in manifests for target in manifest.deletion_targets
        )
        try:
            _assert_no_container_mounts(targets)
        except WorkspaceHelperError:
            counts["workspace_mounts"] = 1
        else:
            counts["workspace_mounts"] = 0
        receipt = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "helper_revision": HELPER_REVISION,
            "boot_id": _boot_id(),
            "continuous_ns": _continuous_ns(),
            "receipt_nonce": receipt_nonce,
            "protected_metadata_root": root,
            "manifest_sha256s": sorted(manifest.digest for manifest in manifests),
            "container_names": sorted(exact_container_names),
            "protected_paths": sorted(exact_protected_paths),
            "counts": counts,
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt


def install_key_policy(
    *, protected_metadata_root: str, key_revision: str, public_key: bytes
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", key_revision):
        raise WorkspaceHelperError("workspace key revision is not path-safe")
    if not public_key.startswith(b"-----BEGIN PUBLIC KEY-----"):
        raise WorkspaceHelperError("workspace cleanup key is not a PEM public key")
    root = os.path.abspath(protected_metadata_root)
    key_path = os.path.join(root, f".workspace-public-{key_revision}.pem")
    policy_path = os.path.join(root, ".workspace-key-policy.json")
    with protected_root_lock(root):
        key_digest = hashlib.sha256(public_key).hexdigest()
        if os.path.lexists(key_path):
            if Path(key_path).read_bytes() != public_key:
                raise WorkspaceHelperError("workspace public key revision changed")
        else:
            fd = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            try:
                os.write(fd, public_key)
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_directory(root)
        policy = _load_json(policy_path) if os.path.lexists(policy_path) else {"keys": {}}
        keys = policy.get("keys")
        if not isinstance(keys, dict):
            raise WorkspaceHelperError("workspace key policy keys changed type")
        for value in keys.values():
            if isinstance(value, dict) and value.get("state") == "active":
                value["state"] = "overlap"
        existing = keys.get(key_revision)
        expected = {
            "state": "active",
            "public_key_path": key_path,
            "public_key_sha256": key_digest,
        }
        if existing and any(
            existing.get(field) != expected[field]
            for field in ("public_key_path", "public_key_sha256")
        ):
            raise WorkspaceHelperError("workspace public key revision identity changed")
        keys[key_revision] = expected
        policy = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "keys": keys,
        }
        policy["policy_sha256"] = sha256_json(policy)
        _atomic_json(policy_path, policy)
        return {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "key_revision": key_revision,
            "public_key_path": key_path,
            "public_key_sha256": key_digest,
            "policy_path": policy_path,
            "policy_sha256": policy["policy_sha256"],
        }


def reservation_intent_path(manifest: WorkspaceManifest) -> str:
    return os.path.join(
        manifest.roots["protected_metadata_root"],
        f"{manifest.launch_id}.reservation-intent.json",
    )


def _create_placeholder(path: str, *, mode: int) -> FileIdentity:
    parent = os.path.dirname(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        os.fchmod(fd, mode)
        os.fsync(fd)
        identity = FileIdentity.from_stat(os.fstat(fd))
    finally:
        os.close(fd)
    _fsync_directory(parent)
    return identity


def _write_reserved_regular(path: str, expected: FileIdentity, payload: bytes) -> FileIdentity:
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        actual = FileIdentity.from_stat(os.fstat(fd))
        if actual.device != expected.device or actual.inode != expected.inode:
            raise WorkspaceHelperError(f"reserved inode changed before write: {path}")
        if actual.kind != "regular":
            raise WorkspaceHelperError(f"reserved path is not a regular file: {path}")
        os.ftruncate(fd, 0)
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        return FileIdentity.from_stat(os.fstat(fd))
    finally:
        os.close(fd)


def _record_phase(intent_path: str, intent: dict[str, Any], phase: str) -> None:
    intent["phase"] = phase
    _atomic_json(intent_path, intent)


def _validate_env(env: Mapping[str, str]) -> bytes:
    lines: list[str] = []
    for key in sorted(env):
        value = str(env[key])
        if not key or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in key):
            raise WorkspaceContractError(f"invalid attempt environment key: {key!r}")
        if not value or any(char in value for char in ("\x00", "\n", "\r")):
            raise WorkspaceContractError(f"invalid attempt environment value for {key}")
        lines.append(f"{key}={value}\n")
    if not lines:
        raise WorkspaceContractError("attempt environment must not be empty")
    return "".join(lines).encode("utf-8")


def reserve_workspace(
    manifest: WorkspaceManifest,
    *,
    payload: bytes,
    attempt_env: Mapping[str, str],
) -> dict[str, Any]:
    if hashlib.sha256(payload).hexdigest() != manifest.payload_sha256:
        raise WorkspaceHelperError("payload bytes do not match manifest payload_sha256")
    env_payload = _validate_env(attempt_env)
    env_fingerprint = hashlib.sha256(env_payload).hexdigest()
    intent_path = reservation_intent_path(manifest)
    with helper_lock(manifest):
        for root_key in ("payloads_root", "outputs_root", "protected_metadata_root"):
            os.makedirs(manifest.roots[root_key], mode=0o700, exist_ok=True)
            _fsync_directory(os.path.dirname(manifest.roots[root_key]))
        if os.path.exists(manifest.reservation_receipt_path):
            receipt = _load_json(manifest.reservation_receipt_path)
            if receipt.get("manifest_sha256") != manifest.digest:
                raise WorkspaceHelperError("existing reservation belongs to a different manifest")
            return receipt
        if os.path.lexists(intent_path):
            intent = _load_json(intent_path)
            if intent.get("manifest_sha256") != manifest.digest:
                raise WorkspaceHelperError("existing reservation intent belongs to another manifest")
            raise WorkspaceHelperError("matching partial reservation requires recovery")
        intent: dict[str, Any] = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "helper_revision": HELPER_REVISION,
            "manifest_sha256": manifest.digest,
            "reservation_nonce": manifest.reservation_nonce,
            "phase": "anchor_written",
            "leaves": {},
        }
        _atomic_json(intent_path, intent)

        payload_identity = _create_placeholder(manifest.payload_path, mode=0o600)
        intent["leaves"]["payload"] = asdict(payload_identity)
        _record_phase(intent_path, intent, "payload_placeholder")
        _write_reserved_regular(manifest.payload_path, payload_identity, payload)
        _record_phase(intent_path, intent, "payload_written")

        env_identity = _create_placeholder(manifest.env_path, mode=0o600)
        intent["leaves"]["env"] = asdict(env_identity)
        _record_phase(intent_path, intent, "env_placeholder")
        _write_reserved_regular(manifest.env_path, env_identity, env_payload)
        intent["env_fingerprint_sha256"] = env_fingerprint
        _record_phase(intent_path, intent, "env_written")

        os.mkdir(manifest.output_path, mode=0o700)
        _fsync_directory(os.path.dirname(manifest.output_path))
        output_identity = _identity(manifest.output_path)
        intent["leaves"]["output"] = asdict(output_identity)
        _record_phase(intent_path, intent, "output_created")

        marker = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "helper_revision": HELPER_REVISION,
            "manifest_sha256": manifest.digest,
            "reservation_nonce": manifest.reservation_nonce,
            "launch_id": manifest.launch_id,
            "generation": manifest.generation,
        }
        _atomic_json(manifest.ownership_marker_path, marker)
        marker_identity = _identity(manifest.ownership_marker_path)
        intent["leaves"]["ownership_marker"] = asdict(marker_identity)
        _record_phase(intent_path, intent, "marker_written")

        receipt = {
            **marker,
            "payload_sha256": manifest.payload_sha256,
            "env_fingerprint_sha256": env_fingerprint,
            "leaves": intent["leaves"],
            "roots": {
                key: asdict(_identity(value))
                for key, value in manifest.roots.items()
                if key.endswith("_root") and os.path.exists(value)
            },
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        _atomic_json(manifest.reservation_receipt_path, receipt)
        _record_phase(intent_path, intent, "reserved")
        return receipt


def release_reservation_intent(manifest: WorkspaceManifest, *, receipt_sha256: str) -> None:
    intent_path = reservation_intent_path(manifest)
    with helper_lock(manifest):
        receipt = _load_json(manifest.reservation_receipt_path)
        if receipt.get("manifest_sha256") != manifest.digest:
            raise WorkspaceHelperError("reservation receipt manifest mismatch")
        if receipt.get("receipt_sha256") != receipt_sha256:
            raise WorkspaceHelperError("reservation receipt digest mismatch")
        try:
            os.unlink(intent_path)
        except FileNotFoundError:
            pass
        _fsync_directory(os.path.dirname(intent_path))
        if os.path.lexists(intent_path):
            raise WorkspaceHelperError("reservation intent remained after unlink")


def _same_identity(path: str, expected: Mapping[str, Any]) -> bool:
    try:
        actual = _identity(path)
    except WorkspaceHelperError:
        return False
    return actual.device == int(expected["device"]) and actual.inode == int(expected["inode"])


def unlink_attempt_env(manifest: WorkspaceManifest, expected: Mapping[str, Any]) -> dict[str, Any]:
    with helper_lock(manifest):
        if not os.path.lexists(manifest.env_path):
            _fsync_directory(os.path.dirname(manifest.env_path))
            return {"status": "absent", "path": manifest.env_path}
        if not _same_identity(manifest.env_path, expected):
            raise WorkspaceHelperError("attempt env inode no longer matches reservation")
        os.unlink(manifest.env_path)
        _fsync_directory(os.path.dirname(manifest.env_path))
        if os.path.lexists(manifest.env_path):
            raise WorkspaceHelperError("attempt env remained after unlink")
        return {"status": "unlinked", "path": manifest.env_path, "identity": dict(expected)}


def _generation_env_intent_path(
    manifest: WorkspaceManifest, container_generation: int
) -> str:
    if container_generation < 2:
        raise WorkspaceHelperError("recovery container generation must be at least two")
    return os.path.join(
        manifest.roots["protected_metadata_root"],
        f"{manifest.launch_id}.env-intent.g{container_generation}.json",
    )


def reserve_generation_env(
    manifest: WorkspaceManifest,
    *,
    container_generation: int,
    attempt_env: Mapping[str, str],
) -> dict[str, Any]:
    payload = _validate_env(attempt_env)
    fingerprint = hashlib.sha256(payload).hexdigest()
    intent_path = _generation_env_intent_path(manifest, container_generation)
    with helper_lock(manifest):
        if os.path.lexists(intent_path):
            intent = _load_json(intent_path)
            if (
                intent.get("manifest_sha256") != manifest.digest
                or int(intent.get("container_generation") or 0) != container_generation
            ):
                raise WorkspaceHelperError("recovery env intent identity changed")
            return {**intent, "intent_path": intent_path}
        if os.path.lexists(manifest.env_path):
            raise WorkspaceHelperError("an older attempt env generation is still present")
        intent = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "helper_revision": HELPER_REVISION,
            "manifest_sha256": manifest.digest,
            "launch_id": manifest.launch_id,
            "container_generation": int(container_generation),
            "env_fingerprint_sha256": fingerprint,
            "phase": "anchor_written",
        }
        _atomic_json(intent_path, intent)
        identity = _create_placeholder(manifest.env_path, mode=0o600)
        intent["env_identity"] = asdict(identity)
        intent["phase"] = "placeholder_fsynced"
        _atomic_json(intent_path, intent)
        _write_reserved_regular(manifest.env_path, identity, payload)
        intent["phase"] = "reserved"
        intent["receipt_sha256"] = sha256_json(intent)
        _atomic_json(intent_path, intent)
        return {**intent, "intent_path": intent_path}


def release_generation_env_intent(
    manifest: WorkspaceManifest,
    *,
    container_generation: int,
    receipt_sha256: str,
) -> dict[str, Any]:
    intent_path = _generation_env_intent_path(manifest, container_generation)
    with helper_lock(manifest):
        if not os.path.lexists(intent_path):
            _fsync_directory(os.path.dirname(intent_path))
            return {"status": "absent"}
        intent = _load_json(intent_path)
        if _receipt_digest(intent, "receipt_sha256") != receipt_sha256:
            raise WorkspaceHelperError("recovery env receipt digest changed")
        os.unlink(intent_path)
        _fsync_directory(os.path.dirname(intent_path))
        return {"status": "released"}


def recover_reservation_intent(manifest: WorkspaceManifest) -> dict[str, Any]:
    intent_path = reservation_intent_path(manifest)
    with helper_lock(manifest):
        if not os.path.lexists(intent_path):
            return {"status": "absent"}
        intent = _load_json(intent_path)
        if intent.get("manifest_sha256") != manifest.digest:
            raise WorkspaceHelperError("reservation intent manifest mismatch")
        leaves = intent.get("leaves") or {}
        for key, path in (
            ("env", manifest.env_path),
            ("payload", manifest.payload_path),
            ("output", manifest.output_path),
            ("ownership_marker", manifest.ownership_marker_path),
        ):
            expected = leaves.get(key)
            if not expected or not os.path.lexists(path):
                continue
            if not _same_identity(path, expected):
                raise WorkspaceHelperError(f"partial reservation {key} identity changed")
            if key == "output":
                os.rmdir(path)
            else:
                os.unlink(path)
            _fsync_directory(os.path.dirname(path))
        if os.path.lexists(manifest.reservation_receipt_path):
            receipt = _load_json(manifest.reservation_receipt_path)
            if receipt.get("manifest_sha256") != manifest.digest:
                raise WorkspaceHelperError("partial reservation receipt mismatch")
            os.unlink(manifest.reservation_receipt_path)
            _fsync_directory(os.path.dirname(manifest.reservation_receipt_path))
        os.unlink(intent_path)
        _fsync_directory(os.path.dirname(intent_path))
        return {"status": "cleaned", "phase": intent.get("phase")}


def _receipt_digest(document: Mapping[str, Any], field: str) -> str:
    supplied = str(document.get(field) or "")
    unsigned = dict(document)
    unsigned.pop(field, None)
    if supplied != sha256_json(unsigned):
        raise WorkspaceHelperError(f"{field} does not match receipt bytes")
    return supplied


def _prepare_path(manifest: WorkspaceManifest, cleanup_attempt_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", cleanup_attempt_id):
        raise WorkspaceHelperError("cleanup attempt id is not path-safe")
    return os.path.join(
        manifest.roots["protected_metadata_root"],
        f"{manifest.launch_id}.delete-prepare.{cleanup_attempt_id}.json",
    )


def _journal_path(manifest: WorkspaceManifest) -> str:
    return os.path.join(
        manifest.roots["protected_metadata_root"],
        f"{manifest.launch_id}.host-deleted.g{manifest.generation}.json",
    )


def _present_identity(path: str) -> dict[str, Any] | None:
    if not os.path.lexists(path):
        return None
    return asdict(_identity(path))


def delete_prepare(
    manifest: WorkspaceManifest,
    *,
    cleanup_row_id: int,
    cleanup_attempt_id: str,
    control_revision: int,
    cursor_sha256: str,
) -> dict[str, Any]:
    if cleanup_row_id < 1 or control_revision < 1:
        raise WorkspaceHelperError("cleanup row and control revision must be positive")
    path = _prepare_path(manifest, cleanup_attempt_id)
    with helper_lock(manifest):
        reservation = _load_json(manifest.reservation_receipt_path)
        _receipt_digest(reservation, "receipt_sha256")
        if reservation.get("manifest_sha256") != manifest.digest:
            raise WorkspaceHelperError("reservation receipt does not match manifest")
        marker = _load_json(manifest.ownership_marker_path)
        if (
            marker.get("manifest_sha256") != manifest.digest
            or marker.get("reservation_nonce") != manifest.reservation_nonce
        ):
            raise WorkspaceHelperError("ownership marker does not match manifest")
        if os.path.lexists(manifest.env_path):
            raise WorkspaceHelperError("credential env must be absent before delete prepare")
        evidence = {
            "payload": _present_identity(manifest.payload_path),
            "output": _present_identity(manifest.output_path),
            "ownership_marker": _present_identity(manifest.ownership_marker_path),
            "reservation_receipt": _present_identity(manifest.reservation_receipt_path),
        }
        reserved_leaves = reservation.get("leaves") or {}
        for key in ("payload", "output", "ownership_marker"):
            actual = evidence[key]
            expected = reserved_leaves.get(key)
            if actual is not None and (
                not expected
                or actual["device"] != expected["device"]
                or actual["inode"] != expected["inode"]
            ):
                raise WorkspaceHelperError(f"{key} identity changed before delete prepare")
        now_ns = _continuous_ns()
        receipt = {
            "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
            "helper_revision": HELPER_REVISION,
            "machine": manifest.machine,
            "launch_id": manifest.launch_id,
            "generation": manifest.generation,
            "cleanup_row_id": int(cleanup_row_id),
            "cleanup_attempt_id": cleanup_attempt_id,
            "control_revision": int(control_revision),
            "manifest_sha256": manifest.digest,
            "reservation_receipt_sha256": reservation["receipt_sha256"],
            "cursor_sha256": cursor_sha256,
            "challenge": base64.urlsafe_b64encode(os.urandom(24)).decode("ascii").rstrip("="),
            "boot_id": _boot_id(),
            "created_monotonic_ns": now_ns,
            "deadline_monotonic_ns": now_ns + PREPARE_LIFETIME_NS,
            "evidence": evidence,
        }
        receipt["prepare_receipt_sha256"] = sha256_json(receipt)
        _atomic_json(path, receipt)
        return {**receipt, "prepare_receipt_path": path}


def remove_prepare_receipt(
    manifest: WorkspaceManifest,
    *,
    cleanup_attempt_id: str,
    prepare_receipt_sha256: str,
) -> dict[str, Any]:
    path = _prepare_path(manifest, cleanup_attempt_id)
    with helper_lock(manifest):
        if not os.path.lexists(path):
            _fsync_directory(os.path.dirname(path))
            return {"status": "absent"}
        receipt = _load_json(path)
        if _receipt_digest(receipt, "prepare_receipt_sha256") != prepare_receipt_sha256:
            raise WorkspaceHelperError("prepare receipt digest changed")
        os.unlink(path)
        _fsync_directory(os.path.dirname(path))
        return {"status": "unlinked"}


def remove_host_deleted_journal(
    manifest: WorkspaceManifest, *, host_deleted_journal_sha256: str
) -> dict[str, Any]:
    path = _journal_path(manifest)
    with helper_lock(manifest):
        if not os.path.lexists(path):
            _fsync_directory(os.path.dirname(path))
            return {"status": "absent"}
        journal = _load_json(path)
        if (
            _receipt_digest(journal, "host_deleted_journal_sha256")
            != host_deleted_journal_sha256
        ):
            raise WorkspaceHelperError("host-deleted journal digest changed")
        if (
            journal.get("manifest_sha256") != manifest.digest
            or int(journal.get("generation") or 0) != manifest.generation
        ):
            raise WorkspaceHelperError("host-deleted journal belongs to another workspace")
        os.unlink(path)
        _fsync_directory(os.path.dirname(path))
        return {"status": "unlinked", "path": path}


def _assert_no_container_mounts(targets: tuple[str, ...]) -> None:
    listing = subprocess.run(
        ["docker", "ps", "-aq"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if listing.returncode != 0:
        raise WorkspaceHelperError("cannot inventory Docker containers before deletion")
    ids = [value for value in listing.stdout.splitlines() if value.strip()]
    if not ids:
        return
    inspected = subprocess.run(
        ["docker", "inspect", *ids],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if inspected.returncode != 0:
        raise WorkspaceHelperError("cannot inspect Docker containers before deletion")
    try:
        containers = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise WorkspaceHelperError("Docker inspect returned invalid JSON") from exc
    for container in containers:
        for mount in container.get("Mounts") or []:
            source = str(mount.get("Source") or "").rstrip("/")
            if not source:
                continue
            for target in targets:
                exact = target.rstrip("/")
                if (
                    source == exact
                    or source.startswith(exact + "/")
                    or exact.startswith(source + "/")
                ):
                    raise WorkspaceHelperError(
                        f"container {container.get('Id') or '?'} still binds cleanup target"
                    )


def _policy_path(manifest: WorkspaceManifest) -> str:
    return os.path.join(manifest.roots["protected_metadata_root"], ".workspace-key-policy.json")


def _verify_signature(
    manifest: WorkspaceManifest,
    *,
    envelope: CleanupBatchEnvelope,
    signature_base64: str,
) -> None:
    policy = _load_json(_policy_path(manifest))
    _receipt_digest(policy, "policy_sha256")
    keys = policy.get("keys")
    if not isinstance(keys, dict) or envelope.key_revision not in keys:
        raise WorkspaceHelperError("cleanup signing key is not trusted by helper policy")
    key = keys[envelope.key_revision]
    if not isinstance(key, dict) or key.get("state") not in {"active", "overlap"}:
        raise WorkspaceHelperError("cleanup signing key is not active")
    public_key_path = str(key.get("public_key_path") or "")
    public_key_sha256 = str(key.get("public_key_sha256") or "")
    try:
        public_key = Path(public_key_path).read_bytes()
    except OSError as exc:
        raise WorkspaceHelperError("cleanup public key is unavailable") from exc
    if hashlib.sha256(public_key).hexdigest() != public_key_sha256:
        raise WorkspaceHelperError("cleanup public key checksum changed")
    try:
        signature = base64.b64decode(signature_base64, validate=True)
    except ValueError as exc:
        raise WorkspaceHelperError("cleanup signature is not valid base64") from exc
    fd, signature_path = tempfile.mkstemp(prefix="rlab-workspace-signature-")
    try:
        os.write(fd, signature)
        os.close(fd)
        fd = -1
        verified = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-verify",
                public_key_path,
                "-signature",
                signature_path,
            ],
            input=canonical_json_bytes(envelope.as_dict()),
            capture_output=True,
            check=False,
            timeout=5,
        )
        if verified.returncode != 0:
            raise WorkspaceHelperError("cleanup batch signature verification failed")
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(signature_path)
        except FileNotFoundError:
            pass


def _consume_epoch(manifest: WorkspaceManifest, envelope: CleanupBatchEnvelope) -> None:
    path = os.path.join(manifest.roots["protected_metadata_root"], ".workspace-epochs.json")
    epochs: dict[str, Any] = {}
    if os.path.lexists(path):
        epochs = _load_json(path)
    previous = int(epochs.get(envelope.machine) or 0)
    if envelope.epoch <= previous:
        raise WorkspaceHelperError("cleanup batch epoch is stale or replayed")
    epochs[envelope.machine] = envelope.epoch
    _atomic_json(path, epochs)
    persisted = _load_json(path)
    if int(persisted.get(envelope.machine) or 0) != envelope.epoch:
        raise WorkspaceHelperError("cleanup batch epoch did not persist")


class _MutationDeadline(RuntimeError):
    pass


def _check_deadline(deadline_ns: int) -> None:
    if _continuous_ns() >= deadline_ns:
        raise _MutationDeadline("workspace deletion quantum expired")


def _remove_directory_contents(fd: int, *, deadline_ns: int) -> None:
    for entry in list(os.scandir(fd)):
        _check_deadline(deadline_ns)
        if entry.is_dir(follow_symlinks=False):
            child = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=fd,
            )
            try:
                _remove_directory_contents(child, deadline_ns=deadline_ns)
                os.fsync(child)
            finally:
                os.close(child)
            os.rmdir(entry.name, dir_fd=fd)
        else:
            os.unlink(entry.name, dir_fd=fd)
        os.fsync(fd)


def _remove_exact_tree(path: str, expected: Mapping[str, Any], *, deadline_ns: int) -> None:
    if not os.path.lexists(path):
        _fsync_directory(os.path.dirname(path))
        return
    if not _same_identity(path, expected):
        raise WorkspaceHelperError(f"workspace tree identity changed: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _remove_directory_contents(fd, deadline_ns=deadline_ns)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rmdir(path)
    _fsync_directory(os.path.dirname(path))


def _unlink_exact(path: str, expected: Mapping[str, Any] | None) -> None:
    if not os.path.lexists(path):
        _fsync_directory(os.path.dirname(path))
        return
    if expected is not None and not _same_identity(path, expected):
        raise WorkspaceHelperError(f"workspace leaf identity changed: {path}")
    os.unlink(path)
    _fsync_directory(os.path.dirname(path))


def _delete_one(
    manifest: WorkspaceManifest,
    *,
    row: Mapping[str, Any],
    prepare: Mapping[str, Any],
    deadline_ns: int,
) -> dict[str, Any]:
    evidence = prepare.get("evidence") or {}
    _unlink_exact(manifest.payload_path, evidence.get("payload"))
    _unlink_exact(manifest.env_path, None)
    output = evidence.get("output")
    if output is not None:
        _remove_exact_tree(manifest.output_path, output, deadline_ns=deadline_ns)
    elif os.path.lexists(manifest.output_path):
        raise WorkspaceHelperError("output appeared after delete prepare")
    _check_deadline(deadline_ns)
    _unlink_exact(manifest.ownership_marker_path, evidence.get("ownership_marker"))
    _unlink_exact(manifest.reservation_receipt_path, evidence.get("reservation_receipt"))
    for target in manifest.deletion_targets:
        if os.path.lexists(target):
            raise WorkspaceHelperError(f"cleanup target remained after deletion: {target}")
    journal = {
        "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
        "helper_revision": HELPER_REVISION,
        "launch_id": manifest.launch_id,
        "generation": manifest.generation,
        "cleanup_row_id": int(row["row_id"]),
        "cleanup_attempt_id": str(row["cleanup_attempt_id"]),
        "manifest_sha256": manifest.digest,
        "prepare_receipt_sha256": str(row["prepare_receipt_sha256"]),
        "deleted_targets": list(manifest.deletion_targets),
        "boot_id": _boot_id(),
    }
    journal["host_deleted_journal_sha256"] = sha256_json(journal)
    path = _journal_path(manifest)
    _atomic_json(path, journal)
    return {**journal, "host_deleted_journal_path": path}


def delete_commit(
    *,
    envelope: CleanupBatchEnvelope,
    signature_base64: str,
    manifests: Mapping[int, WorkspaceManifest],
) -> dict[str, Any]:
    envelope.validate()
    if _boot_id() != envelope.boot_id:
        raise WorkspaceHelperError("cleanup authority was issued for another host boot")
    now_ns = _continuous_ns()
    if now_ns >= envelope.monotonic_deadline_ns:
        raise WorkspaceHelperError("cleanup authority expired before commit")
    first = manifests.get(envelope.rows[0].row_id)
    if first is None:
        raise WorkspaceHelperError("cleanup manifest set is incomplete")
    if any(manifest.machine != envelope.machine for manifest in manifests.values()):
        raise WorkspaceHelperError("cleanup batch crosses machine identities")
    with helper_lock(first):
        _verify_signature(first, envelope=envelope, signature_base64=signature_base64)
        prepares: dict[int, dict[str, Any]] = {}
        targets: list[str] = []
        for row in envelope.rows:
            manifest = manifests.get(row.row_id)
            if manifest is None or manifest.digest != row.manifest_sha256:
                raise WorkspaceHelperError("cleanup row manifest mismatch")
            path = _prepare_path(manifest, row.cleanup_attempt_id)
            prepare = _load_json(path)
            if _receipt_digest(prepare, "prepare_receipt_sha256") != row.prepare_receipt_sha256:
                raise WorkspaceHelperError("cleanup prepare receipt mismatch")
            if (
                prepare.get("boot_id") != envelope.boot_id
                or int(prepare.get("deadline_monotonic_ns") or 0) < now_ns
                or prepare.get("cursor_sha256") != row.starting_cursor_sha256
                or int(prepare.get("control_revision") or 0) != envelope.control_revision
            ):
                raise WorkspaceHelperError("cleanup prepare authority is stale")
            prepares[row.row_id] = prepare
            targets.extend(manifest.deletion_targets)
        _assert_no_container_mounts(tuple(targets))
        _consume_epoch(first, envelope)
        mutation_deadline = min(
            envelope.monotonic_deadline_ns, _continuous_ns() + MUTATION_LIFETIME_NS
        )
        completed: list[dict[str, Any]] = []
        partial: list[int] = []
        for row in envelope.rows:
            manifest = manifests[row.row_id]
            try:
                completed.append(
                    _delete_one(
                        manifest,
                        row=asdict(row),
                        prepare=prepares[row.row_id],
                        deadline_ns=mutation_deadline,
                    )
                )
            except _MutationDeadline:
                partial.append(row.row_id)
                break
        return {
            "status": "complete" if not partial else "partial",
            "batch_id": envelope.batch_id,
            "epoch": envelope.epoch,
            "completed": completed,
            "partial_row_ids": partial,
        }


def _request() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise WorkspaceHelperError("helper input must be one JSON object") from exc
    if not isinstance(value, dict):
        raise WorkspaceHelperError("helper input must be one JSON object")
    return value


def _manifest(request: Mapping[str, Any]) -> WorkspaceManifest:
    value = request.get("manifest")
    if not isinstance(value, Mapping):
        raise WorkspaceHelperError("helper request manifest must be an object")
    return workspace_manifest_from_mapping(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rlab exact workspace helper")
    parser.add_argument(
        "operation",
        choices=(
            "reserve",
            "release-reservation-intent",
            "recover-intent",
            "unlink-env",
            "reserve-generation-env",
            "release-generation-env-intent",
            "delete-prepare",
            "delete-commit",
            "remove-prepare",
            "remove-host-deleted-journal",
            "drain-zero",
            "install-key-policy",
            "doctor",
        ),
    )
    args = parser.parse_args(argv)
    try:
        request = _request()
        manifest = (
            _manifest(request)
            if args.operation
            not in {"delete-commit", "drain-zero", "install-key-policy", "doctor"}
            else None
        )
        if args.operation == "doctor":
            result = {
                "protocol_version": WORKSPACE_HELPER_PROTOCOL_VERSION,
                "helper_revision": HELPER_REVISION,
                "boot_id": _boot_id(),
                "continuous_ns": _continuous_ns(),
            }
        elif args.operation == "drain-zero":
            manifests_value = request.get("manifests")
            if not isinstance(manifests_value, list):
                raise WorkspaceHelperError("drain-zero manifests must be a list")
            result = drain_zero_receipt(
                protected_metadata_root=str(request.get("protected_metadata_root") or ""),
                manifests=tuple(
                    workspace_manifest_from_mapping(value) for value in manifests_value
                ),
                exact_container_names=tuple(
                    str(value) for value in (request.get("exact_container_names") or ())
                ),
                exact_protected_paths=tuple(
                    str(value) for value in (request.get("exact_protected_paths") or ())
                ),
                receipt_nonce=str(request.get("receipt_nonce") or ""),
            )
        elif args.operation == "install-key-policy":
            result = install_key_policy(
                protected_metadata_root=str(request.get("protected_metadata_root") or ""),
                key_revision=str(request.get("key_revision") or ""),
                public_key=base64.b64decode(
                    str(request.get("public_key_base64") or ""), validate=True
                ),
            )
        elif args.operation == "reserve":
            assert manifest is not None
            payload = base64.b64decode(str(request.get("payload_base64") or ""), validate=True)
            env = request.get("attempt_env")
            if not isinstance(env, Mapping):
                raise WorkspaceHelperError("reserve attempt_env must be an object")
            result = reserve_workspace(manifest, payload=payload, attempt_env=dict(env))
        elif args.operation == "release-reservation-intent":
            assert manifest is not None
            release_reservation_intent(
                manifest, receipt_sha256=str(request.get("receipt_sha256") or "")
            )
            result = {"status": "released"}
        elif args.operation == "recover-intent":
            assert manifest is not None
            result = recover_reservation_intent(manifest)
        elif args.operation == "unlink-env":
            assert manifest is not None
            expected = request.get("expected_identity")
            if not isinstance(expected, Mapping):
                raise WorkspaceHelperError("unlink-env expected_identity must be an object")
            result = unlink_attempt_env(manifest, expected)
        elif args.operation == "reserve-generation-env":
            assert manifest is not None
            env = request.get("attempt_env")
            if not isinstance(env, Mapping):
                raise WorkspaceHelperError(
                    "reserve-generation-env attempt_env must be an object"
                )
            result = reserve_generation_env(
                manifest,
                container_generation=int(request.get("container_generation") or 0),
                attempt_env=dict(env),
            )
        elif args.operation == "release-generation-env-intent":
            assert manifest is not None
            result = release_generation_env_intent(
                manifest,
                container_generation=int(request.get("container_generation") or 0),
                receipt_sha256=str(request.get("receipt_sha256") or ""),
            )
        elif args.operation == "delete-prepare":
            assert manifest is not None
            result = delete_prepare(
                manifest,
                cleanup_row_id=int(request.get("cleanup_row_id") or 0),
                cleanup_attempt_id=str(request.get("cleanup_attempt_id") or ""),
                control_revision=int(request.get("control_revision") or 0),
                cursor_sha256=str(request.get("cursor_sha256") or ""),
            )
        elif args.operation == "remove-prepare":
            assert manifest is not None
            result = remove_prepare_receipt(
                manifest,
                cleanup_attempt_id=str(request.get("cleanup_attempt_id") or ""),
                prepare_receipt_sha256=str(request.get("prepare_receipt_sha256") or ""),
            )
        elif args.operation == "remove-host-deleted-journal":
            assert manifest is not None
            result = remove_host_deleted_journal(
                manifest,
                host_deleted_journal_sha256=str(
                    request.get("host_deleted_journal_sha256") or ""
                ),
            )
        else:
            envelope_value = request.get("envelope")
            manifests_value = request.get("manifests")
            if not isinstance(envelope_value, Mapping) or not isinstance(
                manifests_value, Mapping
            ):
                raise WorkspaceHelperError("delete-commit requires envelope and manifests")
            envelope = cleanup_batch_from_mapping(envelope_value)
            manifests = {
                int(row_id): workspace_manifest_from_mapping(value)
                for row_id, value in manifests_value.items()
            }
            result = delete_commit(
                envelope=envelope,
                signature_base64=str(request.get("signature") or ""),
                manifests=manifests,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "error_type": type(exc).__name__}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
