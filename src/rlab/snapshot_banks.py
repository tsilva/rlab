from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


MAX_ARCHIVE_BYTES = 64 * 1024**2
MAX_ARCHIVE_MEMBERS = 512
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024**2
MAX_STATE_BYTES = 4 * 1024**2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SNAPSHOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True)
class BreakoutSnapshotBank:
    uri: str
    archive_sha256: str
    manifest: Mapping[str, Any]
    state_ids: tuple[str, ...]
    states: Mapping[str, bytes]
    observation_sha256: Mapping[str, str]


def _read_archive(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("snapshot bank archive is too large")
        return path.read_bytes()
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        request = Request(uri, headers={"Accept": "application/octet-stream"})
        with urlopen(request, timeout=120) as response:
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None and int(raw_length) > MAX_ARCHIVE_BYTES:
                raise ValueError("snapshot bank archive is too large")
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise ValueError("snapshot bank archive is too large")
        return payload
    raise ValueError("snapshot_bank_uri must be an https:// or file:// locator")


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe snapshot archive member: {name!r}")
    return path


def _archive_files(payload: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.TarError as exc:
        raise ValueError("snapshot bank is not a readable tar archive") from exc
    with archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("snapshot bank archive contains too many members")
        for member in members:
            path = _safe_member_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"snapshot archive member is not a regular file: {member.name}")
            if member.size < 0 or member.size > MAX_STATE_BYTES:
                raise ValueError(f"snapshot archive member has invalid size: {member.name}")
            total += int(member.size)
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError("snapshot bank archive expands beyond the allowed size")
            normalized = str(path)
            if normalized in files:
                raise ValueError(f"duplicate snapshot archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read snapshot archive member: {member.name}")
            value = source.read(MAX_STATE_BYTES + 1)
            if len(value) != member.size:
                raise ValueError(f"truncated snapshot archive member: {member.name}")
            files[normalized] = value
    return files


def _manifest_file(files: Mapping[str, bytes]) -> tuple[str, Mapping[str, Any]]:
    candidates = sorted(name for name in files if PurePosixPath(name).name == "manifest.json")
    if len(candidates) != 1:
        raise ValueError("snapshot bank archive must contain exactly one manifest.json")
    name = candidates[0]
    try:
        value = json.loads(files[name])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("snapshot bank manifest is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("snapshot bank manifest must contain a JSON object")
    return name, value


def _manifest_member(manifest_name: str, relative_name: str) -> str:
    relative = _safe_member_name(relative_name)
    if relative.is_absolute():
        raise ValueError("snapshot state file must be relative to the manifest")
    return str(PurePosixPath(manifest_name).parent / relative)


def load_breakout_snapshot_bank(uri: str, expected_sha256: str) -> BreakoutSnapshotBank:
    uri = str(uri or "").strip()
    expected_sha256 = str(expected_sha256 or "").strip().lower()
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("snapshot_bank_sha256 must contain 64 lowercase hexadecimal characters")
    payload = _read_archive(uri)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"snapshot bank archive hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    files = _archive_files(payload)
    manifest_name, manifest = _manifest_file(files)
    if manifest.get("document_type") != "rlab.breakout_snapshot_bank":
        raise ValueError("snapshot bank manifest has an unsupported document_type")
    if manifest.get("format_version") != 1:
        raise ValueError("snapshot bank manifest has an unsupported format_version")
    raw_snapshots = manifest.get("snapshots")
    if not isinstance(raw_snapshots, list) or not raw_snapshots:
        raise ValueError("snapshot bank manifest must declare at least one snapshot")
    states: dict[str, bytes] = {}
    observation_sha256: dict[str, str] = {}
    for entry in raw_snapshots:
        if not isinstance(entry, Mapping):
            raise ValueError("snapshot bank entries must be objects")
        state_id = str(entry.get("id") or "")
        if not _SNAPSHOT_ID.fullmatch(state_id):
            raise ValueError(f"invalid snapshot id: {state_id!r}")
        if state_id in states:
            raise ValueError(f"duplicate snapshot id: {state_id!r}")
        state_file = str(entry.get("state_file") or "")
        member_name = _manifest_member(manifest_name, state_file)
        try:
            state = files[member_name]
        except KeyError as exc:
            raise ValueError(f"snapshot state file is missing: {state_file!r}") from exc
        expected_state_sha = str(entry.get("state_sha256") or "").lower()
        if not _SHA256.fullmatch(expected_state_sha):
            raise ValueError(f"snapshot {state_id!r} has an invalid state_sha256")
        if hashlib.sha256(state).hexdigest() != expected_state_sha:
            raise ValueError(f"snapshot {state_id!r} state hash mismatch")
        if int(entry.get("state_size_bytes") or -1) != len(state):
            raise ValueError(f"snapshot {state_id!r} state size mismatch")
        if not state.startswith(b"BTO1"):
            raise ValueError(f"snapshot {state_id!r} is not a Breakout Turbo state")
        expected_observation_sha = str(entry.get("observation_sha256") or "").lower()
        if not _SHA256.fullmatch(expected_observation_sha):
            raise ValueError(f"snapshot {state_id!r} has an invalid observation_sha256")
        states[state_id] = state
        observation_sha256[state_id] = expected_observation_sha
    if len(set(observation_sha256.values())) != len(observation_sha256):
        raise ValueError("snapshot bank contains duplicate policy observation stacks")
    return BreakoutSnapshotBank(
        uri=uri,
        archive_sha256=actual_sha256,
        manifest=dict(manifest),
        state_ids=tuple(states),
        states=states,
        observation_sha256=observation_sha256,
    )


def validate_breakout_snapshot_environment(
    bank: BreakoutSnapshotBank,
    *,
    game: str,
    frame_skip: int,
    frame_stack: int,
    observation_size: int,
    obs_crop: tuple[int, int, int, int] | None,
    obs_crop_mode: str,
    obs_crop_fill: int,
    obs_resize_algorithm: str,
    sticky_action_prob: float,
) -> None:
    environment = bank.manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("snapshot bank manifest is missing its environment contract")
    expected = {
        "game": game,
        "provider": "breakout-turbo-env",
        "frame_skip": int(frame_skip),
        "frame_stack": int(frame_stack),
        "observation_size": int(observation_size),
        "obs_crop": list(obs_crop or (0, 0, 0, 0)),
        "obs_crop_mode": obs_crop_mode,
        "obs_crop_fill": int(obs_crop_fill),
        "obs_resize_algorithm": obs_resize_algorithm,
        "obs_grayscale": True,
        "obs_layout": "chw",
        "sticky_action_prob": float(sticky_action_prob),
        "reward_clip": False,
        "action_meanings": ["noop", "button", "right", "left"],
    }
    mismatches = {
        key: (environment.get(key), value)
        for key, value in expected.items()
        if environment.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"snapshot bank environment contract mismatch: {details}")
