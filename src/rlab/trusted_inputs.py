from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from rlab.dotenv import load_env_file


MAX_MODEL_FILES = 64
MAX_MODEL_FILE_BYTES = 8 * 1024**3
MAX_MODEL_TOTAL_BYTES = 16 * 1024**3
MAX_METADATA_BYTES = 8 * 1024**2
MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_MEMBER_BYTES = 4 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 12 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 1000
APPROVAL_ENV = "RLAB_MODEL_APPROVAL"
SOURCE_ALLOWLIST_ENV = "RLAB_MODEL_SOURCE_ALLOWLIST"


class ModelApprovalError(PermissionError):
    pass


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass
class StagedModelInput:
    source: str
    root: Path
    model_path: Path
    manifest: tuple[ManifestEntry, ...]
    manifest_hash: str
    source_identity: str | None = None

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> StagedModelInput:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


@dataclass
class ApprovedModelInput:
    staged: StagedModelInput
    approval_hash: str
    internal_execution: str | None = None

    @property
    def model_path(self) -> Path:
        return self.staged.model_path

    @property
    def root(self) -> Path:
        return self.staged.root

    @property
    def manifest_hash(self) -> str:
        return self.staged.manifest_hash

    def verify(self) -> None:
        if self.approval_hash != self.staged.manifest_hash:
            raise ModelApprovalError("model approval hash does not match the staged manifest")
        actual = _manifest_for_staged_root(self.staged.root, self.staged.manifest)
        if actual != self.staged.manifest:
            raise ModelApprovalError("staged model bytes changed after approval")
        if (
            _manifest_hash(actual, source_identity=self.staged.source_identity)
            != self.manifest_hash
        ):
            raise ModelApprovalError("staged model manifest changed after approval")

    def cleanup(self) -> None:
        self.staged.cleanup()

    def __enter__(self) -> ApprovedModelInput:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_hash(manifest: tuple[ManifestEntry, ...], *, source_identity: str | None) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "domain": "rlab.approved-model-input",
                "version": 1,
                "source_identity": source_identity,
                "files": [entry.as_dict() for entry in manifest],
            }
        )
    ).hexdigest()


def _safe_name(value: str) -> str:
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or len(PurePosixPath(value).parts) != 1
        or value in {".", ".."}
    ):
        raise ValueError(f"unsafe model closure path {value!r}")
    return value


def _bounded_json(path: Path) -> Mapping[str, Any] | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata file exceeds {MAX_METADATA_BYTES} bytes: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON metadata before model approval: {path}") from exc
    return value if isinstance(value, Mapping) else None


def _closure_paths(model_path: Path) -> tuple[Path, ...]:
    parent = model_path.parent
    names = {model_path.name}
    versioned_model = model_path.with_suffix(".model.json")
    versioned_recipe = model_path.with_suffix(".recipe.json")
    adjacent_metadata = model_path.with_suffix(".metadata.json")
    if versioned_model.is_file():
        names.add(versioned_model.name)
        if versioned_recipe.is_file():
            names.add(versioned_recipe.name)
    elif model_path.name == "model.zip" and (parent / "model.json").is_file():
        names.add("model.json")
        if (parent / "recipe.json").is_file():
            names.add("recipe.json")
    elif adjacent_metadata.is_file():
        names.add(adjacent_metadata.name)
    release_manifest = parent / "release_manifest.json"
    if release_manifest.is_file():
        names.add(release_manifest.name)
        document = _bounded_json(release_manifest)
        artifacts = document.get("artifacts") if isinstance(document, Mapping) else None
        if isinstance(artifacts, Mapping):
            bound_names = {_safe_name(str(name)) for name in artifacts}
            missing = sorted(name for name in bound_names if not (parent / name).is_file())
            if missing:
                raise ValueError(
                    "release manifest binds missing model closure files: " + ", ".join(missing)
                )
            names.update(bound_names)
    paths = tuple(parent / name for name in sorted(names) if (parent / name).is_file())
    if model_path not in paths:
        raise FileNotFoundError(model_path)
    if len(paths) > MAX_MODEL_FILES:
        raise ValueError(f"model closure exceeds {MAX_MODEL_FILES} files")
    for path in paths:
        if path.suffix == ".json":
            _bounded_json(path)
    return paths


def _copy_regular_file(source: Path, destination: Path) -> ManifestEntry:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(source, flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"model closure entry is not a regular file: {source}")
        if before.st_size > MAX_MODEL_FILE_BYTES:
            raise ValueError(f"model closure file is too large: {source}")
        digest = hashlib.sha256()
        written = 0
        with destination.open("xb") as output:
            os.chmod(destination, 0o600)
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"model closure entry changed while staging: {source}")
        if written != before.st_size:
            raise ValueError(f"model closure entry was truncated while staging: {source}")
        return ManifestEntry(destination.name, written, digest.hexdigest())
    finally:
        os.close(source_fd)


def _preflight_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        return
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"model archive exceeds {MAX_ARCHIVE_MEMBERS} members")
        seen: set[str] = set()
        for info in infos:
            normalized = PurePosixPath(info.filename)
            if (
                normalized.is_absolute()
                or ".." in normalized.parts
                or "\\" in info.filename
                or info.filename in seen
            ):
                raise ValueError(f"unsafe or duplicate model archive member {info.filename!r}")
            seen.add(info.filename)
            if info.flag_bits & 1:
                raise ValueError(f"encrypted model archive member is unsupported: {info.filename}")
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f"model archive member is too large: {info.filename}")
            total += info.file_size
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError("model archive has excessive uncompressed size")
            if info.file_size and (
                info.compress_size == 0
                or info.file_size / info.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO
            ):
                raise ValueError(f"model archive member has excessive compression: {info.filename}")


def stage_model_input(
    model_path: str | Path,
    *,
    source_identity: str | None = None,
) -> StagedModelInput:
    source = Path(model_path).expanduser()
    if source.is_symlink():
        raise ValueError(f"model path must not be a symlink: {source}")
    paths = _closure_paths(source)
    total = sum(path.stat().st_size for path in paths)
    if total > MAX_MODEL_TOTAL_BYTES:
        raise ValueError(f"model closure exceeds {MAX_MODEL_TOTAL_BYTES} bytes")
    stage_root = Path(tempfile.mkdtemp(prefix="rlab-model-stage-"))
    os.chmod(stage_root, 0o700)
    try:
        free = shutil.disk_usage(stage_root).free
        if free < total + 64 * 1024**2:
            raise OSError("insufficient free space for private model staging")
        entries = tuple(_copy_regular_file(path, stage_root / path.name) for path in paths)
        staged_model = stage_root / source.name
        staged_closure = {path.name for path in _closure_paths(staged_model)}
        copied_closure = {entry.path for entry in entries}
        if staged_closure != copied_closure:
            raise ValueError("model closure selection changed while it was being staged")
        _preflight_zip(staged_model)
        manifest = tuple(sorted(entries, key=lambda entry: entry.path))
        return StagedModelInput(
            source=str(source),
            source_identity=source_identity,
            root=stage_root,
            model_path=staged_model,
            manifest=manifest,
            manifest_hash=_manifest_hash(manifest, source_identity=source_identity),
        )
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _manifest_for_staged_root(
    root: Path, expected: tuple[ManifestEntry, ...]
) -> tuple[ManifestEntry, ...]:
    paths = tuple(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ModelApprovalError("staged model closure contains a non-regular entry")
    actual_names = {path.name for path in paths}
    expected_names = {entry.path for entry in expected}
    if actual_names != expected_names:
        raise ModelApprovalError("staged model closure file set changed after approval")
    entries = []
    for entry in expected:
        path = root / entry.path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ModelApprovalError("staged model closure entry is not regular")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ModelApprovalError("staged model closure changed during verification")
            entries.append(ManifestEntry(entry.path, size, digest.hexdigest()))
        finally:
            os.close(descriptor)
    return tuple(entries)


def _source_allowlist_patterns() -> tuple[str, ...]:
    load_env_file(key_filter=lambda key: key == SOURCE_ALLOWLIST_ENV)
    value = str(os.environ.get(SOURCE_ALLOWLIST_ENV, ""))
    return tuple(pattern for item in value.split(",") if (pattern := item.strip()))


def _matching_source_allowlist_pattern(staged: StagedModelInput) -> str | None:
    source = str(staged.source_identity or staged.source).strip()
    return next(
        (pattern for pattern in _source_allowlist_patterns() if fnmatchcase(source, pattern)),
        None,
    )


def approve_staged_model(
    staged: StagedModelInput,
    *,
    expected_hash: str | None = None,
    interactive: bool | None = None,
) -> ApprovedModelInput:
    supplied = str(expected_hash or os.environ.get(APPROVAL_ENV, "")).strip().lower()
    if supplied:
        if supplied != staged.manifest_hash:
            raise ModelApprovalError(
                f"model approval hash mismatch: expected {staged.manifest_hash}, got {supplied}"
            )
    elif _matching_source_allowlist_pattern(staged) is None:
        allow_prompt = sys.stdin.isatty() if interactive is None else bool(interactive)
        if not allow_prompt:
            raise ModelApprovalError(
                "external model approval is required; rerun interactively or set "
                f"{APPROVAL_ENV}={staged.manifest_hash}, or configure "
                f"{SOURCE_ALLOWLIST_ENV} for trusted sources"
            )
        print(
            "External Python model content can execute arbitrary code with your current "
            "operating-system authority, including access to ambient credentials.",
            file=sys.stderr,
        )
        print(f"Source: {staged.source_identity or staged.source}", file=sys.stderr)
        for entry in staged.manifest:
            print(f"  {entry.sha256}  {entry.size_bytes:>10}  {entry.path}", file=sys.stderr)
        print(f"Approval manifest: {staged.manifest_hash}", file=sys.stderr)
        answer = input("Run this exact model closure for this invocation? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise ModelApprovalError("external model execution was not approved")
    approved = ApprovedModelInput(staged=staged, approval_hash=staged.manifest_hash)
    approved.verify()
    return approved


def stage_and_approve_model(
    model_path: str | Path,
    *,
    source_identity: str | None = None,
    expected_hash: str | None = None,
    interactive: bool | None = None,
) -> ApprovedModelInput:
    staged = stage_model_input(model_path, source_identity=source_identity)
    try:
        return approve_staged_model(
            staged,
            expected_hash=expected_hash,
            interactive=interactive,
        )
    except Exception:
        staged.cleanup()
        raise


def approve_internal_model(
    model_path: str | Path,
    *,
    execution_id: str,
) -> ApprovedModelInput:
    if not execution_id.strip():
        raise ValueError("internal model approval requires an execution ID")
    staged = stage_model_input(model_path, source_identity=f"internal:{execution_id}")
    approved = ApprovedModelInput(
        staged=staged,
        approval_hash=staged.manifest_hash,
        internal_execution=execution_id,
    )
    approved.verify()
    return approved
