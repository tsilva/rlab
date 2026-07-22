from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from rlab.dataset_contract import (
    STORAGE_FORMAT_LOSSLESS_VIDEO,
    collection_fingerprint,
    episode_content_fingerprint,
    feature_identity,
    features_append_compatible,
    grouped_episode_rows,
    validate_contract_artifacts,
    validate_v3,
)
from rlab.dataset_media import iter_episode_frames, iter_selected_frames


COLLECTION_IDENTITY_VERSION = 1
TRANSACTION_VERSION = 1
DEFAULT_DATASET_ROOT = Path.home() / ".rlab" / "datasets"
MAX_STAGED_FILES = 100_000
MAX_STAGED_FILE_BYTES = 16 * 1024**3
MAX_STAGED_TOTAL_BYTES = 64 * 1024**3


def _datasets_module():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "dataset support is not installed; run ./install.sh --extra dataset"
        ) from exc
    return datasets


def dataset_root(value: Path | None) -> Path:
    return (value or DEFAULT_DATASET_ROOT).expanduser().resolve()


def _reference_key(reference: str) -> str:
    text = str(reference).strip()
    if not text or "\x00" in text:
        raise ValueError("dataset reference must be non-empty")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CollectionPaths:
    reference: str
    directory: Path

    @property
    def identity(self) -> Path:
        return self.directory / "identity.json"

    @property
    def lock(self) -> Path:
        return self.directory / ".lock"

    @property
    def current(self) -> Path:
        return self.directory / "current"

    @property
    def next(self) -> Path:
        return self.directory / "next"

    @property
    def backup(self) -> Path:
        return self.directory / "backup"

    @property
    def phase(self) -> Path:
        return self.directory / "transaction.json"

    @property
    def quarantine(self) -> Path:
        return self.directory / "quarantine"


def collection_paths(reference: str, root: Path | None = None) -> CollectionPaths:
    base = dataset_root(root)
    return CollectionPaths(reference, base / "collections" / _reference_key(reference))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _collection_lock(paths: CollectionPaths, *, exclusive: bool) -> Iterator[None]:
    paths.directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON document at {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _ensure_identity(paths: CollectionPaths) -> None:
    expected = {
        "version": COLLECTION_IDENTITY_VERSION,
        "reference": paths.reference,
        "key": paths.directory.name,
    }
    if paths.identity.exists():
        if _read_json(paths.identity) != expected:
            raise ValueError(f"dataset collection identity mismatch at {paths.directory}")
    else:
        _atomic_json(paths.identity, expected)


def _check_identity(paths: CollectionPaths) -> None:
    expected = {
        "version": COLLECTION_IDENTITY_VERSION,
        "reference": paths.reference,
        "key": paths.directory.name,
    }
    if not paths.identity.is_file() or _read_json(paths.identity) != expected:
        raise ValueError(f"dataset collection identity mismatch at {paths.directory}")


def _load_arrow_tree(path: Path):
    datasets = _datasets_module()
    if not path.is_dir() or not (path / "dataset_info.json").is_file():
        raise ValueError(f"not a Hugging Face save_to_disk dataset tree: {path}")
    dataset = datasets.load_from_disk(str(path), keep_in_memory=False)
    if "observations" in dataset.column_names:
        dataset = dataset.cast_column("observations", datasets.Image(decode=False))
    return dataset


@dataclass(frozen=True)
class TreeValidation:
    path: Path
    dataset: Any
    summary: Any
    feature_id: str
    episode_fingerprints: Mapping[str, str]
    collection_fingerprint: str


def _reject_unsafe_tree_entries(path: Path) -> None:
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"dataset tree contains a symbolic link: {entry}")
        if not entry.is_file() and not entry.is_dir():
            raise ValueError(f"dataset tree contains a non-regular entry: {entry}")


@dataclass
class StagedDatasetTree:
    path: Path
    _temporary: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        self._temporary.cleanup()


def stage_dataset_tree(source: Path) -> StagedDatasetTree:
    source = source.expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"dataset source must be a regular directory: {source}")
    entries = sorted(source.rglob("*"))
    files = [path for path in entries if not path.is_dir()]
    if len(files) > MAX_STAGED_FILES:
        raise ValueError(f"dataset source exceeds {MAX_STAGED_FILES} files")
    for path in entries:
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError(f"dataset source contains an unsafe entry: {path}")
    total = sum(path.stat().st_size for path in files)
    if total > MAX_STAGED_TOTAL_BYTES:
        raise ValueError(f"dataset source exceeds {MAX_STAGED_TOTAL_BYTES} bytes")
    temporary = tempfile.TemporaryDirectory(prefix="rlab-dataset-stage-")
    root = Path(temporary.name)
    os.chmod(root, 0o700)
    staged = root / "tree"
    staged.mkdir(mode=0o700)
    try:
        if shutil.disk_usage(root).free < total + 64 * 1024**2:
            raise OSError("insufficient free space for private dataset staging")
        for directory in (path for path in entries if path.is_dir()):
            (staged / directory.relative_to(source)).mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in files:
            destination = staged / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _stage_regular_file(path, destination)
        _fsync_tree(staged)
        return StagedDatasetTree(staged, temporary)
    except Exception:
        temporary.cleanup()
        raise


def _stage_regular_file(source: Path, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_STAGED_FILE_BYTES:
            raise ValueError(f"unsafe or oversized dataset source file: {source}")
        written = 0
        with destination.open("xb") as output:
            os.chmod(destination, 0o600)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

        if written != before.st_size or identity(before) != identity(after):
            raise ValueError(f"dataset source changed while staging: {source}")
    finally:
        os.close(descriptor)


def validate_tree(path: Path) -> TreeValidation:
    if path.expanduser().is_symlink():
        raise ValueError(f"dataset tree must not be a symbolic link: {path}")
    path = path.expanduser().resolve()
    _reject_unsafe_tree_entries(path)
    dataset = _load_arrow_tree(path)
    summary = validate_v3(dataset, label=str(path))
    documents = validate_contract_artifacts(path, summary)
    fingerprints: dict[str, str] = {}
    for episode_id, rows in grouped_episode_rows(dataset).items():
        frames = iter(iter_episode_frames(rows, root=path))

        def next_frame(_row: Mapping[str, Any]) -> Any:
            return next(frames)

        environment_id = str(rows[0]["environment_contract_id"])
        collector_id = rows[0].get("collector_contract_id")
        relevant_documents = {
            name: payload
            for name, payload in documents.items()
            if environment_id in name or (collector_id is not None and str(collector_id) in name)
        }
        fingerprints[episode_id] = episode_content_fingerprint(
            rows,
            frame_loader=next_frame,
            contract_documents=relevant_documents,
        )
        try:
            next(frames)
        except StopIteration:
            pass
        else:
            raise ValueError(f"episode {episode_id} has unbound media frames")
    fingerprint = collection_fingerprint(dataset.features, fingerprints.values())
    return TreeValidation(
        path=path,
        dataset=dataset,
        summary=summary,
        feature_id=feature_identity(dataset.features),
        episode_fingerprints=fingerprints,
        collection_fingerprint=fingerprint,
    )


def _fsync_tree(path: Path) -> None:
    directories = [path]
    for entry in path.rglob("*"):
        if entry.is_file():
            _fsync_file(entry)
        elif entry.is_dir():
            directories.append(entry)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _write_phase(paths: CollectionPaths, document: Mapping[str, Any], phase: str) -> None:
    updated = dict(document)
    updated["phase"] = phase
    _atomic_json(paths.phase, updated)


def _quarantine_transaction(paths: CollectionPaths, reason: str) -> None:
    transaction_id = "unknown"
    if paths.phase.exists():
        with contextlib.suppress(Exception):
            transaction_id = str(_read_json(paths.phase).get("transaction_id") or "unknown")
    destination = paths.quarantine / f"{transaction_id}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for candidate in (paths.current, paths.next, paths.backup, paths.phase):
        if candidate.exists():
            os.replace(candidate, destination / candidate.name)
            moved.append(candidate.name)
    _atomic_json(
        destination / "report.json",
        {"version": 1, "reason": reason, "moved": moved, "transaction_id": transaction_id},
    )
    _fsync_directory(destination)
    _fsync_directory(paths.directory)


def _recover_locked(paths: CollectionPaths) -> None:
    if not paths.phase.exists():
        if paths.next.exists() or paths.backup.exists():
            _quarantine_transaction(paths, "orphan transaction trees without a phase marker")
            raise ValueError(f"quarantined an ambiguous dataset transaction at {paths.directory}")
        return
    try:
        document = _read_json(paths.phase)
        if document.get("version") != TRANSACTION_VERSION:
            raise ValueError("unsupported transaction version")
        phase = str(document.get("phase") or "")
        had_current = document.get("had_current")
        if not isinstance(had_current, bool):
            raise ValueError("transaction had_current must be boolean")

        if phase == "prepared":
            if (
                had_current
                and paths.current.exists()
                and paths.next.exists()
                and not paths.backup.exists()
            ):
                os.replace(paths.current, paths.backup)
                _fsync_directory(paths.directory)
                _write_phase(paths, document, "backup_moved")
                phase = "backup_moved"
            elif (
                had_current
                and not paths.current.exists()
                and paths.next.exists()
                and paths.backup.exists()
            ):
                _write_phase(paths, document, "backup_moved")
                phase = "backup_moved"
            elif (
                not had_current
                and not paths.current.exists()
                and paths.next.exists()
                and not paths.backup.exists()
            ):
                os.replace(paths.next, paths.current)
                _fsync_directory(paths.directory)
                _write_phase(paths, document, "current_installed")
                phase = "current_installed"
            elif (
                not had_current
                and paths.current.exists()
                and not paths.next.exists()
                and not paths.backup.exists()
            ):
                _write_phase(paths, document, "current_installed")
                phase = "current_installed"
            else:
                raise ValueError("impossible prepared transaction layout")

        if phase == "backup_moved":
            if (
                had_current
                and not paths.current.exists()
                and paths.next.exists()
                and paths.backup.exists()
            ):
                os.replace(paths.next, paths.current)
                _fsync_directory(paths.directory)
                _write_phase(paths, document, "current_installed")
                phase = "current_installed"
            elif (
                had_current
                and paths.current.exists()
                and not paths.next.exists()
                and paths.backup.exists()
            ):
                _write_phase(paths, document, "current_installed")
                phase = "current_installed"
            else:
                raise ValueError("impossible backup_moved transaction layout")

        if phase == "current_installed":
            if (
                not paths.current.exists()
                or paths.next.exists()
                or (had_current != paths.backup.exists())
            ):
                raise ValueError("impossible current_installed transaction layout")
            validation = validate_tree(paths.current)
            if validation.collection_fingerprint != document.get("new_fingerprint"):
                raise ValueError("installed current fingerprint does not match the transaction")
            _write_phase(paths, document, "current_validated")
            phase = "current_validated"

        if phase == "current_validated":
            if not paths.current.exists() or paths.next.exists():
                raise ValueError("impossible current_validated transaction layout")
            if paths.backup.exists():
                shutil.rmtree(paths.backup)
                _fsync_directory(paths.directory)
            paths.phase.unlink()
            _fsync_directory(paths.directory)
            return
        raise ValueError(f"unknown transaction phase {phase!r}")
    except Exception as exc:
        _quarantine_transaction(paths, str(exc))
        raise ValueError(
            f"quarantined an invalid dataset transaction at {paths.directory}"
        ) from exc


def _install_next_locked(
    paths: CollectionPaths,
    *,
    old_fingerprint: str | None,
    new_fingerprint: str,
) -> None:
    if paths.phase.exists() or paths.backup.exists():
        raise RuntimeError("dataset transaction state was not recovered")
    if not paths.next.exists():
        raise RuntimeError("dataset transaction has no next tree")
    had_current = paths.current.exists()
    document = {
        "version": TRANSACTION_VERSION,
        "transaction_id": str(uuid.uuid4()),
        "had_current": had_current,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "phase": "prepared",
    }
    _fsync_tree(paths.next)
    validate_tree(paths.next)
    _write_phase(paths, document, "prepared")
    _recover_locked(paths)


def _copy_artifact_directories(source: Path, target: Path) -> None:
    for name in ("videos", "environments", "collectors"):
        source_directory = source / name
        if source_directory.exists():
            target_directory = target / name
            for source_file in source_directory.rglob("*"):
                if source_file.is_dir():
                    continue
                relative = source_file.relative_to(source_directory)
                target_file = target_directory / relative
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if target_file.exists():
                    if not target_file.is_file() or not _files_equal(source_file, target_file):
                        raise ValueError(
                            f"immutable dataset artifact conflict at {name}/{relative}"
                        )
                    continue
                shutil.copy2(source_file, target_file)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _save_dataset_tree(dataset: Any, target: Path, artifact_sources: Sequence[Path]) -> None:
    if target.exists():
        shutil.rmtree(target)
    dataset.save_to_disk(str(target))
    for source in artifact_sources:
        _copy_artifact_directories(source, target)


def _merge_datasets(existing: TreeValidation | None, incoming: TreeValidation):
    if existing is None:
        return incoming.dataset, [incoming.path]
    compatible, promoted = features_append_compatible(
        existing.dataset.features, incoming.dataset.features
    )
    if not compatible:
        raise ValueError(
            "incoming dataset features are incompatible with the collection; use another reference"
        )
    existing_ids = set(existing.episode_fingerprints)
    new_indices: list[int] = []
    for index, episode_id in enumerate(incoming.dataset["episode_id"]):
        normalized = str(episode_id)
        if normalized in existing_ids:
            if (
                incoming.episode_fingerprints[normalized]
                != existing.episode_fingerprints[normalized]
            ):
                raise ValueError(f"episode UUID conflict for {normalized}")
        else:
            new_indices.append(index)
    if not new_indices:
        return existing.dataset, [existing.path]
    datasets = _datasets_module()
    selected = incoming.dataset.select(new_indices)
    if promoted:
        existing_dataset = existing.dataset.cast(incoming.dataset.features)
    elif existing.dataset.features != selected.features:
        selected = selected.cast(existing.dataset.features)
        existing_dataset = existing.dataset
    else:
        existing_dataset = existing.dataset
    return datasets.concatenate_datasets([existing_dataset, selected]), [
        existing.path,
        incoming.path,
    ]


def adopt_tree(source: Path, reference: str, *, root: Path | None = None) -> TreeValidation:
    staged = stage_dataset_tree(source)
    try:
        incoming = validate_tree(staged.path)
        paths = collection_paths(reference, root)
        with _collection_lock(paths, exclusive=True):
            _ensure_identity(paths)
            _recover_locked(paths)
            existing = validate_tree(paths.current) if paths.current.exists() else None
            merged, artifact_sources = _merge_datasets(existing, incoming)
            if existing is not None and merged is existing.dataset:
                return existing
            _save_dataset_tree(merged, paths.next, artifact_sources)
            new_validation = validate_tree(paths.next)
            _install_next_locked(
                paths,
                old_fingerprint=existing.collection_fingerprint if existing else None,
                new_fingerprint=new_validation.collection_fingerprint,
            )
            return validate_tree(paths.current)
    finally:
        staged.cleanup()


def combine_trees(sources: Sequence[Path], target: Path) -> TreeValidation:
    """Build one validated tree from compatible episode packages without mutating them."""

    if not sources:
        raise ValueError("cannot combine an empty dataset package list")
    validations = [validate_tree(source) for source in sources]
    feature_id = validations[0].feature_id
    storage_format = validations[0].summary.storage_format
    seen: dict[str, str] = {}
    for validation in validations:
        if (
            validation.feature_id != feature_id
            or validation.summary.storage_format != storage_format
        ):
            raise ValueError("recording session packages have incompatible physical features")
        for episode_id, fingerprint in validation.episode_fingerprints.items():
            if episode_id in seen:
                raise ValueError(f"duplicate episode UUID in recording session: {episode_id}")
            seen[episode_id] = fingerprint
    datasets = _datasets_module()
    combined = datasets.concatenate_datasets([value.dataset for value in validations])
    _save_dataset_tree(combined, target, [value.path for value in validations])
    return validate_tree(target)


class LoadedDataset:
    def __init__(
        self,
        validation: TreeValidation,
        lock_context: Any | None = None,
        cleanup: Any | None = None,
    ) -> None:
        self.validation = validation
        self._lock_context = lock_context
        self._cleanup = cleanup

    @property
    def dataset(self):
        return self.validation.dataset

    @property
    def path(self) -> Path:
        return self.validation.path

    def close(self) -> None:
        if self._lock_context is not None:
            self._lock_context.__exit__(None, None, None)
            self._lock_context = None
        if self._cleanup is not None:
            self._cleanup()
            self._cleanup = None

    def __enter__(self) -> LoadedDataset:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _explicit_source_path(source: str) -> Path | None:
    candidate = Path(source).expanduser()
    if candidate.exists():
        if candidate.is_symlink():
            raise ValueError(f"dataset source must not be a symbolic link: {candidate}")
        return candidate.resolve()
    return None


def open_source(source: str, *, root: Path | None = None) -> LoadedDataset:
    explicit = _explicit_source_path(source)
    if explicit is not None:
        staged = stage_dataset_tree(explicit)
        try:
            return LoadedDataset(validate_tree(staged.path), cleanup=staged.cleanup)
        except Exception:
            staged.cleanup()
            raise
    if source.startswith("hf://"):
        from rlab.dataset_hub import materialize_hub_source

        materialized = materialize_hub_source(source)
        try:
            return LoadedDataset(validate_tree(materialized.path), cleanup=materialized.cleanup)
        except Exception:
            materialized.cleanup()
            raise
    paths = collection_paths(source, root)
    lock_context = _collection_lock(paths, exclusive=False)
    lock_context.__enter__()
    try:
        if paths.phase.exists() or paths.next.exists() or paths.backup.exists():
            lock_context.__exit__(None, None, None)
            with _collection_lock(paths, exclusive=True):
                _ensure_identity(paths)
                _recover_locked(paths)
            lock_context = _collection_lock(paths, exclusive=False)
            lock_context.__enter__()
        _check_identity(paths)
        if not paths.current.exists():
            raise FileNotFoundError(f"no local dataset collection named {source!r}")
        return LoadedDataset(validate_tree(paths.current), lock_context)
    except Exception:
        lock_context.__exit__(None, None, None)
        raise


def preflight_recording_reference(reference: str, *, root: Path | None = None) -> None:
    """Reject a physical collection mismatch before model or provider construction."""
    paths = collection_paths(reference, root)
    if not paths.directory.exists():
        return
    with _collection_lock(paths, exclusive=True):
        _ensure_identity(paths)
        _recover_locked(paths)
        if not paths.current.exists():
            return
        validation = validate_tree(paths.current)
        if validation.summary.storage_format != STORAGE_FORMAT_LOSSLESS_VIDEO:
            raise ValueError(
                f"recording reference {reference!r} uses "
                f"{validation.summary.storage_format!r}; use a separate reference for "
                f"{STORAGE_FORMAT_LOSSLESS_VIDEO!r} recording"
            )


def _environment_fps(validation: TreeValidation, rows: Sequence[Mapping[str, Any]]) -> float:
    contract_id = str(rows[0]["environment_contract_id"])
    document = _read_json(validation.path / "environments" / contract_id / "environment.json")
    return float(document["fps"])


def list_command(args: Any) -> int:
    root = dataset_root(args.root)
    if args.reference:
        try:
            loaded_context = open_source(args.reference, root=root)
        except Exception as exc:
            from rlab.dataset_hub import HubFeatureGroupsError

            if not isinstance(exc, HubFeatureGroupsError):
                raise
            print(f"reference: {args.reference}")
            print("read-only incompatible feature groups:")
            for group in exc.groups:
                print(f"  {args.reference}#group={group}")
            return 0
        with loaded_context as loaded:
            print(f"reference: {args.reference}")
            print(f"episodes: {loaded.validation.summary.episodes}")
            print(f"transitions: {loaded.validation.summary.transitions}")
            print(f"storage: {loaded.validation.summary.storage_format}")
            for index, (episode_id, rows) in enumerate(
                grouped_episode_rows(loaded.dataset).items(), start=1
            ):
                print(f"{index}: {episode_id} transitions={len(rows) - 1}")
        return 0
    collections = root / "collections"
    if not collections.exists():
        return 0
    for directory in sorted(path for path in collections.iterdir() if path.is_dir()):
        identity = directory / "identity.json"
        if identity.is_file():
            print(str(_read_json(identity).get("reference")))
    return 0


def verify_command(args: Any) -> int:
    with open_source(args.source, root=args.root) as loaded:
        if args.reexecute:
            from rlab.dataset_record import reexecute_dataset

            reexecute_dataset(loaded.validation)
        summary = loaded.validation.summary
        print(
            f"verified Gymrec v3: {summary.episodes} episodes, "
            f"{summary.transitions} transitions, {summary.storage_format}"
        )
        print(f"fingerprint: {loaded.validation.collection_fingerprint}")
    return 0


def video_command(args: Any) -> int:
    from rlab.video import write_video

    with open_source(args.source, root=args.root) as loaded:
        episodes = list(grouped_episode_rows(loaded.dataset).values())
        first = int(args.first) - 1
        last = int(args.last) if args.last is not None else len(episodes)
        selected = episodes[first:last]
        if not selected:
            raise ValueError("episode selection is empty")
        fps = float(args.fps or _environment_fps(loaded.validation, selected[0]))
        write_video(
            iter_selected_frames(selected, root=loaded.path),
            args.output.expanduser().resolve(),
            fps,
            int(args.scale),
        )
        print(f"wrote {args.output}")
    return 0


def play_command(args: Any) -> int:
    with open_source(args.source, root=args.root) as loaded:
        episodes = list(grouped_episode_rows(loaded.dataset).values())
        index = int(args.episode) - 1
        if index >= len(episodes):
            raise ValueError(f"episode {args.episode} does not exist")
        rows = episodes[index]
        frames = iter(iter_episode_frames(rows, root=loaded.path))
        first = next(frames)
        viewer = _DatasetViewer(first.shape, int(args.scale))
        fps = float(args.fps or _environment_fps(loaded.validation, rows))
        delay = 1.0 / fps
        try:
            for frame in chain((first,), frames):
                started = time.monotonic()
                if not viewer.show(frame):
                    break
                time.sleep(max(0.0, delay - (time.monotonic() - started)))
        finally:
            viewer.close()
    return 0


class _DatasetViewer:
    """Small provider-free viewer for already-decoded dataset RGB frames."""

    def __init__(self, frame_shape: tuple[int, int, int], scale: int) -> None:
        if scale < 1:
            raise ValueError("viewer scale must be >= 1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pygame = pygame
        height, width, channels = frame_shape
        if channels != 3:
            raise ValueError(f"dataset playback requires RGB frames, got {frame_shape}")
        self.size = (width * scale, height * scale)
        pygame.init()
        pygame.display.set_caption("rlab dataset")
        self.screen = pygame.display.set_mode(self.size)

    def show(self, frame: Any) -> bool:
        for event in self.pygame.event.get():
            if event.type in {self.pygame.QUIT, self.pygame.WINDOWCLOSE}:
                return False
            if event.type == self.pygame.KEYDOWN and getattr(event, "key", None) in {
                self.pygame.K_ESCAPE,
                self.pygame.K_q,
            }:
                return False
        surface = self.pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(self.pygame.transform.scale(surface, self.size), (0, 0))
        self.pygame.display.flip()
        return True

    def close(self) -> None:
        self.pygame.quit()


def adopt_command(args: Any) -> int:
    result = adopt_tree(args.source, args.reference, root=args.root)
    print(
        f"adopted {result.summary.episodes} episodes as {args.reference!r}; "
        f"fingerprint={result.collection_fingerprint}"
    )
    return 0
