from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.dataset_contract import (
    feature_identity,
    features_append_compatible,
    grouped_episode_rows,
)


SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@", re.I),
    re.compile(r"\bwandb(?:_|-)?api(?:_|-)?key\s*[:=]\s*['\"]?[A-Za-z0-9]{20,}", re.I),
)
SENSITIVE_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "apikey", "private_key"}
)


@dataclass
class MaterializedHubSource:
    path: Path
    repo_id: str
    revision: str
    _temporary: tempfile.TemporaryDirectory

    def cleanup(self) -> None:
        self._temporary.cleanup()


class HubFeatureGroupsError(ValueError):
    def __init__(self, source: str, groups: Sequence[str]) -> None:
        self.source = source
        self.groups = tuple(groups)
        choices = ", ".join(f"{source}#group={group}" for group in groups)
        super().__init__(f"Hub dataset has incompatible read-only feature groups: {choices}")


def _parse_source(value: str) -> tuple[str, str | None, str | None]:
    text, separator, fragment = str(value).partition("#")
    text = text.removeprefix("hf://").strip().strip("/")
    group = None
    if separator:
        if not fragment.startswith("group=") or not re.fullmatch(
            r"[0-9a-f]{64}", fragment.removeprefix("group=")
        ):
            raise ValueError("Hub dataset fragment must be #group=<64-lowercase-hex-id>")
        group = fragment.removeprefix("group=")
    if "@" in text:
        repo_id, revision = text.rsplit("@", 1)
    else:
        repo_id, revision = text, None
    if len(repo_id.split("/")) != 2 or not all(repo_id.split("/")):
        raise ValueError("Hugging Face dataset source must be hf://owner/repository[@revision]")
    return repo_id, revision, group


def _copy_artifacts(snapshot: Path, target: Path) -> None:
    for name in ("videos", "environments", "collectors"):
        source = snapshot / name
        if source.exists():
            shutil.copytree(source, target / name, dirs_exist_ok=True)


def materialize_hub_source(source: str) -> MaterializedHubSource:
    try:
        from datasets import Dataset, concatenate_datasets
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hub dataset support requires the Rlab dataset extra") from exc
    repo_id, requested_revision, requested_group = _parse_source(source)
    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=requested_revision or "main")
    immutable_revision = str(info.sha or "")
    if not immutable_revision:
        raise RuntimeError(f"could not resolve an immutable revision for {repo_id}")
    temporary = tempfile.TemporaryDirectory(prefix="rlab-hf-dataset-")
    base = Path(temporary.name)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=immutable_revision,
                local_dir=base / "snapshot",
            )
        )
        shards = sorted((snapshot / "data").glob("*.parquet"))
        if not shards:
            raise ValueError(f"Hub dataset {repo_id}@{immutable_revision} has no Parquet shards")
        parts = [Dataset.from_parquet(str(path)) for path in shards]
        groups: list[dict[str, Any]] = []
        for part in parts:
            for group in groups:
                compatible, promoted = features_append_compatible(group["features"], part.features)
                if compatible:
                    group["parts"].append(part)
                    if promoted:
                        group["features"] = part.features
                    break
            else:
                groups.append({"features": part.features, "parts": [part]})
        by_id = {feature_identity(group["features"]): group for group in groups}
        if requested_group is None and len(by_id) > 1:
            raise HubFeatureGroupsError(source, sorted(by_id))
        selected_id = requested_group or next(iter(by_id))
        if selected_id not in by_id:
            raise ValueError(
                f"Hub dataset feature group {selected_id} does not exist; "
                f"available groups: {', '.join(sorted(by_id))}"
            )
        selected_group = by_id[selected_id]
        selected_parts = [
            part
            if part.features == selected_group["features"]
            else part.cast(selected_group["features"])
            for part in selected_group["parts"]
        ]
        dataset = (
            selected_parts[0] if len(selected_parts) == 1 else concatenate_datasets(selected_parts)
        )
        tree = base / "tree"
        dataset.save_to_disk(str(tree))
        _copy_artifacts(snapshot, tree)
        return MaterializedHubSource(tree, repo_id, immutable_revision, temporary)
    except Exception:
        temporary.cleanup()
        raise


def _scan_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in SENSITIVE_KEYS and item not in (None, "", False):
                raise ValueError(f"secret egress gate rejected populated field {path}.{key}")
            _scan_value(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scan_value(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"secret egress gate rejected content at {path}")


def _secret_egress_gate(validation: Any) -> None:
    for episode_id, rows in grouped_episode_rows(validation.dataset).items():
        for index, row in enumerate(rows):
            _scan_value(dict(row), path=f"episode[{episode_id}].row[{index}]")
    for name in ("environments", "collectors"):
        directory = validation.path / name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".json",
                ".md",
                ".txt",
                ".toml",
                ".yaml",
                ".yml",
            }:
                continue
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError(f"secret egress gate rejected oversized text document {path}")
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                _scan_value(json.loads(text), path=str(path))
            else:
                _scan_value(text, path=str(path))


def _remote_head(api: Any, repo_id: str, revision: str) -> tuple[bool, str | None]:
    try:
        info = api.dataset_info(repo_id=repo_id, revision=revision)
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return False, None
        raise
    return True, str(info.sha or "") or None


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for name in ("videos", "environments", "collectors"):
        directory = root / name
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    return hashes


def _immutable_file_operations(
    *, source: Path, known_hashes: Mapping[str, str], operation_type: Any
) -> tuple[list[Any], dict[str, str]]:
    operations = []
    additions = {}
    for name in ("videos", "environments", "collectors"):
        directory = source / name
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            existing = known_hashes.get(relative)
            if existing is not None:
                if existing != digest:
                    raise ValueError(f"immutable Hub artifact conflict at {relative}")
                continue
            operations.append(operation_type(path_in_repo=relative, path_or_fileobj=str(path)))
            additions[relative] = digest
    return operations, additions


class HubAppendSession:
    """One trusted remote inventory followed by expected-parent append commits."""

    def __init__(self, target: str, *, revision: str = "main") -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Hub dataset upload requires the Rlab dataset extra") from exc
        from rlab.dataset_store import validate_tree

        self.target = target
        self.revision = revision
        self.api = HfApi()
        exists, self.head = _remote_head(self.api, target, revision)
        self.features = None
        self.episode_fingerprints: dict[str, str] = {}
        self.known_artifacts: dict[str, str] = {}
        self.sessions: list[Mapping[str, Any]] = []
        if exists:
            if not self.head:
                raise RuntimeError(f"could not resolve trusted remote head for {target}")
            remote = materialize_hub_source(f"hf://{target}@{self.head}")
            try:
                validation = validate_tree(remote.path)
                self.features = validation.dataset.features
                self.episode_fingerprints = dict(validation.episode_fingerprints)
                snapshot = remote.path.parent / "snapshot"
                self.known_artifacts = _artifact_hashes(snapshot)
                sidecar = snapshot / "rlab-sessions.json"
                if sidecar.is_file():
                    document = json.loads(sidecar.read_text(encoding="utf-8"))
                    if document.get("version") != 1 or not isinstance(
                        document.get("rlab_sessions"), list
                    ):
                        raise ValueError("remote Rlab session sidecar is unsupported")
                    self.sessions = list(document["rlab_sessions"])
            finally:
                remote.cleanup()
        else:
            self.api.create_repo(repo_id=target, repo_type="dataset", exist_ok=False)
            _exists, self.head = _remote_head(self.api, target, revision)
            if not self.head:
                raise RuntimeError(f"could not resolve newly created repository head for {target}")

    def append(self, source: Path) -> str:
        try:
            from huggingface_hub import CommitOperationAdd
        except ImportError as exc:
            raise RuntimeError("Hub dataset upload requires the Rlab dataset extra") from exc
        from rlab.dataset_store import validate_tree

        incoming = validate_tree(source)
        _secret_egress_gate(incoming)
        promoted = False
        if self.features is not None:
            compatible, promoted = features_append_compatible(
                self.features, incoming.dataset.features
            )
            if not compatible:
                raise ValueError("remote dataset has incompatible features and is read-only")
        new_episode_ids = []
        for episode_id, fingerprint in incoming.episode_fingerprints.items():
            existing = self.episode_fingerprints.get(episode_id)
            if existing is None:
                new_episode_ids.append(episode_id)
            elif existing != fingerprint:
                raise ValueError(f"remote episode UUID conflict for {episode_id}")
        if not new_episode_ids:
            return str(self.head)
        selected_indices = [
            index
            for index, episode_id in enumerate(incoming.dataset["episode_id"])
            if str(episode_id) in set(new_episode_ids)
        ]
        selected = incoming.dataset.select(selected_indices)
        if self.features is not None and not promoted and selected.features != self.features:
            selected = selected.cast(self.features)
        with tempfile.TemporaryDirectory(prefix="rlab-hf-commit-") as temporary_name:
            temporary = Path(temporary_name)
            shard = temporary / f"rlab-{uuid.uuid4().hex}.parquet"
            selected.to_parquet(str(shard))
            shard_path = f"data/{shard.name}"
            operations = [CommitOperationAdd(path_in_repo=shard_path, path_or_fileobj=str(shard))]
            artifact_operations, artifact_additions = _immutable_file_operations(
                source=incoming.path,
                known_hashes=self.known_artifacts,
                operation_type=CommitOperationAdd,
            )
            operations.extend(artifact_operations)
            sidecar = temporary / "rlab-sessions.json"
            session_entry = {
                "session_id": str(incoming.dataset[0]["session_id"]),
                "parent_sha": self.head,
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "fingerprint": incoming.episode_fingerprints[episode_id],
                    }
                    for episode_id in new_episode_ids
                ],
                "environment_contracts": sorted(incoming.summary.environment_contracts),
                "collector_contracts": sorted(incoming.summary.collector_contracts),
                "representative_episode": new_episode_ids[0],
            }
            sidecar.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "rlab_sessions": [*self.sessions, session_entry],
                    },
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            operations.append(
                CommitOperationAdd(path_in_repo="rlab-sessions.json", path_or_fileobj=str(sidecar))
            )
            try:
                result = self.api.create_commit(
                    repo_id=self.target,
                    repo_type="dataset",
                    revision=self.revision,
                    parent_commit=self.head,
                    operations=operations,
                    commit_message=f"Append {len(new_episode_ids)} verified Rlab episode(s)",
                )
            except Exception:
                recovered_head = self._recover_lost_response(
                    previous_head=str(self.head), incoming=incoming, new_episode_ids=new_episode_ids
                )
                if recovered_head is None:
                    raise
                new_head = recovered_head
            else:
                new_head = str(
                    getattr(result, "oid", None) or getattr(result, "commit_id", None) or ""
                )
                if not new_head:
                    _exists, resolved = _remote_head(self.api, self.target, self.revision)
                    new_head = str(resolved or "")
        if not new_head:
            raise RuntimeError("Hub commit succeeded without an immutable revision")
        self.head = new_head
        self.features = (
            incoming.dataset.features if promoted or self.features is None else self.features
        )
        self.episode_fingerprints.update(
            {
                episode_id: incoming.episode_fingerprints[episode_id]
                for episode_id in new_episode_ids
            }
        )
        self.known_artifacts.update(artifact_additions)
        self.sessions.append(session_entry)
        return new_head

    def _recover_lost_response(
        self, *, previous_head: str, incoming: Any, new_episode_ids: Sequence[str]
    ) -> str | None:
        exists, candidate_head = _remote_head(self.api, self.target, self.revision)
        if not exists or not candidate_head or candidate_head == previous_head:
            return None
        commits = self.api.list_repo_commits(
            repo_id=self.target, repo_type="dataset", revision=self.revision
        )
        latest = commits[0] if commits else None
        parents = set(getattr(latest, "parents", None) or getattr(latest, "parent_ids", None) or ())
        if previous_head not in parents:
            return None
        from rlab.dataset_store import validate_tree

        materialized = materialize_hub_source(f"hf://{self.target}@{candidate_head}")
        try:
            validation = validate_tree(materialized.path)
            expected_ids = set(self.episode_fingerprints) | set(new_episode_ids)
            if set(validation.episode_fingerprints) != expected_ids:
                return None
            for episode_id in new_episode_ids:
                if (
                    validation.episode_fingerprints[episode_id]
                    != incoming.episode_fingerprints[episode_id]
                ):
                    return None
            return candidate_head
        finally:
            materialized.cleanup()


def upload_tree(source: Path, target: str, *, revision: str = "main") -> str:
    return HubAppendSession(target, revision=revision).append(source)


def upload_command(args: Any) -> int:
    from rlab.dataset_store import open_source

    with open_source(args.source, root=args.root) as loaded:
        revision = upload_tree(loaded.path, args.target, revision=args.revision)
    print(f"uploaded to hf://{args.target}@{revision}")
    return 0
