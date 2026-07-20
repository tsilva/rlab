from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rlab.modal_eval_storage import ObjectNotFound, ObjectStore, file_sha256, object_store_base_uri
from rlab.modal_eval_storage import write_downloaded_file


ROM_ASSET_SCHEMA_VERSION = 2
ROM_ASSET_STATE_SCHEMA_VERSION = 1
ROM_ASSET_IDENTITY_ALGORITHM = "sha1-provider-body-v1"
ROM_ASSET_PREFIX = "rom-assets/v2"
DEFAULT_LOCAL_ROM_CACHE = Path("~/.cache/rlab/roms").expanduser()
CONTAINER_ROM_CACHE = Path("/rom-cache")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "game",
        "filename",
        "size_bytes",
        "sha256",
        "object_uri",
        "provider_rom_identity",
        "provider_rom_identity_algorithm",
    }
)


def rom_asset_state_path(repo_root: Path | None = None) -> Path:
    override = os.environ.get("RLAB_ROM_ASSET_STATE")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root
    return root / "logs" / "fleet" / "rom-assets.json"


def legacy_rom_asset_state_path(repo_root: Path | None = None) -> Path:
    override = os.environ.get("RLAB_MODAL_EVAL_ASSET_STATE")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root
    return root / "logs" / "fleet" / "modal-eval-assets.json"


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": ROM_ASSET_STATE_SCHEMA_VERSION, "games": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("games"), dict):
        raise ValueError(f"invalid ROM asset state: {path}")
    return value


def load_rom_asset_state(path: Path | None = None) -> dict[str, Any]:
    target = rom_asset_state_path() if path is None else path
    state = _read_state(target)
    if state.get("games") or target.is_file():
        return state
    return _read_state(legacy_rom_asset_state_path(target.parents[2]))


def write_rom_asset_state(value: Mapping[str, Any], path: Path | None = None) -> Path:
    target = rom_asset_state_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    fd, name = tempfile.mkstemp(prefix=".rom-assets-", dir=target.parent, text=True)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _safe_game(game: object) -> str:
    value = str(game or "").strip()
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("ROM asset game must be a safe non-empty identifier")
    return value


def _safe_basename(filename: object) -> str:
    value = str(filename or "").strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("ROM asset filename must be a safe basename")
    return value


def validate_rom_asset_manifest(
    value: Mapping[str, Any],
    *,
    expected_game: str | None = None,
    require_object_uri: bool = True,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("ROM asset manifest must be an object")
    manifest = dict(value)
    version = int(manifest.get("schema_version") or 1)
    if version != ROM_ASSET_SCHEMA_VERSION and not (allow_legacy and version == 1):
        raise ValueError(f"unsupported ROM asset manifest schema_version: {version}")
    if version == ROM_ASSET_SCHEMA_VERSION:
        unknown = sorted(set(manifest) - _MANIFEST_FIELDS)
        if unknown:
            raise ValueError(f"unknown ROM asset manifest field(s): {', '.join(unknown)}")
    game = _safe_game(manifest.get("game"))
    if expected_game is not None and game != _safe_game(expected_game):
        raise ValueError(f"ROM asset game mismatch: expected {expected_game!r}, got {game!r}")
    filename = _safe_basename(manifest.get("filename"))
    sha256 = str(manifest.get("sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("ROM asset sha256 must be 64 lowercase hexadecimal characters")
    provider_identity = str(manifest.get("provider_rom_identity") or "").strip().lower()
    if not _SHA1_RE.fullmatch(provider_identity):
        raise ValueError("ROM asset provider identity must be 40 lowercase hexadecimal characters")
    algorithm = str(
        manifest.get("provider_rom_identity_algorithm") or ROM_ASSET_IDENTITY_ALGORITHM
    ).strip()
    if algorithm != ROM_ASSET_IDENTITY_ALGORITHM:
        raise ValueError(f"unsupported provider ROM identity algorithm: {algorithm}")
    size_value = manifest.get("size_bytes")
    if size_value is None and allow_legacy and version == 1:
        size_bytes = None
    else:
        if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 1:
            raise ValueError("ROM asset size_bytes must be a positive integer")
        size_bytes = int(size_value)
    object_uri = str(manifest.get("object_uri") or "").strip()
    if require_object_uri:
        parsed = urlparse(object_uri)
        if parsed.scheme not in {"s3", "file"}:
            raise ValueError("ROM asset object_uri must use s3:// or file://")
    elif object_uri and urlparse(object_uri).scheme not in {"s3", "file"}:
        raise ValueError("ROM asset object_uri must use s3:// or file://")
    normalized = {
        "schema_version": version,
        "game": game,
        "filename": filename,
        "sha256": sha256,
        "provider_rom_identity": provider_identity,
        "provider_rom_identity_algorithm": algorithm,
    }
    if size_bytes is not None:
        normalized["size_bytes"] = size_bytes
    if object_uri:
        normalized["object_uri"] = object_uri
    return normalized


def portable_rom_asset_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_rom_asset_manifest(
        value,
        require_object_uri=False,
        allow_legacy=True,
    )
    return {
        "schema_version": ROM_ASSET_SCHEMA_VERSION,
        "sha256": manifest["sha256"],
        "provider_rom_identity": manifest["provider_rom_identity"],
        "provider_rom_identity_algorithm": manifest["provider_rom_identity_algorithm"],
    }


def rom_asset_manifests_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_manifest = validate_rom_asset_manifest(left, allow_legacy=True)
    right_manifest = validate_rom_asset_manifest(right, allow_legacy=True)
    return (
        left_manifest["game"] == right_manifest["game"]
        and portable_rom_asset_identity(left_manifest) == portable_rom_asset_identity(right_manifest)
    )


def manifest_from_train_config(
    train_config: Mapping[str, Any],
    *,
    expected_game: str | None = None,
) -> dict[str, Any] | None:
    current = train_config.get("rom_asset_manifest")
    legacy = train_config.get("checkpoint_eval_asset_manifest")
    if current is not None and not isinstance(current, Mapping):
        raise ValueError("rom_asset_manifest must be an object or null")
    if legacy is not None and not isinstance(legacy, Mapping):
        raise ValueError("checkpoint_eval_asset_manifest must be an object or null")
    if current is not None and legacy is not None and not rom_asset_manifests_equal(current, legacy):
        raise ValueError("ROM asset manifest conflicts with checkpoint_eval_asset_manifest")
    selected = current if isinstance(current, Mapping) else legacy
    if selected is None:
        return None
    return validate_rom_asset_manifest(
        selected,
        expected_game=expected_game,
        allow_legacy=True,
    )


def provider_rom_identity(path: Path, algorithm: str = ROM_ASSET_IDENTITY_ALGORITHM) -> str:
    if algorithm != ROM_ASSET_IDENTITY_ALGORITHM:
        raise ValueError(f"unsupported provider ROM identity algorithm: {algorithm}")
    import stable_retro

    system = stable_retro.get_romfile_system(str(path))
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        if system == "Nes":
            header = handle.read(16)
            if len(header) != 16:
                raise ValueError(f"NES ROM header is truncated: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_rom_file(path: Path, manifest: Mapping[str, Any]) -> Path:
    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    if not path.is_file():
        raise FileNotFoundError(f"ROM cache file is missing: {path}")
    size = path.stat().st_size
    if normalized.get("size_bytes") is not None and size != int(normalized["size_bytes"]):
        raise ValueError(f"ROM size mismatch for {path}")
    if file_sha256(path) != normalized["sha256"]:
        raise ValueError(f"ROM sha256 mismatch for {path}")
    if provider_rom_identity(path, normalized["provider_rom_identity_algorithm"]) != normalized[
        "provider_rom_identity"
    ]:
        raise ValueError(f"ROM provider identity mismatch for {path}")
    return path


def cache_path(cache_root: Path, manifest: Mapping[str, Any]) -> Path:
    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    return (
        cache_root.expanduser()
        / "sha256"
        / normalized["sha256"]
        / normalized["filename"]
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_rom_file(source: Path, manifest: Mapping[str, Any], cache_root: Path) -> Path:
    import fcntl

    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    verify_rom_file(source, normalized)
    destination = cache_path(cache_root, normalized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / ".lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                return verify_rom_file(destination, normalized)
            except (FileNotFoundError, ValueError):
                destination.unlink(missing_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{destination.stem}.",
                suffix=destination.suffix,
                dir=destination.parent,
            )
            temporary = Path(name)
            try:
                with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        target_handle.write(chunk)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                verify_rom_file(temporary, normalized)
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
            return verify_rom_file(destination, normalized)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ensure_rom_cache(
    manifest: Mapping[str, Any],
    *,
    cache_root: Path = DEFAULT_LOCAL_ROM_CACHE,
    store: ObjectStore | None = None,
) -> Path:
    normalized = validate_rom_asset_manifest(manifest, allow_legacy=True)
    destination = cache_path(cache_root, normalized)
    try:
        return verify_rom_file(destination, normalized)
    except (FileNotFoundError, ValueError):
        pass
    object_store = store or ObjectStore(object_store_base_uri())
    payload = object_store.get_bytes(str(normalized["object_uri"]))
    with tempfile.NamedTemporaryFile(
        prefix="rlab-rom-download-",
        suffix=Path(normalized["filename"]).suffix,
        delete=False,
    ) as handle:
        handle.write(payload)
        source = Path(handle.name)
    try:
        return install_rom_file(source, normalized, cache_root)
    finally:
        source.unlink(missing_ok=True)


def stage_rom_from_url(
    manifest: Mapping[str, Any],
    *,
    url: str,
    cache_root: Path,
) -> Path:
    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    destination = cache_path(cache_root, normalized)
    try:
        return verify_rom_file(destination, normalized)
    except (FileNotFoundError, ValueError):
        pass
    with tempfile.TemporaryDirectory(prefix="rlab-rom-stage-") as temporary:
        source = write_downloaded_file(
            url,
            Path(temporary) / normalized["filename"],
        )
        return install_rom_file(source, normalized, cache_root)


def _expected_provider_identities(game: str) -> set[str]:
    import stable_retro

    expected_path = stable_retro.data.get_file_path(
        game,
        "rom.sha",
        inttype=stable_retro.data.Integrations.ALL,
    )
    return {
        line.strip().lower()
        for line in Path(expected_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def discover_rom_path(
    game: str,
    *,
    rom_path: Path | None = None,
    source_dir: Path = Path("~/roms"),
) -> Path:
    game = _safe_game(game)
    expected = _expected_provider_identities(game)
    if rom_path is not None:
        candidate = rom_path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"ROM path does not exist: {candidate}")
        if provider_rom_identity(candidate) not in expected:
            raise ValueError(f"ROM does not match the provider identity for {game}")
        return candidate

    root = source_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ROM source directory does not exist: {root}")
    matches: dict[str, list[Path]] = {}
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            identity = provider_rom_identity(candidate)
        except Exception:
            continue
        if identity in expected:
            matches.setdefault(file_sha256(candidate), []).append(candidate)
    if not matches:
        raise FileNotFoundError(f"no provider-compatible ROM found for {game} under {root}")
    if len(matches) > 1:
        preview = ", ".join(str(paths[0]) for paths in matches.values())
        raise ValueError(
            f"multiple distinct ROM files match {game}; pass --rom-path explicitly: {preview}"
        )
    return next(iter(matches.values()))[0]


def _object_key(manifest: Mapping[str, Any]) -> str:
    return (
        f"{ROM_ASSET_PREFIX}/objects/sha256/{manifest['sha256']}/{manifest['filename']}"
    )


def _revision_key(manifest: Mapping[str, Any]) -> str:
    return f"{ROM_ASSET_PREFIX}/manifests/{manifest['game']}/{manifest['sha256']}.json"


def _pointer_key(game: str) -> str:
    return f"{ROM_ASSET_PREFIX}/games/{_safe_game(game)}.json"


def _cache_manifest_locally(manifest: Mapping[str, Any]) -> None:
    state = load_rom_asset_state()
    state["schema_version"] = ROM_ASSET_STATE_SCHEMA_VERSION
    state.setdefault("games", {})[str(manifest["game"])] = dict(manifest)
    write_rom_asset_state(state)


def rom_asset_manifest_for_game(
    game: str,
    *,
    store: ObjectStore | None = None,
) -> dict[str, Any]:
    game = _safe_game(game)
    object_store = store or ObjectStore(object_store_base_uri())
    try:
        pointer = object_store.get_json(_pointer_key(game))
    except ObjectNotFound as exc:
        raise ValueError(
            f"ROM asset for {game!r} is not provisioned; run: rlab rom sync --game {game}"
        ) from exc
    manifest = validate_rom_asset_manifest(pointer, expected_game=game)
    head = object_store.head(str(manifest["object_uri"]))
    if int(head["size"]) != int(manifest["size_bytes"]):
        raise ValueError(f"ROM asset object size mismatch for {game}")
    remote_sha = str(head.get("metadata", {}).get("sha256") or "")
    if object_store.scheme == "s3" and remote_sha != manifest["sha256"]:
        raise ValueError(f"ROM asset object hash metadata mismatch for {game}")
    _cache_manifest_locally(manifest)
    return manifest


def sync_rom_asset(
    game: str,
    *,
    rom_path: Path | None = None,
    source_dir: Path = Path("~/roms"),
    replace: bool = False,
    store: ObjectStore | None = None,
    local_cache_root: Path = DEFAULT_LOCAL_ROM_CACHE,
) -> dict[str, Any]:
    game = _safe_game(game)
    source = discover_rom_path(game, rom_path=rom_path, source_dir=source_dir)
    sha256 = file_sha256(source)
    object_store = store or ObjectStore(object_store_base_uri())
    manifest: dict[str, Any] = {
        "schema_version": ROM_ASSET_SCHEMA_VERSION,
        "game": game,
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256,
        "object_uri": object_store.uri(
            f"{ROM_ASSET_PREFIX}/objects/sha256/{sha256}/{source.name}"
        ),
        "provider_rom_identity": provider_rom_identity(source),
        "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
    }
    manifest = validate_rom_asset_manifest(manifest, expected_game=game)
    object_store.put_file(
        _object_key(manifest),
        source,
        sha256=sha256,
        content_type="application/octet-stream",
    )
    object_store.put_json(_revision_key(manifest), manifest, create_only=True)
    pointer_key = _pointer_key(game)
    current = object_store.get_json_optional(pointer_key)
    if current is None:
        object_store.put_json_conditional(pointer_key, manifest, if_none_match=True)
    else:
        current_manifest = validate_rom_asset_manifest(current, expected_game=game)
        if rom_asset_manifests_equal(current_manifest, manifest):
            manifest = current_manifest
        elif not replace:
            raise ValueError(
                f"ROM asset for {game} is already pinned to {current_manifest['sha256']}; "
                "pass --replace to change it"
            )
        else:
            etag = str(object_store.head(pointer_key).get("etag") or "")
            if not etag:
                raise RuntimeError(f"ROM asset pointer for {game} has no ETag")
            object_store.put_json_conditional(pointer_key, manifest, if_match=etag)
    install_rom_file(source, manifest, local_cache_root)
    _cache_manifest_locally(manifest)
    return manifest
