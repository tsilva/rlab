from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rlab.file_utils import file_sha256


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
    if repo_root is not None:
        return repo_root / ".rlab" / "rom-assets.json"
    return Path("~/.config/rlab/rom-assets.json").expanduser()


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": ROM_ASSET_STATE_SCHEMA_VERSION, "games": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("games"), dict):
        raise ValueError(f"invalid ROM asset state: {path}")
    return value


def load_rom_asset_state(path: Path | None = None) -> dict[str, Any]:
    target = rom_asset_state_path() if path is None else path
    return _read_state(target)


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
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("ROM asset manifest must be an object")
    manifest = dict(value)
    version = int(manifest.get("schema_version") or 0)
    if version != ROM_ASSET_SCHEMA_VERSION:
        raise ValueError(f"unsupported ROM asset manifest schema_version: {version}")
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
    normalized["size_bytes"] = size_bytes
    if object_uri:
        normalized["object_uri"] = object_uri
    return normalized


def portable_rom_asset_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_rom_asset_manifest(
        value,
        require_object_uri=False,
    )
    return {
        "schema_version": ROM_ASSET_SCHEMA_VERSION,
        "sha256": manifest["sha256"],
        "provider_rom_identity": manifest["provider_rom_identity"],
        "provider_rom_identity_algorithm": manifest["provider_rom_identity_algorithm"],
    }


def rom_asset_manifests_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_manifest = validate_rom_asset_manifest(left)
    right_manifest = validate_rom_asset_manifest(right)
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
    if current is not None and not isinstance(current, Mapping):
        raise ValueError("rom_asset_manifest must be an object or null")
    if current is None:
        return None
    return validate_rom_asset_manifest(
        current,
        expected_game=expected_game,
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
) -> Path:
    normalized = validate_rom_asset_manifest(manifest)
    destination = cache_path(cache_root, normalized)
    try:
        return verify_rom_file(destination, normalized)
    except (FileNotFoundError, ValueError):
        pass
    parsed = urlparse(str(normalized["object_uri"]))
    if parsed.scheme != "file":
        raise FileNotFoundError(
            "ROM is not present in the local cache; import it locally or pass --rom-path "
            "when launching the dstack run"
        )
    return install_rom_file(Path(unquote(parsed.path)), normalized, cache_root)


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


def _cache_manifest_locally(manifest: Mapping[str, Any]) -> None:
    state = load_rom_asset_state()
    state["schema_version"] = ROM_ASSET_STATE_SCHEMA_VERSION
    state.setdefault("games", {})[str(manifest["game"])] = dict(manifest)
    write_rom_asset_state(state)


def rom_asset_manifest_for_game(
    game: str,
) -> dict[str, Any]:
    game = _safe_game(game)
    value = load_rom_asset_state().get("games", {}).get(game)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ROM asset for {game!r} is not registered locally; "
            f"run: rlab rom sync --game {game}"
        )
    manifest = validate_rom_asset_manifest(value, expected_game=game)
    verify_rom_file(cache_path(DEFAULT_LOCAL_ROM_CACHE, manifest), manifest)
    return manifest


def sync_rom_asset(
    game: str,
    *,
    rom_path: Path | None = None,
    source_dir: Path = Path("~/roms"),
    replace: bool = False,
    local_cache_root: Path = DEFAULT_LOCAL_ROM_CACHE,
) -> dict[str, Any]:
    game = _safe_game(game)
    source = discover_rom_path(game, rom_path=rom_path, source_dir=source_dir)
    sha256 = file_sha256(source)
    destination = install_rom_file(
        source,
        {
            "schema_version": ROM_ASSET_SCHEMA_VERSION,
            "game": game,
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": sha256,
            "object_uri": source.resolve().as_uri(),
            "provider_rom_identity": provider_rom_identity(source),
            "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
        },
        local_cache_root,
    )
    manifest: dict[str, Any] = {
        "schema_version": ROM_ASSET_SCHEMA_VERSION,
        "game": game,
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256,
        "object_uri": destination.resolve().as_uri(),
        "provider_rom_identity": provider_rom_identity(source),
        "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
    }
    manifest = validate_rom_asset_manifest(manifest, expected_game=game)
    current = load_rom_asset_state().get("games", {}).get(game)
    if isinstance(current, Mapping) and not rom_asset_manifests_equal(current, manifest):
        if not replace:
            raise ValueError(
                f"ROM asset for {game} is already pinned to {current['sha256']}; "
                "pass --replace to change it"
            )
    _cache_manifest_locally(manifest)
    return manifest
