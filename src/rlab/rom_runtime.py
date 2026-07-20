from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rlab.modal_eval_storage import ObjectStore, object_store_base_uri
from rlab.rom_assets import (
    CONTAINER_ROM_CACHE,
    DEFAULT_LOCAL_ROM_CACHE,
    ROM_ASSET_PREFIX,
    cache_path,
    ensure_rom_cache,
    load_rom_asset_state,
    portable_rom_asset_identity,
    rom_asset_manifest_for_game,
    validate_rom_asset_manifest,
    verify_rom_file,
)


@dataclass(frozen=True)
class RomRuntimeBinding:
    manifest: dict[str, Any]
    path: Path

    @property
    def rom_path(self) -> str:
        return str(self.path)


def runtime_cache_root(*, container_default: bool = False) -> Path:
    override = os.environ.get("RLAB_ROM_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return CONTAINER_ROM_CACHE if container_default else DEFAULT_LOCAL_ROM_CACHE


def bind_rom_path(manifest: Mapping[str, Any], path: Path) -> RomRuntimeBinding:
    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    verify_rom_file(path, normalized)
    return RomRuntimeBinding(manifest=normalized, path=path.resolve())


def bind_cached_rom(
    manifest: Mapping[str, Any],
    *,
    cache_root: Path,
) -> RomRuntimeBinding:
    normalized = validate_rom_asset_manifest(
        manifest,
        require_object_uri=False,
        allow_legacy=True,
    )
    return bind_rom_path(normalized, cache_path(cache_root, normalized))


def _manifest_with_locator(manifest: Mapping[str, Any], *, game: str) -> dict[str, Any]:
    normalized = validate_rom_asset_manifest(
        manifest,
        expected_game=game,
        require_object_uri=False,
        allow_legacy=True,
    )
    if normalized.get("object_uri"):
        return normalized
    expected_identity = portable_rom_asset_identity(normalized)
    local = load_rom_asset_state().get("games", {}).get(game)
    if isinstance(local, Mapping):
        candidate = validate_rom_asset_manifest(local, expected_game=game, allow_legacy=True)
        if portable_rom_asset_identity(candidate) == expected_identity:
            return candidate
    try:
        active = rom_asset_manifest_for_game(game)
    except Exception:
        active = None
    if isinstance(active, Mapping) and portable_rom_asset_identity(active) == expected_identity:
        return dict(active)
    store = ObjectStore(object_store_base_uri())
    object_uri = store.uri(
        f"{ROM_ASSET_PREFIX}/objects/sha256/{normalized['sha256']}/{normalized['filename']}"
    )
    store.head(object_uri)
    return {**normalized, "object_uri": object_uri}


def ensure_local_rom_binding(
    manifest: Mapping[str, Any],
    *,
    game: str,
    cache_root: Path | None = None,
) -> RomRuntimeBinding:
    located = _manifest_with_locator(manifest, game=game)
    root = cache_root or runtime_cache_root()
    path = ensure_rom_cache(located, cache_root=root)
    return bind_rom_path(located, path)
