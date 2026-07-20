"""Backward-compatible aliases for the former Modal-only ROM asset API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlab.rom_assets import (
    load_rom_asset_state,
    rom_asset_manifest_for_game,
    rom_asset_state_path,
    sync_rom_asset as _sync_rom_asset,
    write_rom_asset_state,
)


asset_state_path = rom_asset_state_path
load_asset_state = load_rom_asset_state
write_asset_state = write_rom_asset_state
asset_manifest_for_game = rom_asset_manifest_for_game


def sync_rom_asset(
    game: str,
    *,
    rom_path: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    if state_path is not None:
        raise ValueError("custom legacy Modal asset state paths are no longer supported")
    return _sync_rom_asset(game, rom_path=rom_path)
