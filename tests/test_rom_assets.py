from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from rlab.env_identity import environment_identity_from_train_config
from rlab.rom_assets import (
    ROM_ASSET_IDENTITY_ALGORITHM,
    cache_path,
    discover_rom_path,
    ensure_rom_cache,
    manifest_from_train_config,
    provider_rom_identity,
    rom_asset_manifest_for_game,
    sync_rom_asset,
    validate_rom_asset_manifest,
    verify_rom_file,
)
from rlab.rom_cli import build_parser, cmd_status


GAME = "SuperMarioBros-Nes-v0"


def _rom(path: Path, body: bytes) -> Path:
    path.write_bytes(b"NES\x1a" + bytes((1, 1)) + bytes(10) + body)
    return path


def _manifest(path: Path, *, object_uri: str | None = None) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "game": GAME,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "object_uri": object_uri or path.resolve().as_uri(),
        "provider_rom_identity": provider_rom_identity(path),
        "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("filename", "../rom.nes", "safe basename"),
        ("sha256", "xyz", "64 lowercase"),
        ("provider_rom_identity", "xyz", "40 lowercase"),
        ("provider_rom_identity_algorithm", "sha1", "unsupported"),
        ("object_uri", "https://example.invalid/rom.nes", "s3:// or file://"),
        ("unexpected", True, "unknown ROM asset manifest field"),
    ),
)
def test_manifest_v2_validation_is_strict(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        validate_rom_asset_manifest(manifest)


def test_manifest_rejects_wrong_game(tmp_path: Path) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))

    with pytest.raises(ValueError, match="game mismatch"):
        validate_rom_asset_manifest(manifest, expected_game="Other-Nes-v0")


def test_manifest_from_train_config_rejects_retired_schema(tmp_path: Path) -> None:
    current = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    assert manifest_from_train_config(
        {"rom_asset_manifest": current},
        expected_game=GAME,
    )["sha256"] == current["sha256"]
    with pytest.raises(ValueError, match="unsupported.*schema_version"):
        manifest_from_train_config(
            {"rom_asset_manifest": {**current, "schema_version": 1}},
            expected_game=GAME,
        )


def test_discovery_ignores_duplicate_bytes_but_rejects_distinct_matches(tmp_path: Path) -> None:
    first = _rom(tmp_path / "one.nes", b"one")
    duplicate = tmp_path / "duplicate.nes"
    duplicate.write_bytes(first.read_bytes())
    second = _rom(tmp_path / "two.nes", b"two")
    identities = {
        first.resolve(): "a" * 40,
        duplicate.resolve(): "a" * 40,
        second.resolve(): "a" * 40,
    }

    with (
        patch("rlab.rom_assets._expected_provider_identities", return_value={"a" * 40}),
        patch(
            "rlab.rom_assets.provider_rom_identity",
            side_effect=lambda path: identities[path.resolve()],
        ),
        pytest.raises(ValueError, match="multiple distinct ROM files"),
    ):
        discover_rom_path(GAME, source_dir=tmp_path)

    second.unlink()
    with (
        patch("rlab.rom_assets._expected_provider_identities", return_value={"a" * 40}),
        patch(
            "rlab.rom_assets.provider_rom_identity",
            side_effect=lambda path: identities[path.resolve()],
        ),
    ):
        assert discover_rom_path(GAME, source_dir=tmp_path) == duplicate.resolve()


def test_cache_repairs_corruption_from_local_source_and_then_reuses(
    tmp_path: Path,
) -> None:
    source = _rom(tmp_path / "rom.nes", b"one")
    manifest = _manifest(source)
    root = tmp_path / "cache"

    installed = ensure_rom_cache(manifest, cache_root=root)
    verify_rom_file(installed, manifest)
    installed.write_bytes(b"corrupt")
    repaired = ensure_rom_cache(manifest, cache_root=root)
    assert repaired.read_bytes() == source.read_bytes()
    source.unlink()
    assert ensure_rom_cache(manifest, cache_root=root) == repaired


def test_sync_pins_local_identity_and_requires_explicit_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _rom(tmp_path / "first.nes", b"one")
    second = _rom(tmp_path / "second.nes", b"two")
    monkeypatch.setenv("RLAB_ROM_ASSET_STATE", str(tmp_path / "state.json"))
    cache = tmp_path / "cache"

    with (
        patch("rlab.rom_assets.discover_rom_path", return_value=first),
        patch("rlab.rom_assets.DEFAULT_LOCAL_ROM_CACHE", cache),
    ):
        pinned = sync_rom_asset(
            GAME,
            local_cache_root=cache,
        )
        assert rom_asset_manifest_for_game(GAME) == pinned

    with (
        patch("rlab.rom_assets.discover_rom_path", return_value=second),
        pytest.raises(ValueError, match="--replace"),
    ):
        sync_rom_asset(GAME, local_cache_root=cache)

    with (
        patch("rlab.rom_assets.discover_rom_path", return_value=second),
        patch("rlab.rom_assets.DEFAULT_LOCAL_ROM_CACHE", cache),
    ):
        replaced = sync_rom_asset(
            GAME,
            replace=True,
            local_cache_root=cache,
        )
        assert rom_asset_manifest_for_game(GAME) == replaced
    assert replaced["sha256"] != pinned["sha256"]


def test_rom_identity_changes_environment_hash_but_runtime_path_does_not(tmp_path: Path) -> None:
    first = _manifest(_rom(tmp_path / "one.nes", b"one"))
    second = _manifest(_rom(tmp_path / "two.nes", b"two"))
    base = {
        "env_provider": "stable-retro-turbo",
        "game": GAME,
        "state": "Level1-1",
        "task": {},
        "rom_asset_manifest": first,
    }
    first_identity = environment_identity_from_train_config(base)
    changed_path = environment_identity_from_train_config(
        {**base, "env_args": {"rom_path": "/different/cache/location.nes"}}
    )
    changed_rom = environment_identity_from_train_config(
        {**base, "rom_asset_manifest": second}
    )

    assert first_identity == changed_path
    assert first_identity != changed_rom
    assert cache_path(tmp_path / "cache", first).parts[-3:] == (
        "sha256",
        first["sha256"],
        first["filename"],
    )


def test_manifest_never_serializes_a_runtime_path(tmp_path: Path) -> None:
    normalized = validate_rom_asset_manifest(_manifest(_rom(tmp_path / "rom.nes", b"one")))
    assert "rom_path" not in json.dumps(normalized)


def test_status_exit_codes_and_default_scope(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    args = Namespace(game=GAME, json=True)
    with (
        patch("rlab.rom_cli.rom_asset_manifest_for_game", return_value=manifest),
        patch("rlab.rom_cli._local_cache_status", return_value={"status": "hit"}) as local,
    ):
        assert cmd_status(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert payload["games"][0]["caches"] == {"local": {"status": "hit"}}
    local.assert_called_once_with(manifest)

def test_status_rejects_removed_remote_target() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["status", "--target", "modal"])
    assert exc.value.code == 2
