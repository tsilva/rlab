from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from rlab.modal_eval_storage import ObjectStore, file_sha256, object_store_base_uri


def asset_state_path(repo_root: Path | None = None) -> Path:
    override = os.environ.get("RLAB_MODAL_EVAL_ASSET_STATE")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[2] if repo_root is None else repo_root
    return root / "logs" / "fleet" / "modal-eval-assets.json"


def load_asset_state(path: Path | None = None) -> dict[str, Any]:
    target = asset_state_path() if path is None else path
    if not target.is_file():
        return {"schema_version": 1, "games": {}}
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("games"), dict):
        raise ValueError(f"invalid Modal eval asset state: {target}")
    return value


def write_asset_state(value: Mapping[str, Any], path: Path | None = None) -> Path:
    target = asset_state_path() if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    fd, name = tempfile.mkstemp(prefix=".modal-eval-assets-", dir=target.parent, text=True)
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


def asset_manifest_for_game(game: str, *, path: Path | None = None) -> dict[str, Any]:
    state = load_asset_state(path)
    manifest = state["games"].get(str(game))
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Modal eval ROM asset for {game!r} is not provisioned; "
            f"run: rlab eval modal assets sync --game {game}"
        )
    required = ("game", "sha256", "object_uri", "filename", "provider_rom_identity")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"Modal eval asset manifest for {game!r} is missing: {', '.join(missing)}")
    return dict(manifest)


def sync_rom_asset(game: str, *, rom_path: Path | None = None, state_path: Path | None = None) -> dict[str, Any]:
    if rom_path is None:
        import stable_retro

        rom_path = Path(stable_retro.data.get_romfile_path(game))
    rom_path = rom_path.expanduser().resolve()
    if not rom_path.is_file():
        raise FileNotFoundError(f"ROM is not imported for {game}: {rom_path}")
    sha256 = file_sha256(rom_path)
    import stable_retro

    with rom_path.open("rb") as handle:
        if stable_retro.get_romfile_system(str(rom_path)) == "Nes":
            handle.read(16)
        provider_rom_identity = hashlib.sha1(handle.read()).hexdigest()
    expected_path = stable_retro.data.get_file_path(
        game,
        "rom.sha",
        inttype=stable_retro.data.Integrations.ALL,
    )
    expected_identities = Path(expected_path).read_text(encoding="utf-8").splitlines()
    if provider_rom_identity not in expected_identities:
        raise ValueError(f"ROM does not match the provider identity for {game}")
    store = ObjectStore(object_store_base_uri())
    object_uri = store.put_file(
        f"modal-assets/{game}/{sha256}/{rom_path.name}",
        rom_path,
        sha256=sha256,
        content_type="application/octet-stream",
    )
    manifest = {
        "schema_version": 1,
        "game": str(game),
        "filename": rom_path.name,
        "sha256": sha256,
        "object_uri": object_uri,
        "provider_rom_identity": provider_rom_identity,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    store.put_json(
        f"modal-assets/{game}/{sha256}/manifest.json", manifest, create_only=True
    )
    state = load_asset_state(state_path)
    state.setdefault("games", {})[str(game)] = manifest
    write_asset_state(state, state_path)
    return manifest
