from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rlab.docker_host import _run_machine_shell
from rlab.dotenv import load_env_file
from rlab.machines import DEFAULT_MACHINE_REGISTRY, load_machine_registry, resolve_machine
from rlab.rom_assets import (
    DEFAULT_LOCAL_ROM_CACHE,
    cache_path,
    ensure_rom_cache,
    load_rom_asset_state,
    rom_asset_manifest_for_game,
    sync_rom_asset,
    validate_rom_asset_manifest,
    verify_rom_file,
)


MODAL_ROM_CACHE_VOLUME = "rlab-rom-cache-v2"
REMOTE_TARGETS = ("beast-2", "beast-3")
TARGET_CHOICES = ("local", *REMOTE_TARGETS, "modal", "all")


def _targets(values: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(values))
    if "all" in requested:
        return ("local", *REMOTE_TARGETS, "modal")
    return requested


def _remote_cache_status(machine_name: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    machine = resolve_machine(load_machine_registry(DEFAULT_MACHINE_REGISTRY), machine_name)
    normalized = validate_rom_asset_manifest(manifest)
    path = cache_path(Path(machine.paths.rom_cache_dir), normalized)
    command = (
        f"if [ ! -f {shlex.quote(str(path))} ]; then echo missing; exit 0; fi; "
        f"actual=$(sha256sum {shlex.quote(str(path))} | awk '{{print $1}}'); "
        f"if [ \"$actual\" = {shlex.quote(normalized['sha256'])} ]; "
        "then echo hit; else echo corrupt:$actual; fi"
    )
    result = _run_machine_shell(machine, command, capture=True)
    if result.returncode != 0:
        return {"status": "error", "detail": (result.stderr or result.stdout).strip()}
    value = result.stdout.strip()
    if value == "hit":
        return {"status": "hit", "path": str(path)}
    if value == "missing":
        return {"status": "missing", "path": str(path)}
    return {"status": "corrupt", "path": str(path), "detail": value}


def _warm_remote(machine_name: str, manifest: Mapping[str, Any], source: Path) -> dict[str, Any]:
    machine = resolve_machine(load_machine_registry(DEFAULT_MACHINE_REGISTRY), machine_name)
    normalized = validate_rom_asset_manifest(manifest)
    destination = cache_path(Path(machine.paths.rom_cache_dir), normalized)
    parent = destination.parent
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    script = "\n".join(
        [
            "set -eu",
            f"if ! mkdir -p {shlex.quote(str(parent))} 2>/dev/null; then "
            f"sudo -n install -d -o \"$(id -u)\" -g \"$(id -g)\" "
            f"{shlex.quote(str(parent))}; fi",
            f"exec 9>{shlex.quote(str(parent / '.lock'))}",
            "flock 9",
            f"if [ -f {shlex.quote(str(destination))} ] && "
            f"[ \"$(sha256sum {shlex.quote(str(destination))} | awk '{{print $1}}')\" = "
            f"{shlex.quote(normalized['sha256'])} ]; then exit 0; fi",
            f"temporary=$(mktemp {shlex.quote(str(parent / '.rom.XXXXXX'))})",
            'trap \'rm -f "$temporary"\' EXIT',
            'base64 -d > "$temporary"',
            f"test \"$(sha256sum \"$temporary\" | awk '{{print $1}}')\" = "
            f"{shlex.quote(normalized['sha256'])}",
            f"mv \"$temporary\" {shlex.quote(str(destination))}",
            "sync",
        ]
    )
    result = _run_machine_shell(machine, script, input_text=encoded, capture=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to warm ROM cache on {machine_name}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return _remote_cache_status(machine_name, normalized)


def _modal_volume(*, create: bool):
    import modal

    return modal.Volume.from_name(
        MODAL_ROM_CACHE_VOLUME,
        create_if_missing=create,
        version=2,
    )


def _modal_path(manifest: Mapping[str, Any]) -> str:
    return str(cache_path(Path("/"), manifest)).lstrip("/")


def _modal_cache_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_rom_asset_manifest(manifest)
    remote_path = _modal_path(normalized)
    try:
        volume = _modal_volume(create=False)
        payload = b"".join(volume.read_file(remote_path))
    except Exception as exc:
        if type(exc).__name__ in {"NotFoundError", "FileNotFoundError"}:
            return {"status": "missing", "path": remote_path}
        return {"status": "error", "path": remote_path, "detail": type(exc).__name__}
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "status": "hit" if digest == normalized["sha256"] else "corrupt",
        "path": remote_path,
        "sha256": digest,
    }


def _warm_modal(manifest: Mapping[str, Any], source: Path) -> dict[str, Any]:
    volume = _modal_volume(create=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(source, _modal_path(manifest))
    status = _modal_cache_status(manifest)
    if status["status"] != "hit":
        raise RuntimeError(f"Modal ROM cache verification failed: {status}")
    return status


def _local_cache_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = cache_path(DEFAULT_LOCAL_ROM_CACHE, manifest)
    try:
        verify_rom_file(path, manifest)
    except FileNotFoundError:
        return {"status": "missing", "path": str(path)}
    except ValueError as exc:
        return {"status": "corrupt", "path": str(path), "detail": str(exc)}
    return {"status": "hit", "path": str(path)}


def _authority_status(game: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        manifest = rom_asset_manifest_for_game(game)
    except Exception as exc:
        return None, {"status": "unhealthy", "detail": f"{type(exc).__name__}: {exc}"}
    return manifest, {
        "status": "healthy",
        "object_uri": manifest["object_uri"],
        "sha256": manifest["sha256"],
    }


def cmd_sync(args: argparse.Namespace) -> int:
    manifest = sync_rom_asset(
        args.game,
        rom_path=args.rom_path,
        source_dir=args.source_dir,
        replace=bool(args.replace),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_warm(args: argparse.Namespace) -> int:
    manifest = rom_asset_manifest_for_game(args.game)
    source = ensure_rom_cache(manifest)
    results: dict[str, Any] = {}
    for target in _targets(args.target):
        if target == "local":
            results[target] = _local_cache_status(manifest)
        elif target == "modal":
            results[target] = _warm_modal(manifest, source)
        else:
            results[target] = _warm_remote(target, manifest, source)
    print(json.dumps({"game": args.game, "targets": results}, indent=2, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    games = [args.game] if args.game else sorted(load_rom_asset_state().get("games", {}))
    if not games:
        print("no ROM assets are registered locally; pass --game", file=sys.stderr)
        return 1
    requested = _targets(args.target or ("local",))
    rows = []
    healthy = True
    for game in games:
        manifest, authority = _authority_status(game)
        row: dict[str, Any] = {"game": game, "authority": authority, "caches": {}}
        if manifest is None:
            healthy = False
            rows.append(row)
            continue
        row["manifest"] = manifest
        for target in requested:
            if target == "local":
                status = _local_cache_status(manifest)
            elif target == "modal":
                status = _modal_cache_status(manifest)
            else:
                status = _remote_cache_status(target, manifest)
            row["caches"][target] = status
            healthy = healthy and status["status"] == "hit"
        rows.append(row)
    payload = {"schema_version": 1, "healthy": healthy, "games": rows}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in rows:
            cache_text = ", ".join(
                f"{target}={value['status']}" for target, value in row["caches"].items()
            )
            print(f"{row['game']}: authority={row['authority']['status']} {cache_text}".rstrip())
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlab rom")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync")
    sync.add_argument("--game", required=True)
    source = sync.add_mutually_exclusive_group()
    source.add_argument("--rom-path", type=Path)
    source.add_argument("--source-dir", type=Path, default=Path("~/roms"))
    sync.add_argument("--replace", action="store_true")
    sync.set_defaults(func=cmd_sync)

    warm = commands.add_parser("warm")
    warm.add_argument("--game", required=True)
    warm.add_argument("--target", action="append", choices=TARGET_CHOICES, required=True)
    warm.set_defaults(func=cmd_warm)

    status = commands.add_parser("status")
    status.add_argument("--game")
    status.add_argument("--target", action="append", choices=TARGET_CHOICES)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
