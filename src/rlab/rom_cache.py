from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlab.rom_assets import ensure_rom_cache, manifest_from_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ensure one queue-owned ROM cache entry.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    train_config = dict(payload.get("job", {}).get("train_config") or {})
    manifest = manifest_from_train_config(
        train_config,
        expected_game=str(train_config.get("game") or ""),
    )
    if manifest is None:
        print(json.dumps({"status": "not_required"}, sort_keys=True))
        return 0
    path = ensure_rom_cache(manifest, cache_root=args.cache_root)
    print(
        json.dumps(
            {"status": "ready", "sha256": manifest["sha256"], "path": str(path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
