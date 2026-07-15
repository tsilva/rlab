from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlab.wandb_publisher import project_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project accepted Modal eval evidence to W&B")
    parser.add_argument("--payload", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection payload must be a mapping")
    project_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
