#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

if __package__:
    from .dockerfile_inputs import marked_dockerfile_bytes
else:
    from dockerfile_inputs import marked_dockerfile_bytes


GPU_KEY_SCHEMA = b"rlab-train-gpu-key-v1\0"


def gpu_key(*, dockerfile: Path, lockfile: Path) -> str:
    digest = hashlib.sha256(GPU_KEY_SCHEMA)
    for name, content in (
        ("Dockerfile.gpu", marked_dockerfile_bytes(path=dockerfile, section="gpu")),
        (lockfile.name, lockfile.read_bytes()),
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Hash train-image GPU foundation inputs")
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=repo_root / "containers/train/Dockerfile",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=repo_root / "containers/train/gpu-linux-amd64.lock",
    )
    args = parser.parse_args()
    print(gpu_key(dockerfile=args.dockerfile, lockfile=args.lockfile))


if __name__ == "__main__":
    main()
