#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

if __package__:
    from .dockerfile_inputs import marked_dockerfile_bytes
else:
    from dockerfile_inputs import marked_dockerfile_bytes

DEPENDENCY_KEY_SCHEMA = b"rlab-train-dependency-key-v2\0"


def _normalized_digest(value: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("gpu digest must be a sha256 digest")
    return f"sha256:{normalized}"


def dependency_key(*, dockerfile: Path, lockfile: Path, gpu_digest: str) -> str:
    digest = hashlib.sha256(DEPENDENCY_KEY_SCHEMA)
    inputs = (
        (
            "Dockerfile.dependencies",
            marked_dockerfile_bytes(path=dockerfile, section="dependency"),
        ),
        (lockfile.name, lockfile.read_bytes()),
        ("gpu.digest", _normalized_digest(gpu_digest).encode()),
    )
    for name, content in inputs:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Hash train-image dependency inputs")
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=repo_root / "containers/train/Dockerfile",
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=repo_root / "containers/train/train-linux-amd64.lock",
    )
    parser.add_argument("--gpu-digest", required=True)
    args = parser.parse_args()
    print(
        dependency_key(
            dockerfile=args.dockerfile,
            lockfile=args.lockfile,
            gpu_digest=args.gpu_digest,
        )
    )


if __name__ == "__main__":
    main()
