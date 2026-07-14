#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


BEGIN_MARKER = b"# dependency-image-inputs-begin"
END_MARKER = b"# dependency-image-inputs-end"


def _dependency_dockerfile_bytes(path: Path) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or not lines[0].startswith(b"# syntax="):
        raise ValueError(f"{path} must begin with a Dockerfile syntax directive")

    begin = [index for index, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    end = [index for index, line in enumerate(lines) if line.strip() == END_MARKER]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        raise ValueError(f"{path} must contain one ordered dependency input marker pair")

    return lines[0] + b"".join(lines[begin[0] : end[0] + 1])


def dependency_key(*, dockerfile: Path, pyproject: Path, lockfile: Path) -> str:
    digest = hashlib.sha256()
    inputs = (
        ("Dockerfile.dependencies", _dependency_dockerfile_bytes(dockerfile)),
        (pyproject.name, pyproject.read_bytes()),
        (lockfile.name, lockfile.read_bytes()),
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
    parser.add_argument("--pyproject", type=Path, default=repo_root / "pyproject.toml")
    parser.add_argument("--lockfile", type=Path, default=repo_root / "uv.lock")
    args = parser.parse_args()
    print(
        dependency_key(
            dockerfile=args.dockerfile,
            pyproject=args.pyproject,
            lockfile=args.lockfile,
        )
    )


if __name__ == "__main__":
    main()
