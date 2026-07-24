#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

if __package__:
    from .dockerfile_inputs import runtime_dockerfile_bytes
else:
    from dockerfile_inputs import runtime_dockerfile_bytes


RUNTIME_INPUT_PATHS = (
    ".dockerignore",
    "THIRD_PARTY_NOTICES.md",
    "containers/train/entrypoint.sh",
    "containers/train/rlab",
    "containers/train/smoke.py",
    "experiments/goals",
    "experiments/modal_eval.yaml",
    "experiments/recipes",
    "pyproject.toml",
    "scripts",
    "src",
)

OVERLAY_KEY_SCHEMA = b"rlab-runtime-overlay-key-v2\0"
RUNTIME_KEY_SCHEMA = b"rlab-runtime-input-v2\0"


def _indexed_blob_contents(
    *, repo_root: Path, object_ids: list[bytes]
) -> list[bytes]:
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo_root,
        check=False,
        input=b"\n".join(object_ids) + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "failed to read runtime input contents"
        )
    cursor = 0
    contents: list[bytes] = []
    output = result.stdout
    for expected_oid in object_ids:
        header_end = output.find(b"\n", cursor)
        if header_end < 0:
            raise RuntimeError("truncated git cat-file response")
        header = output[cursor:header_end].split()
        if len(header) != 3 or header[0] != expected_oid or header[1] != b"blob":
            raise RuntimeError(f"unexpected git object response: {header!r}")
        size = int(header[2])
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(output) or output[content_end : content_end + 1] != b"\n":
            raise RuntimeError("truncated git blob content")
        contents.append(output[content_start:content_end])
        cursor = content_end + 1
    if cursor != len(output):
        raise RuntimeError("unexpected trailing git cat-file output")
    return contents


def overlay_key(*, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", *RUNTIME_INPUT_PATHS],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or "failed to enumerate runtime inputs"
        )
    digest = hashlib.sha256(OVERLAY_KEY_SCHEMA)
    entries = [entry for entry in result.stdout.split(b"\0") if entry]
    if not entries:
        raise RuntimeError("runtime input set is empty")
    normalized_entries: list[tuple[bytes, bytes, bytes]] = []
    for entry in sorted(entries, key=lambda value: value.split(b"\t", 1)[-1]):
        metadata, separator, path = entry.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or fields[2] != b"0":
            raise RuntimeError(f"unexpected git index entry: {entry!r}")
        mode, object_id, _stage = fields
        normalized_entries.append((path, mode, object_id))
    contents = _indexed_blob_contents(
        repo_root=repo_root,
        object_ids=[object_id for _path, _mode, object_id in normalized_entries],
    )
    for (path, mode, _object_id), content in zip(normalized_entries, contents, strict=True):
        digest.update(path)
        digest.update(b"\0")
        digest.update(mode)
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        digest.update(b"\0")
    dockerfile = repo_root / "containers/train/Dockerfile"
    dockerfile_content = runtime_dockerfile_bytes(dockerfile)
    digest.update(b"Dockerfile.runtime\0")
    digest.update(len(dockerfile_content).to_bytes(8, "big"))
    digest.update(dockerfile_content)
    digest.update(b"\0")
    return digest.hexdigest()


def _normalized_digest(value: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("dependency digest must be a sha256 digest")
    return f"sha256:{normalized}"


def runtime_key(*, repo_root: Path, dependency_digest: str) -> str:
    digest = hashlib.sha256(RUNTIME_KEY_SCHEMA)
    digest.update(f"sha256:{overlay_key(repo_root=repo_root)}".encode())
    digest.update(b"\0")
    digest.update(_normalized_digest(dependency_digest).encode())
    digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash immutable train-runtime inputs")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--dependency-digest")
    parser.add_argument("--overlay-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.overlay_only:
        print(overlay_key(repo_root=repo_root))
        return
    if not args.dependency_digest:
        parser.error("--dependency-digest is required unless --overlay-only is used")
    print(runtime_key(repo_root=repo_root, dependency_digest=args.dependency_digest))


if __name__ == "__main__":
    main()
