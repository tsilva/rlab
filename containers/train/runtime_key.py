#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


RUNTIME_INPUT_PATHS = (
    ".dockerignore",
    "containers/train/Dockerfile",
    "containers/train/dependency_key.py",
    "containers/train/entrypoint.sh",
    "containers/train/rlab",
    "containers/train/runtime_key.py",
    "containers/train/smoke.py",
    "experiments/machines.yaml",
    "experiments/modal_eval.yaml",
    "pyproject.toml",
    "scripts",
    "src",
    "uv.lock",
)

RUNTIME_KEY_SCHEMA = b"rlab-runtime-input-v1\0"


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


def runtime_key(*, repo_root: Path) -> str:
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
    digest = hashlib.sha256(RUNTIME_KEY_SCHEMA)
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
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash immutable train-runtime inputs")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(runtime_key(repo_root=args.repo_root.resolve()))


if __name__ == "__main__":
    main()
