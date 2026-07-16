from __future__ import annotations

from pathlib import Path


def marked_dockerfile_bytes(*, path: Path, section: str) -> bytes:
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines or not lines[0].startswith(b"# syntax="):
        raise ValueError(f"{path} must begin with a Dockerfile syntax directive")

    begin_marker = f"# {section}-image-inputs-begin".encode()
    end_marker = f"# {section}-image-inputs-end".encode()
    begin = [index for index, line in enumerate(lines) if line.strip() == begin_marker]
    end = [index for index, line in enumerate(lines) if line.strip() == end_marker]
    if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
        raise ValueError(f"{path} must contain one ordered {section} input marker pair")

    return lines[0] + b"".join(lines[begin[0] : end[0] + 1])


def runtime_dockerfile_bytes(path: Path) -> bytes:
    content = marked_dockerfile_bytes(path=path, section="runtime")
    lines = path.read_bytes().splitlines(keepends=True)
    base_args = b"".join(
        line
        for line in lines
        if line.startswith((b"ARG PYTHON_IMAGE=", b"ARG UV_IMAGE="))
    )
    if base_args.count(b"ARG ") != 2:
        raise ValueError(f"{path} must pin PYTHON_IMAGE and UV_IMAGE exactly once")
    return content.splitlines(keepends=True)[0] + base_args + b"".join(
        content.splitlines(keepends=True)[1:]
    )
