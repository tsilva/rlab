#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


TARGET_PYTHON = (3, 14)
TARGET_INTERPRETER = "cp314"
TRAIN_LOCK_NAME = "train-linux-amd64.lock"
GPU_LOCK_NAME = "gpu-linux-amd64.lock"
GPU_PACKAGE_NAMES = {"torch", "triton"}
GPU_PACKAGE_PREFIXES = ("cuda-", "nvidia-")


def _target_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update(
        {
            "implementation_name": "cpython",
            "implementation_version": "3.14.0",
            "os_name": "posix",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "platform_system": "Linux",
            "python_full_version": "3.14.0",
            "python_version": "3.14",
            "sys_platform": "linux",
        }
    )
    return environment


def _supported_tag_rank() -> dict[object, int]:
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
    platforms.extend(
        [
            "manylinux2014_x86_64",
            "manylinux2010_x86_64",
            "manylinux1_x86_64",
            "linux_x86_64",
        ]
    )
    tags = [
        *cpython_tags(python_version=TARGET_PYTHON, platforms=platforms),
        *compatible_tags(
            python_version=TARGET_PYTHON,
            interpreter=TARGET_INTERPRETER,
            platforms=platforms,
        ),
    ]
    return {tag: index for index, tag in enumerate(dict.fromkeys(tags))}


def _exported_requirements(repo_root: Path) -> tuple[Requirement, ...]:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--only-group",
            "train-runtime",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "uv export failed")
    requirements: list[Requirement] = []
    for line in result.stdout.splitlines():
        if not line or line.startswith((" ", "#")):
            continue
        requirements.append(Requirement(line.removesuffix("\\").rstrip()))
    environment = _target_environment()
    selected = [
        requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(environment)
    ]
    return tuple(selected)


def _locked_packages(lockfile: Path) -> dict[tuple[str, str], dict[str, object]]:
    payload = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    packages: dict[tuple[str, str], dict[str, object]] = {}
    for package in payload.get("package", []):
        if not isinstance(package, dict):
            continue
        name = canonicalize_name(str(package.get("name") or ""))
        version = str(package.get("version") or "")
        if name and version:
            packages[(name, version)] = package
    return packages


def _selected_artifact(
    package: dict[str, object], tag_rank: dict[object, int]
) -> dict[str, object]:
    source = package.get("source")
    if not isinstance(source, dict) or set(source) != {"registry"}:
        raise ValueError(f"{package.get('name')} must resolve from one registry source")
    candidates: list[tuple[int, dict[str, object]]] = []
    for wheel in package.get("wheels", []):
        if not isinstance(wheel, dict):
            continue
        filename = unquote(Path(urlparse(str(wheel.get("url") or "")).path).name)
        try:
            _name, _version, _build, tags = parse_wheel_filename(filename)
        except ValueError:
            continue
        ranks = [tag_rank[tag] for tag in tags if tag in tag_rank]
        if ranks:
            candidates.append((min(ranks), wheel))
    if candidates:
        return min(candidates, key=lambda candidate: candidate[0])[1]
    sdist = package.get("sdist")
    if isinstance(sdist, dict):
        return sdist
    raise ValueError(
        f"{package.get('name')}=={package.get('version')} has no compatible "
        "CPython 3.14 manylinux_2_28 x86_64 artifact"
    )


def projected_requirements(repo_root: Path) -> dict[str, str]:
    packages = _locked_packages(repo_root / "uv.lock")
    tag_rank = _supported_tag_rank()
    projected: dict[str, str] = {}
    for requirement in _exported_requirements(repo_root):
        name = canonicalize_name(requirement.name)
        versions = [specifier.version for specifier in requirement.specifier if specifier.operator == "=="]
        if len(versions) != 1:
            raise ValueError(f"exported requirement must contain one exact pin: {requirement}")
        version = versions[0]
        package = packages.get((name, version))
        if package is None:
            raise ValueError(f"missing locked package for {name}=={version}")
        artifact = _selected_artifact(package, tag_rank)
        artifact_hash = str(artifact.get("hash") or "")
        if not artifact_hash.startswith("sha256:"):
            raise ValueError(f"missing sha256 artifact hash for {name}=={version}")
        projected[name] = f"{name}=={version} --hash={artifact_hash}"
    return dict(sorted(projected.items()))


def _render(lines: dict[str, str]) -> str:
    return "".join(f"{line}\n" for line in lines.values())


def _is_gpu_package(name: str) -> bool:
    return name in GPU_PACKAGE_NAMES or name.startswith(GPU_PACKAGE_PREFIXES)


def projection_contents(repo_root: Path) -> dict[str, str]:
    train = projected_requirements(repo_root)
    gpu = {name: line for name, line in train.items() if _is_gpu_package(name)}
    if "torch" not in gpu or "triton" not in gpu or not any(
        name.startswith("nvidia-") for name in gpu
    ):
        raise ValueError("GPU projection is missing Torch, Triton, or NVIDIA packages")
    return {TRAIN_LOCK_NAME: _render(train), GPU_LOCK_NAME: _render(gpu)}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate deterministic CPython 3.14 Linux/amd64 train-image locks."
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = projection_contents(repo_root)
    output_dir = repo_root / "containers" / "train"
    stale: list[str] = []
    for name, content in outputs.items():
        path = output_dir / name
        if args.check:
            try:
                current = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                current = ""
            if current != content:
                stale.append(name)
        else:
            path.write_text(content, encoding="utf-8")
    if stale:
        parser.error(
            "stale train-image lock projection(s): "
            + ", ".join(stale)
            + "; run this command without --check"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
