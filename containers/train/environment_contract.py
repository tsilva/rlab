#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


GPU_ENVIRONMENT = Path("/root/rlab/.venv")
DEPENDENCY_ENVIRONMENT = Path("/opt/rlab-dependencies")
PYTHON_VERSION = "python3.14"


@dataclass(frozen=True)
class Distribution:
    name: str
    version: str
    requires: tuple[str, ...]
    environment: str


def _site_packages(environment: Path) -> Path:
    return environment / "lib" / PYTHON_VERSION / "site-packages"


def read_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line.split(" --hash=", maxsplit=1)[0])
        versions = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        ]
        if requirement.url or requirement.marker or requirement.extras or len(versions) != 1:
            raise ValueError(f"{path}:{number}: expected one exact registry package pin")
        name = canonicalize_name(requirement.name)
        if name in packages:
            raise ValueError(f"{path}:{number}: duplicate package {name}")
        packages[name] = versions[0]
    return packages


def installed_distributions(environments: dict[str, Path]) -> tuple[Distribution, ...]:
    installed: list[Distribution] = []
    for label, environment in environments.items():
        site_packages = _site_packages(environment)
        for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
            name = distribution.metadata.get("Name")
            if not name:
                raise ValueError(f"distribution without a name under {site_packages}")
            installed.append(
                Distribution(
                    name=canonicalize_name(name),
                    version=distribution.version,
                    requires=tuple(distribution.requires or ()),
                    environment=label,
                )
            )
    return tuple(installed)


def validate_distribution_contract(
    *,
    train: dict[str, str],
    gpu: dict[str, str],
    dependencies: dict[str, str],
    installed: tuple[Distribution, ...],
) -> None:
    if set(gpu) & set(dependencies):
        raise ValueError("GPU and non-GPU locks overlap")
    if {**gpu, **dependencies} != train:
        raise ValueError("GPU and non-GPU locks do not exactly reconstruct the train lock")

    by_name: dict[str, Distribution] = {}
    for distribution in installed:
        previous = by_name.get(distribution.name)
        if previous is not None:
            raise ValueError(
                f"duplicate installed distribution {distribution.name}: "
                f"{previous.environment} and {distribution.environment}"
            )
        by_name[distribution.name] = distribution

    expected_names = set(train)
    installed_names = set(by_name)
    if missing := sorted(expected_names - installed_names):
        raise ValueError(f"missing installed distributions: {', '.join(missing)}")
    if unexpected := sorted(installed_names - expected_names):
        raise ValueError(f"unexpected installed distributions: {', '.join(unexpected)}")

    mismatches = [
        f"{name}: expected {train[name]}, installed {by_name[name].version}"
        for name in sorted(train)
        if by_name[name].version != train[name]
    ]
    if mismatches:
        raise ValueError("installed version mismatches: " + "; ".join(mismatches))

    environment = default_environment()
    environment["extra"] = ""
    unsatisfied: list[str] = []
    for distribution in installed:
        for raw_requirement in distribution.requires:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = by_name.get(canonicalize_name(requirement.name))
            if dependency is None:
                unsatisfied.append(f"{distribution.name} requires missing {requirement}")
            elif requirement.url:
                unsatisfied.append(f"{distribution.name} has unsupported direct requirement {requirement}")
            elif requirement.specifier and not requirement.specifier.contains(
                dependency.version, prereleases=True
            ):
                unsatisfied.append(
                    f"{distribution.name} requires {requirement}, installed {dependency.version}"
                )
    if unsatisfied:
        raise ValueError("unsatisfied installed requirements: " + "; ".join(sorted(unsatisfied)))


def _validate_runtime_layout() -> None:
    dependency_site = _site_packages(DEPENDENCY_ENVIRONMENT)
    gpu_site = _site_packages(GPU_ENVIRONMENT)
    bridge = dependency_site / "rlab-gpu.pth"
    if bridge.read_text(encoding="utf-8").splitlines() != [str(gpu_site)]:
        raise ValueError(f"{bridge} must contain only {gpu_site}")

    expected_path_prefix = [
        str(DEPENDENCY_ENVIRONMENT / "bin"),
        str(GPU_ENVIRONMENT / "bin"),
    ]
    if os.environ.get("PATH", "").split(os.pathsep)[:2] != expected_path_prefix:
        raise ValueError("PATH must select the non-GPU venv before the GPU venv")
    if Path(sys.executable) != DEPENDENCY_ENVIRONMENT / "bin" / "python":
        raise ValueError(f"unexpected interpreter: {sys.executable}")
    if Path(sys.prefix) != DEPENDENCY_ENVIRONMENT:
        raise ValueError(f"unexpected Python prefix: {sys.prefix}")
    if str(gpu_site) not in sys.path:
        raise ValueError("GPU site-packages bridge is not active")

    bad_shebangs: list[str] = []
    expected_shebang = f"#!{DEPENDENCY_ENVIRONMENT}/bin/python"
    for executable in sorted((DEPENDENCY_ENVIRONMENT / "bin").iterdir()):
        if executable.is_symlink() or not executable.is_file():
            continue
        with executable.open("rb") as handle:
            first_line = handle.readline(4096).rstrip(b"\r\n")
        if first_line.startswith(b"#!") and b"python" in first_line:
            if first_line.decode("utf-8") != expected_shebang:
                bad_shebangs.append(str(executable))
    if bad_shebangs:
        raise ValueError("non-GPU console scripts use the wrong venv: " + ", ".join(bad_shebangs))


def validate_environment(
    *, train_lock: Path, gpu_lock: Path, dependency_lock: Path
) -> None:
    train = read_lock(train_lock)
    gpu = read_lock(gpu_lock)
    dependencies = read_lock(dependency_lock)
    installed = installed_distributions(
        {"gpu": GPU_ENVIRONMENT, "dependencies": DEPENDENCY_ENVIRONMENT}
    )
    validate_distribution_contract(
        train=train,
        gpu=gpu,
        dependencies=dependencies,
        installed=installed,
    )
    _validate_runtime_layout()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate the merged train-image Python environment contract."
    )
    parser.add_argument(
        "--train-lock",
        type=Path,
        default=repo_root / "containers/train/train-linux-amd64.lock",
    )
    parser.add_argument(
        "--gpu-lock",
        type=Path,
        default=repo_root / "containers/train/gpu-linux-amd64.lock",
    )
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        default=repo_root / "containers/train/train-dependencies-linux-amd64.lock",
    )
    args = parser.parse_args()
    validate_environment(
        train_lock=args.train_lock,
        gpu_lock=args.gpu_lock,
        dependency_lock=args.dependency_lock,
    )
    print("rlab_train_environment_contract=ok")


if __name__ == "__main__":
    main()
