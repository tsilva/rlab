from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_INSTANCE_CONFIG = "experiments/instances.yaml"
DEFAULT_COMPUTE_TARGET = "rtx4090"
FLEET_TARGET_KINDS = {"fleet", "docker", "docker-fleet"}


def load_json_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a config object")
    return data


def load_instance_config(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    config_path = path or repo_root / DEFAULT_INSTANCE_CONFIG
    return load_json_file(config_path)


def instance_defaults(
    instance_config: dict[str, Any],
    target: str = DEFAULT_COMPUTE_TARGET,
) -> dict[str, Any]:
    instances = instance_config.get("instances", {})
    if not isinstance(instances, dict):
        raise ValueError("instances config must contain an instances object")
    instance = instances.get(target)
    canonical_name = target
    if not isinstance(instance, dict):
        for name, candidate in instances.items():
            if not isinstance(candidate, dict):
                continue
            aliases = candidate.get("aliases", [])
            if isinstance(aliases, list) and target in {str(alias) for alias in aliases}:
                instance = candidate
                canonical_name = str(name)
                break
    if not isinstance(instance, dict):
        known = ", ".join(sorted(str(name) for name in instances)) or "<none>"
        raise ValueError(f"instances config must contain target {target!r}; known targets: {known}")
    resolved = dict(instance)
    resolved.setdefault("name", canonical_name)
    resolved["selected_target"] = target
    return resolved


def instance_label(instance: dict[str, Any]) -> str:
    return str(instance.get("label") or instance.get("name") or DEFAULT_COMPUTE_TARGET)


def target_kind(instance: dict[str, Any]) -> str:
    return str(instance.get("kind", "")).strip().lower()
