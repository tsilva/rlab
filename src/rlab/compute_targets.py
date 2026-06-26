from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_INSTANCE_CONFIG = "experiments/instances.json"
DEFAULT_COMPUTE_TARGET = "rtx4090"
LOCAL_TARGET_KINDS = {"local"}
SKYPILOT_TARGET_KINDS = {"skypilot", ""}
MODAL_TARGET_KINDS = {"modal"}


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_instance_config(repo_root: Path, path: Path | None = None) -> dict[str, Any]:
    config_path = path or repo_root / DEFAULT_INSTANCE_CONFIG
    return load_json_file(config_path)


def target_name(payload: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    for key in ("target", "compute_target", "instance"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return DEFAULT_COMPUTE_TARGET


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


def canonical_target_name(
    instance_config: dict[str, Any],
    target: str = DEFAULT_COMPUTE_TARGET,
) -> str:
    return str(instance_defaults(instance_config, target).get("name", target))


def rtx4090_defaults(instance_config: dict[str, Any]) -> dict[str, Any]:
    return instance_defaults(instance_config, DEFAULT_COMPUTE_TARGET)


def instance_label(instance: dict[str, Any]) -> str:
    return str(instance.get("label") or instance.get("name") or DEFAULT_COMPUTE_TARGET)


def target_kind(instance: dict[str, Any]) -> str:
    return str(instance.get("kind", "skypilot")).strip().lower()


def launch_infra(instance: dict[str, Any]) -> str | None:
    value = str(instance.get("infra", "")).strip()
    return value or None


def ensure_available_target(instance: dict[str, Any]) -> None:
    if instance.get("available") is False:
        reason = str(instance.get("disabled_reason") or "target is marked unavailable")
        raise ValueError(f"target {instance_label(instance)!r} is unavailable: {reason}")


def ensure_skypilot_target(instance: dict[str, Any]) -> None:
    ensure_available_target(instance)
    kind = target_kind(instance)
    if kind not in SKYPILOT_TARGET_KINDS:
        raise ValueError(
            f"target {instance_label(instance)!r} has kind {kind!r}; "
            "use the matching compute launcher instead of SkyPilot"
        )


def ensure_modal_target(instance: dict[str, Any]) -> None:
    ensure_available_target(instance)
    kind = target_kind(instance)
    if kind not in MODAL_TARGET_KINDS:
        raise ValueError(
            f"target {instance_label(instance)!r} has kind {kind!r}; "
            "use a Modal target such as modal-t4"
        )


def ensure_local_target(instance: dict[str, Any]) -> None:
    ensure_available_target(instance)
    kind = target_kind(instance)
    if kind not in LOCAL_TARGET_KINDS:
        raise ValueError(
            f"target {instance_label(instance)!r} has kind {kind!r}; "
            "use a local target such as local-macbook"
        )
