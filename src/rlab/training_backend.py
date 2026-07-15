from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class BackendUnavailableError(RuntimeError):
    """Raised when a known training backend is intentionally unavailable."""


class GracefulStopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def request(self, reason: str) -> None:
        self.requested = True
        self.reason = reason


@dataclass
class BackendContext:
    train_config: Mapping[str, Any]
    args: argparse.Namespace
    environment: Any
    run_dir: Path
    checkpoint_dir: Path
    metric_store: Any
    wandb_enabled: bool
    stop_flag: Any

    def mark_ready(self) -> Path:
        path = self.run_dir / "learner_ready.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "ready_at_unix": time.time(),
                    "training_backend_id": training_backend_id(self.train_config),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


class TrainingBackend(Protocol):
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None: ...

    def run(self, context: BackendContext) -> None: ...


_BACKEND_MODULES = {
    "rlab.jerk": "rlab.training.jerk",
    "sb3.a2c": "rlab.training.sb3_a2c",
    "sb3.ppo": "rlab.training.sb3_ppo",
    "rlab.ppo": "rlab.training.planned",
    "rlab.a2c": "rlab.training.planned",
}

CHECKPOINT_EVAL_ACCEPTANCE = "checkpoint_eval"
FIRST_TRAINING_SUCCESS_ACCEPTANCE = "first_training_success"


def registered_training_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(_BACKEND_MODULES))


def _backend_module(backend_id: str):
    module_name = _BACKEND_MODULES.get(backend_id)
    if module_name is None:
        known = ", ".join(registered_training_backend_ids())
        raise ValueError(f"unknown training backend {backend_id!r}; known: {known}")
    return importlib.import_module(module_name)


def load_training_backend(backend_id: str) -> TrainingBackend:
    module = _backend_module(backend_id)
    return module.backend_for_id(backend_id)


def training_backend_id(config: Mapping[str, Any]) -> str:
    value = config.get("training_backend")
    if not isinstance(value, Mapping):
        raise ValueError("train_config.training_backend must be an object")
    backend_id = str(value.get("id") or "").strip()
    if not backend_id:
        raise ValueError("train_config.training_backend.id must be a non-empty string")
    return backend_id


def training_backend_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("training_backend")
    if not isinstance(value, Mapping):
        raise ValueError("train_config.training_backend must be an object")
    backend_config = value.get("config")
    if not isinstance(backend_config, Mapping):
        raise ValueError("train_config.training_backend.config must be an object")
    return dict(backend_config)


def training_backend_acceptance_mode(config: Mapping[str, Any]) -> str:
    """Return the backend-declared acceptance authority for a train config."""

    backend_id = training_backend_id(config)
    backend_config = training_backend_config(config)
    resolver = getattr(_backend_module(backend_id), "acceptance_mode", None)
    if resolver is None:
        return CHECKPOINT_EVAL_ACCEPTANCE
    mode = str(resolver(backend_id, backend_config)).strip()
    return mode or CHECKPOINT_EVAL_ACCEPTANCE


def accepts_first_training_success(config: Mapping[str, Any]) -> bool:
    return training_backend_acceptance_mode(config) == FIRST_TRAINING_SUCCESS_ACCEPTANCE


def normalize_training_backend(
    value: Any,
    *,
    common_config: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"id", "config"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    backend_id = str(value.get("id") or "").strip()
    if not backend_id:
        raise ValueError(f"{label}.id must be a non-empty string")
    backend_config = value.get("config", {})
    if not isinstance(backend_config, Mapping):
        raise ValueError(f"{label}.config must be an object")
    module = _backend_module(backend_id)
    normalized = module.normalize_config(dict(backend_config), label=f"{label}.config")
    backend = module.backend_for_id(backend_id)
    backend.validate(common_config, normalized)
    return {"id": backend_id, "config": normalized}


def training_backend_contract_payload() -> dict[str, Any]:
    return {
        backend_id: _backend_module(backend_id).contract_payload(backend_id)
        for backend_id in registered_training_backend_ids()
    }


def training_backend_config_hash(config: Mapping[str, Any]) -> str:
    backend = config.get("training_backend")
    if not isinstance(backend, Mapping):
        return ""
    backend_config = backend.get("config")
    if not isinstance(backend_config, Mapping):
        return ""
    encoded = json.dumps(
        backend_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_backend_runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> dict[str, str]:
    module = _backend_module(backend_id)
    return dict(module.runtime_metadata(backend_id, backend_config))
