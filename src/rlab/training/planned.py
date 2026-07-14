from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlab.training_backend import BackendContext, BackendUnavailableError


PLANNED_BACKENDS = {
    "rlab.ppo": "custom PPO",
    "rlab.a2c": "A2C",
}


def _component_id(value: Any, *, label: str, allowed: set[str]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - {"id", "config"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    component_id = str(value.get("id") or "").strip()
    if component_id not in allowed:
        raise ValueError(f"{label}.id must be one of {', '.join(sorted(allowed))}")
    config = value.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"{label}.config must be an object")
    return component_id


def normalize_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    allowed = {"model", "value_normalization", "intrinsic_rewards"}
    unexpected = sorted(set(config) - allowed)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = dict(config)
    if "model" in normalized:
        _component_id(normalized["model"], label=f"{label}.model", allowed={"model1", "model2"})
    if "value_normalization" in normalized:
        _component_id(
            normalized["value_normalization"],
            label=f"{label}.value_normalization",
            allowed={"standard", "popart"},
        )
    intrinsic = normalized.get("intrinsic_rewards", [])
    if not isinstance(intrinsic, Sequence) or isinstance(intrinsic, str | bytes):
        raise ValueError(f"{label}.intrinsic_rewards must be a list")
    for index, component in enumerate(intrinsic):
        _component_id(
            component,
            label=f"{label}.intrinsic_rewards[{index}]",
            allowed={"rnd", "phash"},
        )
    normalized.setdefault("intrinsic_rewards", [])
    return normalized


@dataclass(frozen=True)
class UnavailableBackend:
    backend_id: str
    feature_name: str

    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config, backend_config
        raise BackendUnavailableError(
            f"training backend {self.backend_id!r} is planned but unavailable; "
            f"{self.feature_name} will be implemented in a future milestone"
        )

    def run(self, context: BackendContext) -> None:
        del context
        raise BackendUnavailableError(f"training backend {self.backend_id!r} is unavailable")


def backend_for_id(backend_id: str) -> UnavailableBackend:
    feature_name = PLANNED_BACKENDS.get(backend_id)
    if feature_name is None:
        raise ValueError(f"planned backend module does not define {backend_id!r}")
    return UnavailableBackend(backend_id, feature_name)


def contract_payload(backend_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "planned",
        "feature": PLANNED_BACKENDS[backend_id],
        "fields": {
            "model": ["model1", "model2"],
            "value_normalization": ["standard", "popart"],
            "intrinsic_rewards": ["rnd", "phash"],
        },
    }


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    del backend_config
    return {"training_backend_id": backend_id, "algorithm_id": backend_id.rsplit(".", 1)[-1]}
