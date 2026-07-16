from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "experiments" / "eval_capacity.yaml"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def workload_key(train_config: Mapping[str, Any]) -> str:
    game = str(train_config.get("game") or "").strip()
    state = str(train_config.get("state") or "").strip()
    if not state:
        environment = train_config.get("checkpoint_eval_environment")
        if isinstance(environment, Mapping):
            state = str(environment.get("state") or "").strip()
    if not game or not state:
        raise ValueError("eval capacity workload requires a materialized game and state")
    return f"{game}:{state}:acceptance-v1"


@dataclass(frozen=True)
class EvalCapacityPolicy:
    document: dict[str, Any]
    sha256: str

    @property
    def effective_capacity(self) -> int:
        return int(self.document["effective_capacity"])

    @property
    def admission_utilization(self) -> float:
        return float(self.document["admission_utilization"])

    @property
    def hard_safety_cap(self) -> int:
        return int(self.document["hard_safety_cap"])

    def reservation(self, train_config: Mapping[str, Any]) -> dict[str, Any] | None:
        if str(train_config.get("checkpoint_eval_backend") or "") != "modal":
            return None
        if not bool(train_config.get("stop_on_acceptance")):
            return None
        key = workload_key(train_config)
        workloads = self.document.get("workloads")
        workload = workloads.get(key) if isinstance(workloads, Mapping) else None
        if not isinstance(workload, Mapping):
            raise ValueError(f"eval capacity policy has no benchmark for {key}")
        if str(workload.get("status") or "") != "accepted":
            raise ValueError(f"eval capacity benchmark for {key} is not accepted")
        training_fps = float(workload["training_fps_upper_bound"])
        full_p95 = float(workload["full_eval_p95_seconds"])
        interval = int(train_config.get("checkpoint_freq") or 0)
        if training_fps <= 0 or full_p95 <= 0 or interval <= 0:
            raise ValueError("eval capacity reservation inputs must be positive")
        minimum = training_fps * full_p95 / self.effective_capacity * 1.25
        selected = int(workload["selected_checkpoint_interval_steps"])
        if selected < math.ceil(minimum):
            raise ValueError("capacity policy checkpoint interval is below the derived minimum")
        if interval < selected:
            raise ValueError(
                f"checkpoint_freq={interval} is below benchmark-selected cadence {selected}"
            )
        eval_load = training_fps * full_p95 / interval
        return {
            "schema_version": 1,
            "backend": "modal",
            "policy_sha256": self.sha256,
            "workload_key": key,
            "training_fps_upper_bound": training_fps,
            "full_eval_p95_seconds": full_p95,
            "checkpoint_interval_steps": interval,
            "minimum_interval_steps": math.ceil(minimum),
            "selected_checkpoint_interval_steps": selected,
            "eval_load": eval_load,
            "admission_utilization": self.admission_utilization,
            "effective_capacity": self.effective_capacity,
            "hard_safety_cap": self.hard_safety_cap,
        }


def load_eval_capacity_policy(path: Path = DEFAULT_POLICY_PATH) -> EvalCapacityPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("eval capacity policy must be a mapping")
    document = dict(raw)
    if int(document.get("schema_version") or 0) != 1:
        raise ValueError("unsupported eval capacity policy schema version")
    if str(document.get("backend") or "") != "modal":
        raise ValueError("eval capacity policy backend must be modal")
    effective = int(document.get("effective_capacity") or 0)
    hard = int(document.get("hard_safety_cap") or 0)
    utilization = float(document.get("admission_utilization") or 0.0)
    if effective != 3:
        raise ValueError("Modal effective capacity must be three")
    if hard != 20:
        raise ValueError("Modal hard safety cap must remain twenty")
    if not 0.0 < utilization <= 0.8:
        raise ValueError("eval admission utilization must be in (0, 0.8]")
    return EvalCapacityPolicy(document=document, sha256=_canonical_sha256(document))
