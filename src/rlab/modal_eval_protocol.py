from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from rlab.early_stop import evaluate_early_stop_config
from rlab.env_registry import resolve_env_provider


PROTOCOL_SCHEMA_VERSION = 1
SEED_PROTOCOL = "vector-lane-v1"


def _sha256(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return text


def validate_announcement(
    announcement: Mapping[str, Any], *, materialized_train_config: Mapping[str, Any]
) -> dict[str, Any]:
    if int(announcement.get("schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("checkpoint announcement schema version mismatch")
    runtime_ref = str(announcement.get("runtime_image_ref") or "")
    if "@sha256:" not in runtime_ref:
        raise ValueError("checkpoint announcement runtime image is not immutable")
    if runtime_ref != str(materialized_train_config.get("runtime_image_ref") or ""):
        raise ValueError("checkpoint announcement runtime identity mismatch")
    _sha256(announcement.get("sha256"), label="checkpoint hash")
    _sha256(announcement.get("metadata_sha256"), label="checkpoint metadata hash")
    eval_contract = announcement.get("eval")
    if not isinstance(eval_contract, Mapping):
        raise ValueError("checkpoint announcement is missing eval contract")
    if str(eval_contract.get("seed_protocol") or "") != SEED_PROTOCOL:
        raise ValueError("checkpoint announcement seed protocol mismatch")
    asset = eval_contract.get("asset")
    environment = eval_contract.get("environment")
    provider = str(environment.get("env_provider") or "") if isinstance(environment, Mapping) else ""
    requires_rom_asset = (
        resolve_env_provider(provider).uses_stable_retro_roms if provider else True
    )
    if requires_rom_asset and not isinstance(asset, Mapping):
        raise ValueError("checkpoint announcement asset contract is missing")
    if isinstance(asset, Mapping):
        _sha256(asset.get("sha256"), label="ROM hash")
        if not str(asset.get("object_uri") or "") or not str(asset.get("provider_rom_identity") or ""):
            raise ValueError("checkpoint announcement asset identity is incomplete")
    elif asset is not None:
        raise ValueError("checkpoint announcement asset contract must be an object or null")
    expected_environment = materialized_train_config.get("checkpoint_eval_environment")
    expected_stages = materialized_train_config.get("checkpoint_eval_stages") or []
    expected_asset = materialized_train_config.get("checkpoint_eval_asset_manifest")
    expected_max_steps = int(materialized_train_config.get("post_train_eval_max_steps") or 0)
    if expected_max_steps <= 0 and isinstance(expected_environment, Mapping):
        task = expected_environment.get("task")
        termination = task.get("termination") if isinstance(task, Mapping) else None
        expected_max_steps = int(
            termination.get("max_episode_steps")
            if isinstance(termination, Mapping)
            and termination.get("max_episode_steps") is not None
            else 0
        )
    expected = {
        "environment": expected_environment,
        "stages": expected_stages,
        "n_envs": int(materialized_train_config.get("checkpoint_eval_n_envs") or 1),
        "max_steps": expected_max_steps,
        "seed": int(materialized_train_config.get("checkpoint_eval_seed") or 10_000),
        "seed_protocol": str(
            materialized_train_config.get("checkpoint_eval_seed_protocol") or SEED_PROTOCOL
        ),
        "asset": expected_asset,
        "promotion_episodes": int(
            materialized_train_config.get("post_train_eval_episodes") or 100
        ),
    }
    if canonical_json(dict(eval_contract)) != canonical_json(expected):
        raise ValueError("checkpoint announcement eval contract does not match the queued contract")
    return dict(announcement)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_execution_contract(
    *,
    checkpoint_sha256: str,
    runtime_image_ref: str,
    eval_environment: Mapping[str, Any],
    episodes: int,
    n_envs: int,
    max_steps: int,
    seed: int,
    seed_protocol: str,
    asset_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if seed_protocol != SEED_PROTOCOL:
        raise ValueError(f"unsupported eval seed protocol: {seed_protocol}")
    contract = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "checkpoint_sha256": str(checkpoint_sha256),
        "runtime_image_ref": str(runtime_image_ref),
        "environment": dict(eval_environment),
        "episodes": int(episodes),
        "n_envs": int(n_envs),
        "max_steps": int(max_steps),
        "deterministic": False,
        "seed": int(seed),
        "seed_protocol": seed_protocol,
        "asset": (
            {
                str(key): value
                for key, value in asset_manifest.items()
                if str(key) != "object_uri"
            }
            if asset_manifest is not None
            else None
        ),
    }
    if contract["episodes"] < 1 or contract["n_envs"] < 1 or contract["max_steps"] < 1:
        raise ValueError("eval episodes, n_envs, and max_steps must be positive")
    if asset_manifest is not None and not str(asset_manifest.get("sha256") or ""):
        raise ValueError("eval asset manifest must include sha256")
    return contract


def execution_key(contract: Mapping[str, Any]) -> str:
    return stable_hash(dict(contract))


def job_key(
    *,
    train_job_id: int,
    ledger_id: int,
    stage_name: str,
    purpose: str,
    candidate_stop: bool,
    execution_key_value: str,
    decision_rules: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash(
        {
            "train_job_id": int(train_job_id),
            "ledger_id": int(ledger_id),
            "stage_name": str(stage_name),
            "purpose": str(purpose),
            "candidate_stop": bool(candidate_stop),
            "execution_key": str(execution_key_value),
            "decision_rules": [dict(rule) for rule in decision_rules],
        }
    )


def stage_job_descriptor(
    announcement: Mapping[str, Any], *, stage_index: int
) -> dict[str, Any]:
    eval_config = announcement.get("eval")
    if not isinstance(eval_config, Mapping):
        raise ValueError("checkpoint announcement is missing eval contract")
    stages = eval_config.get("stages")
    if not isinstance(stages, list) or stage_index < 0 or stage_index >= len(stages):
        raise ValueError(f"checkpoint eval stage index is unavailable: {stage_index}")
    stage = stages[stage_index]
    if not isinstance(stage, Mapping):
        raise ValueError("checkpoint eval stage must be a mapping")
    stage_name = str(stage.get("name") or "").strip()
    if not stage_name:
        raise ValueError("checkpoint eval stage name is required")
    purpose = str(stage.get("purpose") or ("screen" if stage_index == 0 else "confirm"))
    contract = build_execution_contract(
        checkpoint_sha256=str(announcement["sha256"]),
        runtime_image_ref=str(announcement["runtime_image_ref"]),
        eval_environment=dict(eval_config["environment"]),
        episodes=int(stage["episodes"]),
        n_envs=int(stage.get("n_envs") or eval_config["n_envs"]),
        max_steps=int(eval_config["max_steps"]),
        seed=int(eval_config["seed"]),
        seed_protocol=str(eval_config["seed_protocol"]),
        asset_manifest=(
            dict(eval_config["asset"])
            if isinstance(eval_config.get("asset"), Mapping)
            else None
        ),
    )
    execution = execution_key(contract)
    rules = stage.get("pass") or []
    if not isinstance(rules, list):
        raise ValueError("checkpoint eval stage pass rules must be a list")
    key = job_key(
        train_job_id=int(announcement["train_job_id"]),
        ledger_id=int(announcement["ledger_id"]),
        stage_name=stage_name,
        purpose=purpose,
        candidate_stop=bool(stage.get("candidate_stop", False)),
        execution_key_value=execution,
        decision_rules=rules,
    )
    return {
        "stage_name": stage_name,
        "stage_index": stage_index,
        "purpose": purpose,
        "candidate_stop": bool(stage.get("candidate_stop", False)),
        "decision_rules": [dict(rule) for rule in rules],
        "contract": contract,
        "execution_key": execution,
        "job_key": key,
    }


def promotion_job_descriptor(announcement: Mapping[str, Any]) -> dict[str, Any]:
    eval_config = announcement.get("eval")
    if not isinstance(eval_config, Mapping):
        raise ValueError("checkpoint announcement is missing eval contract")
    contract = build_execution_contract(
        checkpoint_sha256=str(announcement["sha256"]),
        runtime_image_ref=str(announcement["runtime_image_ref"]),
        eval_environment=dict(eval_config["environment"]),
        episodes=int(eval_config["promotion_episodes"]),
        n_envs=int(eval_config["n_envs"]),
        max_steps=int(eval_config["max_steps"]),
        seed=int(eval_config["seed"]),
        seed_protocol=str(eval_config["seed_protocol"]),
        asset_manifest=(
            dict(eval_config["asset"])
            if isinstance(eval_config.get("asset"), Mapping)
            else None
        ),
    )
    execution = execution_key(contract)
    key = job_key(
        train_job_id=int(announcement["train_job_id"]),
        ledger_id=int(announcement["ledger_id"]),
        stage_name="promotion",
        purpose="promotion",
        candidate_stop=False,
        execution_key_value=execution,
        decision_rules=[],
    )
    return {
        "stage_name": "promotion",
        "stage_index": len(eval_config.get("stages") or []),
        "purpose": "promotion",
        "candidate_stop": False,
        "decision_rules": [],
        "contract": contract,
        "execution_key": execution,
        "job_key": key,
    }


def validate_attempt_result(
    result: Mapping[str, Any], *, contract: Mapping[str, Any], attempt_id: str
) -> dict[str, Any]:
    if int(result.get("schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("eval result schema version mismatch")
    if str(result.get("attempt_id") or "") != attempt_id:
        raise ValueError("eval result attempt id mismatch")
    if str(result.get("execution_key") or "") != execution_key(contract):
        raise ValueError("eval result execution key mismatch")
    if str(result.get("checkpoint_sha256") or "") != str(contract["checkpoint_sha256"]):
        raise ValueError("eval result checkpoint hash mismatch")
    if int(result.get("contract_schema_version") or 0) != int(contract["schema_version"]):
        raise ValueError("eval result contract schema version mismatch")
    if str(result.get("runtime_image_ref") or "") != str(contract["runtime_image_ref"]):
        raise ValueError("eval result runtime identity mismatch")
    asset = contract.get("asset")
    expected_rom_sha = str(asset.get("sha256") or "") if isinstance(asset, Mapping) else ""
    if str(result.get("rom_sha256") or "") != expected_rom_sha:
        raise ValueError("eval result ROM hash mismatch")
    if str(result.get("seed_protocol") or "") != str(contract["seed_protocol"]):
        raise ValueError("eval result seed protocol mismatch")
    if int(result.get("n_envs") or 0) != int(contract["n_envs"]):
        raise ValueError("eval result n_envs mismatch")
    if int(result.get("episodes") or 0) != int(contract["episodes"]):
        raise ValueError("eval result episode contract mismatch")
    status = str(result.get("status") or "")
    if status != "succeeded":
        raise ValueError(f"eval attempt did not succeed: {status or 'unknown'}")
    episodes = result.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != int(contract["episodes"]):
        raise ValueError("eval result episode count mismatch")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("eval result metrics must be a mapping")
    def validate_finite(value: object, *, label: str) -> None:
        if isinstance(value, bool | str) or value is None:
            return
        if isinstance(value, int | float):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} is not finite")
            return
        if isinstance(value, Mapping):
            for name, nested in value.items():
                validate_finite(nested, label=f"{label}.{name}")
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                validate_finite(nested, label=f"{label}[{index}]")

    validate_finite(metrics, label="eval metrics")
    validate_finite(episodes, label="eval episodes")
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("eval episode result must be a mapping")
        if episode.get("seed_protocol") != contract["seed_protocol"]:
            raise ValueError("eval episode seed protocol mismatch")
    seen_ordinals: set[tuple[int, int]] = set()
    for episode in episodes:
        lane = int(episode.get("seed_lane", -1))
        ordinal = int(episode.get("seed_episode_ordinal", -1))
        if lane < 0 or lane >= int(contract["n_envs"]) or ordinal < 0:
            raise ValueError("eval episode seed lane or ordinal is invalid")
        if (lane, ordinal) in seen_ordinals:
            raise ValueError("eval episode seed lane/ordinal is duplicated")
        seen_ordinals.add((lane, ordinal))
        expected_seed = int(contract["seed"]) + (ordinal if int(contract["n_envs"]) == 1 else 0)
        if int(episode.get("seed", -1)) != expected_seed:
            raise ValueError("eval episode seed trace is inconsistent")
        start = str(episode.get("start_state") or "").strip()
        if not start:
            raise ValueError("eval episode start-state accounting is missing")
    return dict(result)


def apply_decision_rules(
    metrics: Mapping[str, Any], rules: Sequence[Mapping[str, Any]]
) -> tuple[bool, list[dict[str, object]]]:
    def lookup(name: str) -> float | None:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    passed, observed = evaluate_early_stop_config(rules, lookup)
    return bool(passed), observed
