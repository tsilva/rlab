from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rlab.early_stop import evaluate_early_stop_config, normalize_early_stop_config
from rlab.eval_metrics import episode_is_complete, episode_start_state
from rlab.metric_names import EVAL_FULL_SUCCESS_RATE_MEAN, EVAL_FULL_SUCCESS_RATE_MIN


ACCEPTANCE_PROTOCOL_VERSION = 1
EPISODE_MANIFEST_VERSION = 1
EVIDENCE_POLICY_VERSION = 1


def _episode_seed(base_seed: int, lane: int, ordinal: int) -> int:
    """Mirror BatchRuntime's reset seed protocol exactly."""

    if ordinal == 0:
        return int(base_seed) + int(lane)
    sequence = np.random.SeedSequence([int(base_seed), int(lane), int(ordinal)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _declared_starts(environment: Mapping[str, Any]) -> tuple[str, ...]:
    env_config = environment.get("env_config")
    source = env_config if isinstance(env_config, Mapping) else environment
    state = str(source.get("state") or "").strip()
    if state:
        return (state,)
    states = source.get("states")
    if isinstance(states, Sequence) and not isinstance(states, str | bytes):
        return tuple(str(value).strip() for value in states if str(value).strip())
    return ()


def build_episode_manifest(
    *,
    episodes: int,
    n_envs: int,
    seed: int,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    episodes = int(episodes)
    n_envs = int(n_envs)
    if episodes < 1:
        raise ValueError("acceptance episodes must be positive")
    if n_envs < 1 or n_envs > episodes:
        raise ValueError("acceptance n_envs must be between 1 and episodes")
    quotient, remainder = divmod(episodes, n_envs)
    quotas = [quotient + (1 if lane < remainder else 0) for lane in range(n_envs)]
    declared_starts = _declared_starts(environment)
    if len(declared_starts) not in {0, 1, n_envs}:
        raise ValueError(
            "acceptance start layout must declare one shared start or one fixed start per lane"
        )
    entries: list[dict[str, Any]] = []
    for lane, quota in enumerate(quotas):
        for ordinal in range(quota):
            entry: dict[str, Any] = {
                "episode_id": f"lane-{lane:02d}-episode-{ordinal:03d}",
                "lane": lane,
                "lane_episode_ordinal": ordinal,
                "seed": _episode_seed(seed, lane, ordinal),
            }
            if len(declared_starts) == 1:
                entry["start_state"] = declared_starts[0]
            elif len(declared_starts) == n_envs:
                entry["start_state"] = declared_starts[lane]
            entries.append(entry)
    return {
        "version": EPISODE_MANIFEST_VERSION,
        "episodes": entries,
        "lane_quotas": quotas,
        "declared_starts": list(declared_starts),
    }


def build_checkpoint_eval_contract(
    *,
    environment: Mapping[str, Any],
    episodes: int,
    n_envs: int,
    max_steps: int,
    seed: int,
    seed_protocol: str,
    acceptance: Sequence[Mapping[str, Any]],
    asset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rules = normalize_early_stop_config(acceptance, label="goal.eval.acceptance")
    if not rules:
        raise ValueError("goal.eval.acceptance must contain at least one rule")
    manifest = build_episode_manifest(
        episodes=episodes,
        n_envs=n_envs,
        seed=seed,
        environment=environment,
    )
    return {
        "protocol_version": ACCEPTANCE_PROTOCOL_VERSION,
        "environment": dict(environment),
        "episodes": int(episodes),
        "n_envs": int(n_envs),
        "max_steps": int(max_steps),
        "deterministic": False,
        "action_sampling": "stochastic",
        "seed": int(seed),
        "seed_protocol": str(seed_protocol),
        "acceptance": rules,
        "manifest": manifest,
        "evidence_policy": {
            "version": EVIDENCE_POLICY_VERSION,
            "fail_fast": "first_failed_episode",
            "complete_metrics_prefix": "eval/full",
            "partial_rejection_metrics": False,
            "aggregate_validation": "fleet-recomputed-v1",
        },
        "asset": dict(asset) if isinstance(asset, Mapping) else None,
    }


def checkpoint_eval_contract_from_train_config(
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    environment = train_config.get("checkpoint_eval_environment")
    acceptance = train_config.get("checkpoint_eval_acceptance")
    if not isinstance(environment, Mapping):
        raise ValueError("checkpoint eval environment is not materialized")
    if not isinstance(acceptance, list):
        raise ValueError("checkpoint eval acceptance rules are not materialized")
    max_steps = int(train_config.get("post_train_eval_max_steps") or 0)
    if max_steps <= 0:
        task = environment.get("task")
        termination = task.get("termination") if isinstance(task, Mapping) else None
        max_steps = int(
            termination.get("max_episode_steps")
            if isinstance(termination, Mapping)
            and termination.get("max_episode_steps") is not None
            else 0
        )
    if max_steps <= 0:
        raise ValueError("checkpoint eval max steps are not materialized")
    return build_checkpoint_eval_contract(
        environment=environment,
        episodes=int(train_config.get("post_train_eval_episodes") or 0),
        n_envs=int(train_config.get("checkpoint_eval_n_envs") or 0),
        max_steps=max_steps,
        seed=int(train_config.get("checkpoint_eval_seed") or 10_000),
        seed_protocol=str(train_config.get("checkpoint_eval_seed_protocol") or ""),
        acceptance=acceptance,
        asset=(
            train_config.get("checkpoint_eval_asset_manifest")
            if isinstance(train_config.get("checkpoint_eval_asset_manifest"), Mapping)
            else None
        ),
    )


def manifest_index(contract: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = contract.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("acceptance contract is missing its episode manifest")
    if int(manifest.get("version") or 0) != EPISODE_MANIFEST_VERSION:
        raise ValueError("acceptance episode manifest version mismatch")
    raw_entries = manifest.get("episodes")
    if not isinstance(raw_entries, list) or len(raw_entries) != int(contract["episodes"]):
        raise ValueError("acceptance episode manifest count mismatch")
    index: dict[tuple[int, int], dict[str, Any]] = {}
    ids: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise ValueError("acceptance episode manifest entry must be an object")
        entry = dict(raw)
        key = (int(entry.get("lane", -1)), int(entry.get("lane_episode_ordinal", -1)))
        episode_id = str(entry.get("episode_id") or "")
        if key in index or episode_id in ids or key[0] < 0 or key[1] < 0 or not episode_id:
            raise ValueError("acceptance episode manifest contains an invalid duplicate identity")
        index[key] = entry
        ids.add(episode_id)
    quotas = manifest.get("lane_quotas")
    if not isinstance(quotas, list) or len(quotas) != int(contract["n_envs"]):
        raise ValueError("acceptance episode manifest lane quotas are invalid")
    expected_keys = {
        (lane, ordinal)
        for lane, quota in enumerate(quotas)
        for ordinal in range(int(quota))
    }
    if set(index) != expected_keys or sum(int(value) for value in quotas) != int(
        contract["episodes"]
    ):
        raise ValueError("acceptance episode manifest does not match its fixed lane quotas")
    return index


def acceptance_aggregates(
    episode_rows: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [dict(row) for row in episode_rows]
    failures = sum(not episode_is_complete(row) for row in rows)
    successes = len(rows) - failures
    by_start: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        start = episode_start_state(row) or ""
        by_start.setdefault(start, []).append(row)
    rates = {
        start: sum(episode_is_complete(row) for row in start_rows) / len(start_rows)
        for start, start_rows in sorted(by_start.items())
        if start_rows
    }
    result: dict[str, Any] = {
        "episodes_planned": int(contract["episodes"]),
        "episodes_completed": len(rows),
        "success_count": successes,
        "failure_count": failures,
        "success_rate_by_start": rates,
    }
    if len(rows) == int(contract["episodes"]) and rates:
        result[EVAL_FULL_SUCCESS_RATE_MIN] = min(rates.values())
        result[EVAL_FULL_SUCCESS_RATE_MEAN] = sum(rates.values()) / len(rates)
    return result


def _equal_json_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def aggregates_match(claimed: Mapping[str, Any], computed: Mapping[str, Any]) -> bool:
    if set(claimed) != set(computed):
        return False
    for key, expected in computed.items():
        actual = claimed[key]
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or not aggregates_match(actual, expected):
                return False
        elif not _equal_json_number(actual, expected):
            return False
    return True


def validate_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    verdict: str,
) -> list[dict[str, Any]]:
    planned = manifest_index(contract)
    rows = [dict(row) for row in episode_rows]
    seen: set[tuple[int, int]] = set()
    failure_seen = False
    for row in rows:
        key = (int(row.get("seed_lane", -1)), int(row.get("seed_episode_ordinal", -1)))
        entry = planned.get(key)
        if entry is None or key in seen:
            raise ValueError("acceptance evidence contains an unknown or duplicate episode")
        seen.add(key)
        if str(row.get("episode_id") or "") != str(entry["episode_id"]):
            raise ValueError("acceptance evidence episode identity mismatch")
        if int(row.get("seed", -1)) != int(entry["seed"]):
            raise ValueError("acceptance evidence episode seed mismatch")
        expected_start = str(entry.get("start_state") or "")
        actual_start = episode_start_state(row) or ""
        if expected_start and actual_start != expected_start:
            raise ValueError("acceptance evidence start-state mismatch")
        declared_starts = set(contract["manifest"].get("declared_starts") or [])
        if declared_starts and actual_start not in declared_starts:
            raise ValueError("acceptance evidence contains an undeclared start state")
        if not episode_is_complete(row):
            failure_seen = True
            if verdict == "rejected" and row is not rows[-1]:
                raise ValueError("fail-fast rejection contains episodes after its first failure")
    if verdict == "accepted":
        if set(planned) != seen or failure_seen:
            raise ValueError("accepted evidence must contain every planned successful episode")
    elif verdict == "rejected":
        if not rows or not failure_seen:
            raise ValueError("rejected evidence must contain a valid failed planned episode")
    else:
        raise ValueError("acceptance verdict must be accepted or rejected")
    return rows


def evaluate_acceptance(
    aggregates: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> tuple[bool, list[dict[str, object]]]:
    rules = contract.get("acceptance")
    if not isinstance(rules, list):
        raise ValueError("acceptance contract rules are invalid")
    return evaluate_early_stop_config(rules, lambda metric: aggregates.get(metric))
