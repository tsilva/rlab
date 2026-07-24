from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlab.checkpoint_eval_config import checkpoint_eval_max_steps
from rlab.early_stop import evaluate_early_stop_config, normalize_early_stop_config
from rlab.eval_metrics import episode_is_complete, episode_start_state
from rlab.env_registry import resolve_env_provider
from rlab.metric_names import EVAL_FULL_SUCCESS_RATE_MEAN, EVAL_FULL_SUCCESS_RATE_MIN
from rlab.seeds import EVAL_SEED_START
from rlab.rom_assets import manifest_from_train_config, validate_rom_asset_manifest


ACCEPTANCE_PROTOCOL_VERSION = 1
EPISODE_MANIFEST_VERSION = 1
EVIDENCE_POLICY_VERSION = 1
SEED_PROTOCOL = "vector-lane-v1"


def portable_asset_from_train_config(
    train_config: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the portable asset identity shared by eval and playback recipes."""

    selected_environment = environment or train_config
    provider = str(selected_environment.get("env_provider") or "").strip()
    game = str(selected_environment.get("game") or "").strip()
    if not provider or not game:
        raise ValueError("portable asset environment identity is not materialized")
    asset_value = manifest_from_train_config(train_config, expected_game=game)
    requires_asset = resolve_env_provider(provider).requires_external_rom_asset
    if not requires_asset:
        if asset_value is not None and not isinstance(asset_value, Mapping):
            raise ValueError("environment asset manifest must be an object or null")
        return None
    if not isinstance(asset_value, Mapping):
        raise ValueError("environment asset manifest is not materialized")
    asset = validate_rom_asset_manifest(
        asset_value,
        expected_game=game,
    )
    return {
        key: value
        for key, value in asset.items()
        if key not in {"object_uri", "local_path"}
    }


def _required_int(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int,
) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{label}.{key} must be an integer")
    parsed = int(raw)
    if parsed < minimum:
        raise ValueError(f"{label}.{key} must be at least {minimum}")
    return parsed


@dataclass(frozen=True)
class CheckpointEvalContractCompiler:
    """Strict accessor and serializer for one materialized checkpoint-eval contract."""

    environment: dict[str, Any]
    episodes: int
    n_envs: int
    max_steps: int
    seed: int
    seed_protocol: str
    acceptance: list[dict[str, Any]] | None
    asset: dict[str, Any] | None

    @classmethod
    def from_train_config(
        cls,
        train_config: Mapping[str, Any],
        *,
        portable_asset: bool = False,
        require_asset: bool = True,
        materialize_seed_defaults: bool = False,
    ) -> CheckpointEvalContractCompiler:
        label = "checkpoint eval config"
        environment = train_config.get("checkpoint_eval_environment")
        if not isinstance(environment, Mapping):
            raise ValueError("checkpoint eval environment is not materialized")
        environment = deepcopy(dict(environment))
        provider = str(environment.get("env_provider") or "").strip()
        if not provider:
            raise ValueError("checkpoint eval environment provider is not materialized")

        episodes = _required_int(
            train_config,
            "post_train_eval_episodes",
            label=label,
            minimum=1,
        )
        n_envs = _required_int(
            train_config,
            "checkpoint_eval_n_envs",
            label=label,
            minimum=1,
        )
        if n_envs > episodes:
            raise ValueError("checkpoint eval n_envs must not exceed episodes")
        seed_config = dict(train_config)
        if materialize_seed_defaults:
            seed_config.setdefault("checkpoint_eval_seed", EVAL_SEED_START)
            seed_config.setdefault("checkpoint_eval_seed_protocol", SEED_PROTOCOL)
        seed = _required_int(
            seed_config,
            "checkpoint_eval_seed",
            label=label,
            minimum=0,
        )
        seed_protocol = str(seed_config.get("checkpoint_eval_seed_protocol") or "")
        if seed_protocol != SEED_PROTOCOL:
            raise ValueError(f"unsupported checkpoint eval seed protocol: {seed_protocol!r}")

        acceptance_value = train_config.get("checkpoint_eval_acceptance")
        acceptance: list[dict[str, Any]] | None = None
        if acceptance_value is not None:
            if not isinstance(acceptance_value, list):
                raise ValueError("checkpoint eval acceptance rules must be a list")
            acceptance = [deepcopy(dict(rule)) for rule in acceptance_value]

        asset_value = manifest_from_train_config(
            train_config,
            expected_game=str(environment.get("game") or ""),
        )
        requires_asset = resolve_env_provider(provider).requires_external_rom_asset
        asset: dict[str, Any] | None = None
        if requires_asset:
            if not isinstance(asset_value, Mapping):
                if require_asset:
                    raise ValueError("checkpoint eval asset manifest is not materialized")
            else:
                asset = validate_rom_asset_manifest(
                    asset_value,
                    expected_game=str(environment.get("game") or ""),
                )
                if portable_asset:
                    asset = {
                        key: value
                        for key, value in asset.items()
                        if key not in {"object_uri", "local_path"}
                    }
                elif not str(asset.get("object_uri") or ""):
                    raise ValueError("checkpoint eval asset manifest must include object_uri")
        elif asset_value is not None and not isinstance(asset_value, Mapping):
            raise ValueError("checkpoint eval asset manifest must be an object or null")

        return cls(
            environment=environment,
            episodes=episodes,
            n_envs=n_envs,
            max_steps=checkpoint_eval_max_steps(train_config),
            seed=seed,
            seed_protocol=seed_protocol,
            acceptance=acceptance,
            asset=asset,
        )

    def contract(self, *, require_acceptance: bool) -> dict[str, Any]:
        if require_acceptance:
            if not self.acceptance:
                raise ValueError("checkpoint eval acceptance rules are not materialized")
            return build_checkpoint_eval_contract(
                environment=self.environment,
                episodes=self.episodes,
                n_envs=self.n_envs,
                max_steps=self.max_steps,
                seed=self.seed,
                seed_protocol=self.seed_protocol,
                acceptance=self.acceptance,
                asset=self.asset,
            )
        return {
            "environment": deepcopy(self.environment),
            "action_sampling": "stochastic",
            "episodes": self.episodes,
            "n_envs": self.n_envs,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "seed_protocol": self.seed_protocol,
        }

    def announcement_payload(self, *, stages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "environment": deepcopy(self.environment),
            "stages": deepcopy(stages),
            "n_envs": self.n_envs,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "seed_protocol": self.seed_protocol,
            "asset": deepcopy(self.asset),
            "promotion_episodes": self.episodes,
        }


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
    if int(max_steps) < 1:
        raise ValueError("acceptance max_steps must be positive")
    if int(seed) < 0:
        raise ValueError("acceptance seed must be non-negative")
    if str(seed_protocol) != SEED_PROTOCOL:
        raise ValueError(f"unsupported checkpoint eval seed protocol: {seed_protocol!r}")
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
    train_config: Mapping[str, Any], *, portable_asset: bool = False
) -> dict[str, Any]:
    compiler = CheckpointEvalContractCompiler.from_train_config(
        train_config,
        portable_asset=portable_asset,
        require_asset=not portable_asset,
    )
    return compiler.contract(require_acceptance=True)


def validate_checkpoint_eval_contract(
    contract: Mapping[str, Any], *, portable_asset: bool = False
) -> dict[str, Any]:
    """Validate a serialized acceptance contract without changing its bytes or shape."""

    environment = contract.get("environment")
    acceptance = contract.get("acceptance")
    if not isinstance(environment, Mapping):
        raise ValueError("checkpoint eval contract environment is invalid")
    if not isinstance(acceptance, list):
        raise ValueError("checkpoint eval contract acceptance rules are invalid")
    if int(contract.get("protocol_version") or 0) != ACCEPTANCE_PROTOCOL_VERSION:
        raise ValueError("checkpoint eval contract protocol version mismatch")
    serialized = dict(contract)
    compiler = CheckpointEvalContractCompiler.from_train_config(
        {
            "checkpoint_eval_environment": environment,
            "post_train_eval_episodes": contract.get("episodes"),
            "checkpoint_eval_n_envs": contract.get("n_envs"),
            "post_train_eval_max_steps": contract.get("max_steps"),
            "checkpoint_eval_seed": contract.get("seed"),
            "checkpoint_eval_seed_protocol": contract.get("seed_protocol"),
            "checkpoint_eval_acceptance": acceptance,
            "rom_asset_manifest": contract.get("asset"),
        },
        portable_asset=portable_asset,
        require_asset=not portable_asset,
    )
    if compiler.contract(require_acceptance=True) != serialized:
        raise ValueError("checkpoint eval contract is not canonical")
    manifest_index(serialized)
    return deepcopy(serialized)


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
