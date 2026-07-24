from __future__ import annotations

import copy

import pytest

from rlab.checkpoint_acceptance import (
    CheckpointEvalContractCompiler,
    acceptance_aggregates,
    aggregates_match,
    build_checkpoint_eval_contract,
    manifest_index,
    validate_episode_rows,
)
from rlab.checkpoint_eval_config import checkpoint_eval_max_steps
from rlab.modal_eval_protocol import (
    SEED_PROTOCOL,
    execution_key,
)


def contract(*, episodes: int = 100, n_envs: int = 16) -> dict:
    return build_checkpoint_eval_contract(
        environment={"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
        episodes=episodes,
        n_envs=n_envs,
        max_steps=4500,
        seed=10_000,
        seed_protocol="vector-lane-v1",
        acceptance=[
            {
                "metric": "eval/full/outcome/success/rate/min",
                "operator": ">=",
                "threshold": 1.0,
            }
        ],
    )


def row(entry: dict, *, success: bool = True) -> dict:
    return {
        "episode_id": entry["episode_id"],
        "seed_lane": entry["lane"],
        "seed_episode_ordinal": entry["lane_episode_ordinal"],
        "seed": entry["seed"],
        "start_state": entry["start_state"],
        "outcome": "success" if success else "failure",
        "seed_protocol": "vector-lane-v1",
    }


def test_manifest_has_exact_count_unique_identities_and_fixed_quotas() -> None:
    value = contract()
    manifest = value["manifest"]

    assert manifest["lane_quotas"] == [7, 7, 7, 7] + [6] * 12
    assert len(manifest_index(value)) == 100
    assert len({entry["episode_id"] for entry in manifest["episodes"]}) == 100
    assert {entry["start_state"] for entry in manifest["episodes"]} == {"Level1-1"}


def test_rejection_is_valid_partial_evidence_only_through_first_failure() -> None:
    value = contract(episodes=4, n_envs=2)
    entries = value["manifest"]["episodes"]
    rows = [row(entries[0]), row(entries[1], success=False)]

    assert validate_episode_rows(rows, contract=value, verdict="rejected") == rows
    aggregates = acceptance_aggregates(rows, contract=value)
    assert aggregates["episodes_planned"] == 4
    assert aggregates["episodes_completed"] == 2
    assert aggregates["failure_count"] == 1
    assert not any(name.startswith("eval/full/") for name in aggregates)

    with pytest.raises(ValueError, match="after its first failure"):
        validate_episode_rows(
            [*rows, row(entries[2])], contract=value, verdict="rejected"
        )

def test_accepted_evidence_requires_every_identity_once_and_all_successes() -> None:
    value = contract(episodes=4, n_envs=2)
    rows = [row(entry) for entry in value["manifest"]["episodes"]]

    validate_episode_rows(rows, contract=value, verdict="accepted")
    with pytest.raises(ValueError, match="unknown or duplicate"):
        validate_episode_rows([*rows[:-1], rows[0]], contract=value, verdict="accepted")
    with pytest.raises(ValueError, match="every planned successful"):
        validate_episode_rows(rows[:-1], contract=value, verdict="accepted")


def test_claimed_aggregate_mismatch_is_detected() -> None:
    value = contract(episodes=2, n_envs=1)
    rows = [row(entry) for entry in value["manifest"]["episodes"]]
    computed = acceptance_aggregates(rows, contract=value)
    claimed = {**computed, "success_count": 1}

    assert aggregates_match(computed, computed)
    assert not aggregates_match(claimed, computed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["acceptance"][0].update(threshold=0.99),
        lambda value: value["manifest"]["episodes"][0].update(seed=123),
        lambda value: value["evidence_policy"].update(fail_fast="disabled"),
    ],
)
def test_execution_key_changes_for_acceptance_or_evidence_changes(mutation) -> None:
    baseline = contract(episodes=2, n_envs=1)
    changed = copy.deepcopy(baseline)
    mutation(changed)

    assert execution_key(changed) != execution_key(baseline)


def test_checkpoint_eval_max_steps_prefers_explicit_then_environment() -> None:
    environment = {"task": {"termination": {"max_episode_steps": 50}}}

    assert checkpoint_eval_max_steps(
        {
            "post_train_eval_max_steps": 40,
            "checkpoint_eval_environment": environment,
        }
    ) == 40
    assert checkpoint_eval_max_steps({"checkpoint_eval_environment": environment}) == 50

    with pytest.raises(ValueError, match="not materialized"):
        checkpoint_eval_max_steps({"checkpoint_eval_environment": {}})
    with pytest.raises(ValueError, match="not materialized"):
        checkpoint_eval_max_steps({"post_train_eval_max_steps": 0})


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("post_train_eval_episodes", "post_train_eval_episodes must be an integer"),
        ("checkpoint_eval_n_envs", "checkpoint_eval_n_envs must be an integer"),
        ("checkpoint_eval_seed", "checkpoint_eval_seed must be an integer"),
        ("checkpoint_eval_seed_protocol", "unsupported checkpoint eval seed protocol"),
    ],
)
def test_checkpoint_eval_compiler_rejects_missing_materialized_fields(
    field: str, expected: str
) -> None:
    config = {
        "checkpoint_eval_environment": {
            "env_provider": "rlab",
            "game": "Bandit-v0",
            "task": {"termination": {"max_episode_steps": 50}},
        },
        "post_train_eval_episodes": 3,
        "checkpoint_eval_n_envs": 2,
        "checkpoint_eval_seed": 10_000,
        "checkpoint_eval_seed_protocol": SEED_PROTOCOL,
    }
    config.pop(field)

    with pytest.raises(ValueError, match=expected):
        CheckpointEvalContractCompiler.from_train_config(config)


def test_checkpoint_eval_compiler_rejects_more_lanes_than_episodes() -> None:
    with pytest.raises(ValueError, match="n_envs must not exceed episodes"):
        CheckpointEvalContractCompiler.from_train_config(
            {
                "checkpoint_eval_environment": {
                    "env_provider": "rlab",
                    "game": "Bandit-v0",
                    "task": {"termination": {"max_episode_steps": 50}},
                },
                "post_train_eval_episodes": 1,
                "checkpoint_eval_n_envs": 2,
                "checkpoint_eval_seed": 10_000,
                "checkpoint_eval_seed_protocol": SEED_PROTOCOL,
            }
        )
