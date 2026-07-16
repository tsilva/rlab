from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from rlab.checkpoint_acceptance import (
    acceptance_aggregates,
    aggregates_match,
    build_checkpoint_eval_contract,
    manifest_index,
    validate_episode_rows,
)
from rlab.eval_capacity_policy import load_eval_capacity_policy
from rlab.modal_eval_protocol import execution_key


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


def test_capacity_policy_derives_immutable_load_and_minimum_interval() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "policy.yaml"
        path.write_text(
            """
schema_version: 1
backend: modal
effective_capacity: 3
hard_safety_cap: 20
admission_utilization: 0.8
workloads:
  SuperMarioBros-Nes-v0:Level1-1:acceptance-v1:
    status: accepted
    training_fps_upper_bound: 6000
    full_eval_p95_seconds: 40
    selected_checkpoint_interval_steps: 250000
""".strip()
            + "\n",
            encoding="utf-8",
        )
        policy = load_eval_capacity_policy(path)
        reservation = policy.reservation(
            {
                "game": "SuperMarioBros-Nes-v0",
                "state": "Level1-1",
                "checkpoint_eval_backend": "modal",
                "stop_on_acceptance": True,
                "checkpoint_freq": 250_000,
            }
        )

    assert reservation is not None
    assert reservation["minimum_interval_steps"] == 100_000
    assert reservation["eval_load"] == pytest.approx(0.96)
    assert reservation["policy_sha256"] == policy.sha256
