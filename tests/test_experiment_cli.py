from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlab.dstack_backend import DstackTask
from rlab.experiment_cli import (
    _bind_launch_contract,
    _compute,
    _follow_fingerprint,
    _public_dstack_state,
    _stage_rom,
    _task_name,
    _task_request,
    build_parser,
)
from rlab.policy_bundle import build_recipe_document
from rlab.recipe_documents import compose_train_document
from rlab.run_contracts import new_attempt_id, new_run_id


def test_launch_parser_exposes_bounded_compute_and_hash_bound_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "launch",
            "--goal-file",
            "experiments/goals/goal/_goal.yaml",
            "--recipe-file",
            "experiments/goals/goal/recipes/ppo.yaml",
            "--seed",
            "123",
            "--run-description",
            "one isolated learning-rate ablation",
            "--set",
            "train.backend.config.learning_rate=0.0002",
            "--compute",
            "spot",
            "--target",
            "aws",
            "--max-price",
            "1.25",
            "--max-cost-usd",
            "5",
            "--max-duration",
            "8h",
        ]
    )

    assert args.recipe_overrides == ["train.backend.config.learning_rate=0.0002"]
    assert args.checkpoint_eval_backend == "modal"
    compute = _compute(args)
    assert compute.kind == "spot"
    assert compute.target == "aws"
    assert compute.max_duration_seconds == 8 * 60 * 60
    assert compute.bounded_duration_seconds == 4 * 60 * 60


def test_auto_without_cloud_budget_stays_local() -> None:
    compute = _compute(
        SimpleNamespace(
            compute="auto",
            target=None,
            max_price=None,
            max_cost_usd=None,
            allow_on_demand=False,
            max_duration=3600,
        )
    )
    assert compute.kind == "auto"
    assert compute.max_price is None
    assert compute.bounded_duration_seconds == 3600


def test_public_dstack_state_never_exposes_raw_task_environment() -> None:
    value = _public_dstack_state(
        DstackTask(
            project="main",
            name="run-one",
            status="running",
            raw={
                "fleet": {"name": "b3"},
                "submitted_at": "2026-07-24T16:00:00Z",
                "run_spec": {
                    "configuration": {
                        "env": {
                            "WANDB_API_KEY": "should-never-appear",
                            "RLAB_CONTROL_R2_SECRET_ACCESS_KEY": "also-secret",
                        }
                    }
                },
            },
        )
    )

    encoded = json.dumps(value, sort_keys=True)
    assert value["fleet"] == "b3"
    assert "raw" not in value
    assert "should-never-appear" not in encoded
    assert "also-secret" not in encoded


def test_follow_fingerprint_ignores_only_poll_observation_time() -> None:
    first = {
        "run_id": "rlab-" + "a" * 32,
        "semantic": {
            "observed_at": 1.0,
            "attempts": [{"attempt_id": "attempt-" + "b" * 16}],
        },
    }
    second = {
        **first,
        "semantic": {**first["semantic"], "observed_at": 2.0},
    }

    assert _follow_fingerprint(first) == _follow_fingerprint(second)
    second["semantic"]["attempts"] = [
        {"attempt_id": "attempt-" + "b" * 16},
        {"attempt_id": "attempt-" + "c" * 16},
    ]
    assert _follow_fingerprint(first) != _follow_fingerprint(second)


def test_on_demand_requires_explicit_permission() -> None:
    with pytest.raises(ValueError, match="requires --allow-on-demand"):
        _compute(
            SimpleNamespace(
                compute="on-demand",
                target="aws",
                max_price=2.0,
                max_cost_usd=10.0,
                allow_on_demand=False,
                max_duration=3600,
            )
        )


def test_retry_task_name_preserves_run_and_changes_attempt() -> None:
    run_id = new_run_id()
    attempt_id = new_attempt_id()

    assert _task_name(run_id, attempt_id, initial=True) == run_id
    retry_name = _task_name(run_id, attempt_id, initial=False)
    assert retry_name.startswith(run_id + "-a")
    assert retry_name.endswith(attempt_id.removeprefix("attempt-"))
    assert len(retry_name) <= 63


def test_launch_parser_supports_explicit_training_only_runs() -> None:
    args = build_parser().parse_args(
        [
            "launch",
            "--goal-file",
            "experiments/goals/goal/_goal.yaml",
            "--recipe-file",
            "experiments/goals/goal/recipes/ppo.yaml",
            "--seed",
            "123",
            "--run-description",
            "training-only search rung",
            "--checkpoint-eval-backend",
            "none",
            "--submission-key",
            "autoresearch-study-rung-1",
        ]
    )

    assert args.checkpoint_eval_backend == "none"
    assert args.submission_key == "autoresearch-study-rung-1"


def test_training_only_task_does_not_receive_modal_credentials() -> None:
    manifest = SimpleNamespace(
        run_id=new_run_id(),
        image_digest="docker:example/rlab@sha256:" + "a" * 64,
        compute={
            "selected": {
                "kind": "local",
                "target": "b3",
                "max_price": None,
                "max_cost_usd": None,
                "allow_on_demand": False,
                "max_duration_seconds": 3600,
            },
            "dstack_task": "training-only",
        },
        modal={"enabled": False, "environment_name": "rlab-eval"},
    )

    task = _task_request(manifest, manifest_uri="s3://control/run/manifest.json")

    assert "MODAL_TOKEN_ID" not in task.secret_env
    assert "MODAL_TOKEN_SECRET" not in task.secret_env
    assert not any(value.startswith("MODAL_ENVIRONMENT=") for value in task.secret_env)
    assert task.rom_mount is None


def test_rom_free_provider_does_not_require_or_stage_an_asset() -> None:
    assert (
        _stage_rom(
            SimpleNamespace(),
            env_provider="rlab",
            game="Bandit-v0",
            rom_path=None,
        )
        is None
    )


def test_rom_free_launch_contract_omits_null_asset() -> None:
    goal = Path("experiments/goals/rlab__bandit/_goal.yaml")
    document = compose_train_document(goal, goal.parent / "recipes/ppo.yaml")
    contract = _bind_launch_contract(
        document,
        asset=None,
        checkpoint_eval_backend="none",
    )

    assert "rom_asset_manifest" not in contract["train_config"]
    build_recipe_document(
        contract,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="ROM-free launch contract regression",
        seed=123,
        runtime_image_ref="docker:example/rlab@sha256:" + "b" * 64,
    )
