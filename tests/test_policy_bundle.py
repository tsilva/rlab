from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from rlab.policy_bundle import (
    MODEL_DOCUMENT_TYPE,
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    build_model_document,
    build_recipe_document,
    canonical_json_bytes,
    evaluation_contract,
    load_policy_bundle,
    load_recipe_document,
    validate_recipe_document,
    write_canonical_json,
)
from rlab.eval_runner import normalized_evaluation_request
from rlab.recipe_documents import compose_train_document
from rlab.train_config import validate_and_normalize_train_config
from rlab.training_backend import training_backend_config, training_backend_config_hash


GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = Path("experiments/recipes/mario/single/ppo.yaml")
RUNTIME = "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "b" * 64


def level1_1_recipe_document(*, seed: int = 7) -> dict:
    materialized = compose_train_document(GOAL, RECIPE)
    return build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description=f"Level1-1 PPO seed {seed}",
        seed=seed,
        runtime_image_ref=RUNTIME,
    )


def write_bundle(root: Path) -> None:
    checkpoint = root / "model.zip"
    checkpoint.write_bytes(b"checkpoint bytes")
    recipe_document = level1_1_recipe_document()
    recipe_path = write_canonical_json(root / "recipe.json", recipe_document)
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 500_000,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        ),
        "repo_git_commit": "a" * 40,
    }
    write_canonical_json(
        root / "model.json",
        build_model_document(checkpoint, recipe_path, metadata),
    )


def test_level1_1_recipe_fixture_preserves_distinct_train_and_eval_contracts() -> None:
    document = level1_1_recipe_document()
    train_task = document["recipe"]["train_config"]["task"]
    eval_contract = evaluation_contract(document)
    eval_task = eval_contract["environment"]["task"]

    assert train_task["termination"]["failure"] == ["life_loss", "stalled"]
    assert eval_task["termination"]["failure"] == ["stalled"]
    assert document["recipe"]["train_config"]["obs_crop"] == [32, 0, 0, 0]
    assert eval_contract["action_sampling"] == "stochastic"
    assert eval_contract["seed_protocol"] == "vector-lane-v1"
    assert eval_contract["episodes"] == 100


def test_recipe_materializes_the_backend_config_executed_by_the_learner() -> None:
    materialized = compose_train_document(GOAL, RECIPE)
    document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="normalized backend contract",
        seed=7,
        runtime_image_ref=RUNTIME,
    )

    recipe_train_config = document["recipe"]["train_config"]
    executed_train_config = validate_and_normalize_train_config(materialized["train_config"])

    assert training_backend_config(recipe_train_config) == training_backend_config(
        executed_train_config
    )
    assert training_backend_config_hash(recipe_train_config) == training_backend_config_hash(
        executed_train_config
    )
    assert training_backend_config(recipe_train_config)["device"] == "auto"


@pytest.mark.parametrize(
    "provider",
    ["supermariobrosnes-turbo", "stable-retro-turbo"],
)
def test_recipe_provider_is_exact_and_never_falls_back(provider: str) -> None:
    document = deepcopy(level1_1_recipe_document())

    def replace_provider(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "env_provider":
                    value[key] = provider
                elif key == "env_id" and isinstance(nested, str) and ":" in nested:
                    value[key] = f"{provider}:{nested.split(':', 1)[1]}"
                else:
                    replace_provider(nested)
        elif isinstance(value, list):
            for nested in value:
                replace_provider(nested)

    replace_provider(document["recipe"])
    validated = validate_recipe_document(document, source=f"{provider} recipe fixture")

    assert validated["recipe"]["train_config"]["env_provider"] == provider
    assert evaluation_contract(validated)["environment"]["env_provider"] == provider


def test_canonical_recipe_bytes_are_deterministic_and_newline_terminated() -> None:
    document = level1_1_recipe_document()
    assert canonical_json_bytes(document) == canonical_json_bytes(
        json.loads(canonical_json_bytes(document))
    )
    assert canonical_json_bytes(document).endswith(b"\n")
    assert b" " not in canonical_json_bytes({"z": 1, "a": 2})


def test_future_recipe_version_fails_with_source_and_supported_versions(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "document_type": "rlab.recipe",
                "format_version": 999,
                "recipe": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedPolicyDocumentVersion) as error:
        load_recipe_document(path)
    message = str(error.value)
    assert str(path) in message
    assert "999" in message
    assert "[1]" in message
    assert "Upgrade rlab" in message


def test_future_model_version_fails_before_checkpoint_access(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    model_path = tmp_path / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["format_version"] = 2
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with patch("rlab.policy_bundle.sha256_file", side_effect=AssertionError("checkpoint read")):
        with pytest.raises(UnsupportedPolicyDocumentVersion) as error:
            load_policy_bundle(tmp_path)
    assert MODEL_DOCUMENT_TYPE in str(error.value)
    assert "format_version 2" in str(error.value)


def test_known_recipe_schema_rejects_unknown_fields_and_urls(tmp_path: Path) -> None:
    document = level1_1_recipe_document()
    document["unknown"] = True
    write_canonical_json(tmp_path / "recipe.json", document)
    with pytest.raises(PolicyDocumentError, match="unknown field"):
        load_recipe_document(tmp_path / "recipe.json")

    document = level1_1_recipe_document()
    document["provenance"]["runtime"]["packages"] = {
        "bad": "https://example.invalid/policy"
    }
    write_canonical_json(tmp_path / "recipe.json", document)
    with pytest.raises(PolicyDocumentError, match="URL"):
        load_recipe_document(tmp_path / "recipe.json")


def test_bundle_rejects_checkpoint_and_recipe_hash_mismatches(tmp_path: Path) -> None:
    write_bundle(tmp_path)
    (tmp_path / "model.zip").write_bytes(b"changed")
    with pytest.raises(PolicyDocumentError, match="model.zip hash"):
        load_policy_bundle(tmp_path)

    write_bundle(tmp_path)
    recipe = json.loads((tmp_path / "recipe.json").read_text(encoding="utf-8"))
    recipe["recipe"]["description"] = "changed"
    write_canonical_json(tmp_path / "recipe.json", recipe)
    with pytest.raises(PolicyDocumentError, match="recipe.json hash"):
        load_policy_bundle(tmp_path)


def test_all_source_kinds_normalize_to_identical_eval_and_seed_requests(
    tmp_path: Path,
) -> None:
    write_bundle(tmp_path)
    local = load_policy_bundle(tmp_path, source=str(tmp_path))
    sources = (
        local,
        replace(local, source="wandb://entity/project/artifact:v7", revision="v7"),
        replace(local, source="training://job/42/checkpoint/8", revision="ledger-8"),
        replace(local, source="hf://tsilva/policy", revision="d" * 40),
    )
    requests = [
        normalized_evaluation_request(bundle, episodes=5, n_envs=1)
        for bundle in sources
    ]
    assert requests[1:] == requests[:-1]
    assert len(requests[0]["seed_assignments"]) == 5
    assert requests[0]["seed_assignments"][0] == {
        "lane": 0,
        "lane_episode_ordinal": 0,
        "seed": 10_000,
    }
