from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_mapping
from rlab.env_metadata import training_metadata
from rlab.policy_bundle import (
    MODEL_DOCUMENT_TYPE,
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    build_model_document,
    build_recipe_document,
    canonical_json_bytes,
    evaluation_contract,
    playback_contract,
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
RECIPE = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml")
RUNTIME = "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "b" * 64
POST400_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/post400-r400/_goal.yaml")
POST400_RECIPE = POST400_GOAL.parent / "recipes/ppo-resume-129991680.yaml"
BREAKOUT_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
BREAKOUT_RECIPES = tuple(sorted((BREAKOUT_GOAL.parent / "recipes").glob("*.yaml")))


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


@pytest.mark.parametrize("recipe_path", BREAKOUT_RECIPES)
def test_breakout_bundle_is_playable_but_has_no_evaluation_contract(
    recipe_path: Path,
) -> None:
    materialized = compose_train_document(BREAKOUT_GOAL, recipe_path)
    document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="training-only Breakout",
        seed=7,
        runtime_image_ref=RUNTIME,
    )

    assert document["recipe"]["train_config"]["checkpoint_eval_backend"] == "none"
    assert "eval" not in document["recipe"]
    assert playback_contract(document)["environment"]["game"] == "Breakout-Atari2600-v0"
    with pytest.raises(PolicyDocumentError, match="no evaluation contract"):
        evaluation_contract(document)


def test_post400_acceptance_assigns_every_snapshot_to_a_fixed_lane() -> None:
    materialized = compose_train_document(POST400_GOAL, POST400_RECIPE)
    document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="post-400 acceptance lane regression",
        seed=123,
        runtime_image_ref=RUNTIME,
    )

    contract = evaluation_contract(document)
    starts = [entry["start_state"] for entry in contract["manifest"]["episodes"]]
    counts = Counter(starts)
    assert contract["n_envs"] == 32
    assert len(starts) == 100
    assert len(counts) == 32
    assert set(counts.values()) == {3, 4}


def test_bundle_omits_unselected_reward_default_and_accepts_legacy_false(
    tmp_path: Path,
) -> None:
    materialized = compose_train_document(POST400_GOAL, POST400_RECIPE)
    recipe_document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="no reward catalog regression",
        seed=123,
        runtime_image_ref=RUNTIME,
    )
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint bytes")
    recipe_path = write_canonical_json(tmp_path / "recipe.json", recipe_document)
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 500_000,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        ),
        "reward_shape_is_default": False,
    }
    model_document = build_model_document(checkpoint, recipe_path, metadata)
    assert "reward_shape_is_default" not in model_document["provenance"]

    # Historical bundles emitted the argparse False default even when no reward
    # program was selected; keep those exact artifacts evaluable.
    model_document["provenance"]["reward_shape_is_default"] = False
    write_canonical_json(tmp_path / "model.json", model_document)
    load_policy_bundle(tmp_path)


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
    assert eval_task["termination"]["failure"] == []
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


def test_recipe_materializes_the_environment_identity_executed_by_the_learner() -> None:
    document = level1_1_recipe_document()
    recipe = document["recipe"]
    effective_training_metadata = training_metadata(
        resolve_env_config(env_config_from_mapping(recipe["train_config"]))
    )

    assert recipe["environment"] == effective_training_metadata["environment"]
    assert recipe["environment_hash"] == effective_training_metadata["environment_hash"]
    assert recipe["environment"]["states"] == []
    assert recipe["environment"]["state_probs"] == []


def test_recipe_keeps_eval_asset_identity_but_removes_private_locations() -> None:
    materialized = compose_train_document(GOAL, RECIPE)
    materialized["train_config"]["checkpoint_eval_asset_manifest"] = {
        "schema_version": 1,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "sha256": "c" * 64,
        "provider_rom_identity": "d" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        "object_uri": "s3://private-bucket/mario.nes",
        "local_path": "/private/roms/mario.nes",
    }

    document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="portable evaluation asset",
        seed=7,
        runtime_image_ref=RUNTIME,
    )

    expected_asset = {
        "schema_version": 1,
        "game": "SuperMarioBros-Nes-v0",
        "filename": "mario.nes",
        "sha256": "c" * 64,
        "provider_rom_identity": "d" * 40,
        "provider_rom_identity_algorithm": "sha1-provider-body-v1",
    }
    assert evaluation_contract(document)["asset"] == expected_asset
    assert document["provenance"]["asset"] == expected_asset


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
