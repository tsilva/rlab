from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from rlab.preprocessing import preprocessing_contract
from rlab.policy_bundle import (
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    build_model_document,
    build_recipe_document,
    evaluation_contract_sha256,
    sha256_file,
    write_canonical_json,
)
from rlab.recipe_documents import compose_train_document
from rlab.training_backend import training_backend_config_hash
from rlab.publication import (
    GITATTRIBUTES_TEXT,
    HUGGINGFACE_RELEASE_FILES,
    MIT_LICENSE_TEXT,
    PublicationIdentity,
    build_model_repo_id,
    build_release_manifest,
    normalize_publication_evaluation,
    publication_identity_from_model_metadata,
    publication_model_metadata,
    publication_source_from_model_metadata,
    release_artifact_records,
    render_model_card,
    upgrade_legacy_model_metadata_for_publication,
    validate_release_bundle,
)


def model_metadata(
    *,
    provider: str = "supermariobrosnes-turbo",
    game: str = "SuperMarioBros-Nes-v0",
    grayscale: bool = True,
    resize: tuple[int, int] = (84, 84),
    crop: list[int] | None = None,
    crop_mode: str = "mask",
    frame_stack: int = 4,
    layout: str = "channel_first",
    action_set: str = "simple",
    algorithm: str = "ppo",
    model_class: str = "stable_baselines3.ppo.ppo.PPO",
) -> dict:
    if crop is None and game == "SuperMarioBros-Nes-v0":
        crop = [32, 0, 0, 0]
    return {
        "metadata_version": 6,
        "algorithm_id": algorithm,
        "model_class": model_class,
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": "c" * 64,
        "seed": 7,
        "repo_git_commit": "a" * 40,
        "run_name": "bx0000000000000000-release-s7-20260714T120000Z",
        "wandb_run_id": "run123",
        "wandb_project": "SuperMarioBros-Nes-v0",
        "recipe_slug": "base",
        "checkpoint_step": 4_000_000,
        "training_metadata": {
            "environment_hash": "sha256:environment",
            "environment": {
                "env_id": f"{provider}:{game}",
                "task": {"action": {"set": action_set}},
            },
            "preprocessing": {
                "obs_resize": list(resize),
                "obs_crop": crop,
                "obs_crop_mode": crop_mode,
                "obs_grayscale": grayscale,
                "frame_stack": frame_stack,
                "policy_observation_layout": layout,
            },
        },
    }


def evaluation_payload() -> dict:
    return {
        "action_sampling": "stochastic",
        "protocol": "full",
        "eval/full/checkpoint/step": 4_000_000,
        "eval/full/checkpoint/artifact": "tsilva/project/run-checkpoint:v3",
        "eval/full/episode/count": 30,
        "eval/full/outcome/success/rate/min": 0.8,
        "eval/full/outcome/success/rate/mean": 0.9,
        "eval/full/episode/return/mean": 123.5,
        "eval/full/progress/x/max": 6256,
        "eval/full/by_start": [
            {
                "start_id": "Level1-1",
                "episodes": 15,
                "success_count": 12,
                "success_rate": 0.8,
                "return_mean": 100.0,
            },
            {
                "start_id": "Level1-2",
                "episodes": 15,
                "success_count": 15,
                "success_rate": 1.0,
                "return_mean": 147.0,
            },
        ],
    }


def test_mario_publication_identity_is_exact_and_provider_neutral() -> None:
    native = publication_identity_from_model_metadata("Level1-1", model_metadata())
    retro = publication_identity_from_model_metadata(
        "Level1-1", model_metadata(provider="stable-retro-turbo")
    )

    assert native == retro
    assert native == PublicationIdentity(
        game_family="NES-SuperMarioBros",
        goal="Level1-1",
        policy_variant="gray84-hudmask-stack4-simple",
        algorithm="ppo",
    )
    assert build_model_repo_id(native) == (
        "tsilva/NES-SuperMarioBros_Level1-1_"
        "gray84-hudmask-stack4-simple_ppo"
    )


@pytest.mark.parametrize(
    ("provider", "game", "family"),
    [
        ("stable-retro-turbo", "SuperMarioBros3-Nes-v0", "NES-SuperMarioBros3"),
        ("ale-py", "breakout", "Atari2600-Breakout"),
        ("breakout-turbo-env", "BreakoutTurbo-v0", "BreakoutTurbo"),
        ("stable-retro-turbo", "Breakout-Atari2600-v0", "Atari2600-Breakout"),
        ("ale-py", "ms_pacman", "Atari2600-MsPacman"),
    ],
)
def test_registered_game_families(provider: str, game: str, family: str) -> None:
    metadata = model_metadata(
        provider=provider, game=game, crop=[0, 0, 0, 0], action_set="native"
    )
    identity = publication_identity_from_model_metadata("Goal1", metadata)
    assert identity.game_family == family


def test_policy_variant_records_rgb_shape_crop_stack_layout_and_action() -> None:
    metadata = model_metadata(
        grayscale=False,
        resize=(84, 96),
        crop=[8, 1, 2, 3],
        crop_mode="remove",
        frame_stack=2,
        layout="dict_image_task",
        action_set="native",
    )

    identity = publication_identity_from_model_metadata("Levels_1-1_1-2", metadata)

    assert identity.goal == "Levels-1-1-1-2"
    assert identity.policy_variant == "rgb84x96-crop-t8-r1-b2-l3-stack2-taskdict-native"


def test_policy_variant_accepts_another_registered_action_set() -> None:
    identity = publication_identity_from_model_metadata(
        "Level1-1", model_metadata(action_set="right")
    )
    assert identity.policy_variant.endswith("-right")


@pytest.mark.parametrize(
    ("algorithm", "model_class"),
    [
        ("ppo", "stable_baselines3.ppo.ppo.PPO"),
        ("a2c", "stable_baselines3.a2c.a2c.A2C"),
        ("dqn", "stable_baselines3.dqn.dqn.DQN"),
        ("jerk", "rlab.jerk.JerkPolicy"),
        ("recurrent-ppo", "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO"),
    ],
)
def test_supported_algorithms_are_the_last_axis(algorithm: str, model_class: str) -> None:
    identity = publication_identity_from_model_metadata(
        "Level1-1", model_metadata(algorithm=algorithm, model_class=model_class)
    )
    assert build_model_repo_id(identity).endswith(f"_{algorithm}")


def test_publication_rejects_unknown_family_and_algorithm_mismatch() -> None:
    with pytest.raises(ValueError, match="no registered canonical game family"):
        publication_identity_from_model_metadata(
            "Goal1", model_metadata(provider="gymnasium", game="CustomVector-v0", crop=[])
        )
    with pytest.raises(ValueError, match="incompatible"):
        publication_identity_from_model_metadata(
            "Level1-1", model_metadata(algorithm="a2c", model_class="stable_baselines3.ppo.ppo.PPO")
        )
    with pytest.raises(ValueError, match="no registered canonical game family"):
        publication_identity_from_model_metadata(
            "Level1-1", model_metadata(provider="unregistered-mario-provider")
        )
    with pytest.raises(ValueError, match="unknown action_set"):
        publication_identity_from_model_metadata(
            "Level1-1", model_metadata(action_set="unregistered-actions")
        )


def test_long_repo_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="96"):
        build_model_repo_id(
            PublicationIdentity(
                game_family="NES-SuperMarioBros",
                goal="A" * 70,
                policy_variant="gray84-hudmask-stack4-simple",
                algorithm="ppo",
            )
        )


def test_preprocessing_contract_reads_provider_rgb_and_stack_arguments() -> None:
    contract = preprocessing_contract(
        {
            "env_provider": "supermariobrosnes-turbo",
            "observation_size": 96,
            "env_args": {"obs_grayscale": False, "frame_stack": 2},
        }
    )
    assert contract["obs_grayscale"] is False
    assert contract["obs_resize"] == [96, 96]
    assert contract["frame_stack"] == 2


def test_legacy_metadata_upgrade_requires_explicit_missing_facts() -> None:
    legacy = model_metadata()
    legacy.pop("algorithm_id")
    legacy.pop("model_class")
    training = legacy["training_metadata"]
    training["preprocessing"].pop("obs_crop_mode")
    training["environment"] = {
        "env_id": "supermariobrosnes-turbo:SuperMarioBros-Nes-v0",
        "action": {"action_set": "simple"},
    }

    upgraded = upgrade_legacy_model_metadata_for_publication(
        legacy,
        algorithm_id="ppo",
        model_class="stable_baselines3.ppo.ppo.PPO",
        crop_mode="remove",
    )
    identity = publication_identity_from_model_metadata("Level1-1", upgraded)

    assert identity.policy_variant == "gray84-hudcrop-stack4-simple"
    assert identity.algorithm == "ppo"


def test_publication_evaluation_requires_stochastic_consistent_by_start() -> None:
    deterministic = evaluation_payload()
    deterministic["action_sampling"] = "deterministic"
    with pytest.raises(ValueError, match="action_sampling"):
        normalize_publication_evaluation(deterministic)

    inconsistent = evaluation_payload()
    inconsistent["eval/full/outcome/success/rate/min"] = 0.7
    with pytest.raises(ValueError, match="success_rate_min"):
        normalize_publication_evaluation(inconsistent)


def test_publication_source_requires_explicit_seed_commit_and_matching_step() -> None:
    evaluation = normalize_publication_evaluation(evaluation_payload())
    metadata = model_metadata()
    metadata["repo_git_commit"] = ""
    with pytest.raises(ValueError, match="repo_git_commit"):
        publication_source_from_model_metadata(metadata, evaluation)

    metadata = model_metadata()
    metadata["checkpoint_step"] = 1
    with pytest.raises(ValueError, match="checkpoint_step disagrees"):
        publication_source_from_model_metadata(metadata, evaluation)


def test_release_bundle_has_exact_files_hashes_and_portable_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw_metadata = model_metadata()
        identity = publication_identity_from_model_metadata("Level1-1", raw_metadata)
        metadata = publication_model_metadata(raw_metadata, identity)
        evaluation = normalize_publication_evaluation(evaluation_payload())
        source = publication_source_from_model_metadata(metadata, evaluation)
        contents = {
            ".gitattributes": GITATTRIBUTES_TEXT,
            "LICENSE": MIT_LICENSE_TEXT,
            "model.zip": "checkpoint",
            "replay.mp4": "video",
        }
        for filename, content in contents.items():
            (root / filename).write_text(content, encoding="utf-8")
        composed = compose_train_document(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml"),
            Path("experiments/recipes/mario/single/ppo.yaml"),
        )
        recipe_document = build_recipe_document(
            composed,
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            run_description="release fixture",
            seed=7,
            runtime_image_ref="docker:example.invalid/rlab@sha256:" + "b" * 64,
        )
        write_canonical_json(root / "recipe.json", recipe_document)
        metadata["training_backend_id"] = recipe_document["recipe"]["train_config"][
            "training_backend"
        ]["id"]
        metadata["training_backend_config_hash"] = training_backend_config_hash(
            recipe_document["recipe"]["train_config"]
        )
        metadata["training_metadata"] = {
            "environment_hash": recipe_document["recipe"]["environment_hash"],
            "environment": recipe_document["recipe"]["environment"],
            "preprocessing": recipe_document["recipe"]["environment"]["preprocessing"],
        }
        write_canonical_json(
            root / "model.json",
            build_model_document(root / "model.zip", root / "recipe.json", metadata),
        )
        evaluation_value = evaluation.as_manifest_value()
        evaluation_value.update(
            checkpoint_sha256=sha256_file(root / "model.zip"),
            recipe_sha256=sha256_file(root / "recipe.json"),
            recipe_format_version=recipe_document["format_version"],
            evaluation_contract_sha256=evaluation_contract_sha256(recipe_document),
            exact_contract=True,
        )
        provisional = build_release_manifest(
            identity,
            metadata,
            release_version="v1",
            published_at="2026-07-14T12:00:00Z",
            source=source,
            evaluation=evaluation_value,
            artifacts={},
            youtube_url="https://www.youtube.com/watch?v=example",
        )
        (root / "README.md").write_text(
            render_model_card(provisional, metadata), encoding="utf-8"
        )
        records = release_artifact_records(root)
        manifest = build_release_manifest(
            identity,
            metadata,
            release_version="v1",
            published_at="2026-07-14T12:00:00Z",
            source=source,
            evaluation=evaluation_value,
            artifacts=records,
            youtube_url="https://www.youtube.com/watch?v=example",
        )
        (root / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        assert {path.name for path in root.iterdir()} == HUGGINGFACE_RELEASE_FILES
        assert validate_release_bundle(root) == manifest

        future = deepcopy(manifest)
        future["format_version"] = 999
        write_canonical_json(root / "release_manifest.json", future)
        with patch(
            "rlab.publication.load_policy_bundle",
            side_effect=AssertionError("bundle access"),
        ):
            with pytest.raises(UnsupportedPolicyDocumentVersion, match="999"):
                validate_release_bundle(root)

        malformed = deepcopy(manifest)
        malformed["evaluation"]["unexpected_contract_field"] = True
        write_canonical_json(root / "release_manifest.json", malformed)
        with pytest.raises(PolicyDocumentError, match="unknown field"):
            validate_release_bundle(root)

        broken = deepcopy(manifest)
        broken["source"]["checkpoint_artifact"] = "/Users/example/model.zip"
        (root / "release_manifest.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="absolute local path"):
            validate_release_bundle(root)


def test_release_bundle_rejects_non_file_entries() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for filename in HUGGINGFACE_RELEASE_FILES - {"replay.mp4"}:
            (root / filename).write_text("placeholder", encoding="utf-8")
        (root / "replay.mp4").mkdir()

        with pytest.raises(ValueError, match="regular files"):
            validate_release_bundle(root)
