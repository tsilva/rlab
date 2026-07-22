from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rlab.config_loader import load_mapping_document
from rlab.config_validation import load_goal_contract
from rlab.recipe_documents import compose_train_document
from rlab.reward_programs import validate_reward_shape_catalog


MARIO_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
MARIO_RECIPE = MARIO_GOAL.parent / "recipes/ppo.yaml"
BREAKOUT_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
BREAKOUT_RECIPE = BREAKOUT_GOAL.parent / "recipes/ppo.yaml"


def test_mario_reward_shape_defaults_and_cli_override_materialize_both_phases() -> None:
    default = compose_train_document(MARIO_GOAL, MARIO_RECIPE)
    selected = compose_train_document(
        MARIO_GOAL,
        MARIO_RECIPE,
        recipe_overrides=("reward_shape=score-step-0p01-v1",),
    )

    default_config = default["train_config"]
    selected_config = selected["train_config"]
    assert default_config["reward_shape"] == "score-v1"
    assert default_config["reward_shape_is_default"] is True
    assert default_config["task"]["reward"]["time_penalty"] == 0.0
    assert selected_config["reward_shape"] == "score-step-0p01-v1"
    assert selected_config["reward_shape_is_default"] is False
    assert selected_config["task"]["reward"]["time_penalty"] == 0.01
    assert selected_config["checkpoint_eval_environment"]["task"]["reward"]["time_penalty"] == 0.01
    assert default_config["reward_shape_sha256"] != selected_config["reward_shape_sha256"]
    assert default_config["goal_contract_sha256"] == selected_config["goal_contract_sha256"]
    assert (
        default_config["effective_goal_contract_sha256"]
        != selected_config["effective_goal_contract_sha256"]
    )
    assert "reward_shapes" not in selected["goal"]


def test_catalog_selector_and_raw_reward_override_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown reward_shape"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("reward_shape=missing-v1",),
        )
    with pytest.raises(ValueError, match="reject raw reward overrides"):
        compose_train_document(
            MARIO_GOAL,
            MARIO_RECIPE,
            recipe_overrides=("train.environment.task.reward.time_penalty=0.5",),
        )


def test_non_catalog_goal_remains_compatible() -> None:
    document = compose_train_document(BREAKOUT_GOAL, BREAKOUT_RECIPE)
    assert "reward_shape" not in document["train_config"]
    with pytest.raises(ValueError, match="does not define reward_shapes"):
        compose_train_document(
            BREAKOUT_GOAL,
            BREAKOUT_RECIPE,
            recipe_overrides=("reward_shape=score-v1",),
        )


def test_catalog_definitions_are_complete_strict_and_semantically_unique() -> None:
    goal = load_goal_contract(MARIO_GOAL)
    malformed = copy.deepcopy(goal)
    malformed["reward_shapes"]["definitions"]["score-v1"]["clip_rewards"] = 1
    with pytest.raises(ValueError, match="clip_rewards must be a boolean"):
        validate_reward_shape_catalog(malformed)

    incomplete = copy.deepcopy(goal)
    del incomplete["reward_shapes"]["definitions"]["score-v1"]["death_penalty"]
    with pytest.raises(ValueError, match="missing required field.*death_penalty"):
        validate_reward_shape_catalog(incomplete)

    duplicate = copy.deepcopy(goal)
    duplicate["reward_shapes"]["definitions"]["alias-v1"] = copy.deepcopy(
        duplicate["reward_shapes"]["definitions"]["score-v1"]
    )
    with pytest.raises(ValueError, match="identical executable semantics"):
        validate_reward_shape_catalog(duplicate)


def test_yaml_loader_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("reward_shape: score-v1\nreward_shape: other-v1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key 'reward_shape'"):
        load_mapping_document(path)
