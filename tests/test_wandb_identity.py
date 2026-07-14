from __future__ import annotations

import argparse
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlab.artifacts import init_wandb
from rlab.env import EnvConfig
from rlab.wandb_utils import (
    canonical_wandb_environment,
    game_family_for_environment,
    resolve_wandb_project,
)


@pytest.mark.parametrize(
    ("provider", "game", "project", "family"),
    [
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "super-mario-bros-nes",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "super-mario-bros-nes",
        ),
        ("ale-py", "breakout", "Breakout-Atari2600-v0", "breakout-atari2600"),
        (
            "stable-retro-turbo",
            "Breakout-Atari2600-v0",
            "Breakout-Atari2600-v0",
            "breakout-atari2600",
        ),
        ("ale-py", "ms_pacman", "MsPacman-Atari2600-v0", "ms-pacman-atari2600"),
        (
            "stable-retro-turbo",
            "MsPacman-Atari2600-v0",
            "MsPacman-Atari2600-v0",
            "ms-pacman-atari2600",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros3-Nes-v0",
            "SuperMarioBros3-Nes-v0",
            "super-mario-bros-3-nes",
        ),
    ],
)
def test_canonical_wandb_environment_mapping(provider, game, project, family) -> None:
    assert canonical_wandb_environment(provider, game) == (project, family)


def test_explicit_project_wins_and_unknown_environment_falls_back() -> None:
    assert (
        resolve_wandb_project("custom-project", "breakout", env_provider="ale-py")
        == "custom-project"
    )
    assert resolve_wandb_project(None, "CustomNativeVector-v0", env_provider="gymnasium") == (
        "CustomNativeVector-v0"
    )
    assert game_family_for_environment("gymnasium", "CustomNativeVector-v0") == (
        "custom-native-vector-v0"
    )


def test_init_wandb_records_resolved_identity_and_submission_group() -> None:
    captured = {}

    class FakeRun:
        def define_metric(self, *_args, **_kwargs) -> None:
            return None

    def fake_init(**kwargs):
        captured.update(kwargs)
        return FakeRun()

    args = argparse.Namespace(
        wandb=True,
        wandb_tags="goal_id:alepy__breakout,recipe_id:base",
        wandb_entity="entity",
        wandb_project=None,
        wandb_group="bx0123456789abcdef",
        run_name="bx0123456789abcdef-base-s123-20260714T120000Z",
        run_description="offline identity canary",
        wandb_mode="offline",
        wandb_run_id="rlab-0123456789abcdef01234567",
    )
    config = EnvConfig(
        env_provider="ale-py",
        game="breakout",
        state=None,
    )

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("rlab.artifacts.load_wandb_env"),
        patch.dict(sys.modules, {"wandb": SimpleNamespace(init=fake_init)}),
    ):
        init_wandb(args, tmp, config)

    assert captured["project"] == "Breakout-Atari2600-v0"
    assert captured["group"] == "bx0123456789abcdef"
    assert captured["id"] == "rlab-0123456789abcdef01234567"
    assert captured["name"] == args.run_name
    assert captured["config"]["wandb_project"] == "Breakout-Atari2600-v0"
    assert captured["config"]["game_family"] == "breakout-atari2600"
    assert captured["config"]["environment"]["env_id"] == "ale-py:breakout"
    assert "environment_hash" in captured["config"]
    assert "game_family:breakout-atari2600" in captured["tags"]
