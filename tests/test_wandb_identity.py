from __future__ import annotations

import argparse
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlab.env import EnvConfig
from rlab.wandb_publisher import _start_wandb
from rlab.wandb_utils import (
    canonical_wandb_environment,
    game_family_for_environment,
    resolve_wandb_project,
)


@pytest.mark.parametrize(
    ("provider", "game", "project", "family"),
    [
        ("rlab", "Bandit-v0", "Bandit-v0", "Bandit"),
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "NES-SuperMarioBros",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros-Nes-v0",
            "NES-SuperMarioBros",
        ),
        ("ale-py", "breakout", "Breakout-Atari2600-v0", "Atari2600-Breakout"),
        (
            "breakout-turbo-env",
            "BreakoutTurbo-v0",
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
        ),
        (
            "stable-retro-turbo",
            "Breakout-Atari2600-v0",
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
        ),
        ("ale-py", "ms_pacman", "MsPacman-Atari2600-v0", "Atari2600-MsPacman"),
        (
            "stable-retro-turbo",
            "MsPacman-Atari2600-v0",
            "MsPacman-Atari2600-v0",
            "Atari2600-MsPacman",
        ),
        (
            "stable-retro-turbo",
            "SuperMarioBros3-Nes-v0",
            "SuperMarioBros3-Nes-v0",
            "NES-SuperMarioBros3",
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


def test_historical_env_id_fallbacks_are_preserved() -> None:
    assert canonical_wandb_environment(None, "SuperMarioBros-Nes-v0") == (
        "SuperMarioBros-Nes-v0",
        "NES-SuperMarioBros",
    )
    assert canonical_wandb_environment("legacy-provider", "breakout") == (
        "breakout",
        "Atari2600-Breakout",
    )
    with pytest.raises(ValueError, match="no registered canonical game family"):
        game_family_for_environment(None, "SuperMarioBros-Nes-v0", strict=True)


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
        patch("rlab.wandb_publisher.load_wandb_env"),
        patch.dict(sys.modules, {"wandb": SimpleNamespace(init=fake_init)}),
    ):
        _start_wandb(args, run_dir=tmp, config=config)

    assert captured["project"] == "Breakout-Atari2600-v0"
    assert captured["group"] == "bx0123456789abcdef"
    assert captured["id"] == "rlab-0123456789abcdef01234567"
    assert captured["name"] == args.run_name
    assert captured["config"]["wandb_project"] == "Breakout-Atari2600-v0"
    assert captured["config"]["game_family"] == "Atari2600-Breakout"
    assert captured["config"]["environment"]["env_id"] == "ale-py:breakout"
    assert "environment_hash" in captured["config"]
    assert "game_family:Atari2600-Breakout" in captured["tags"]
