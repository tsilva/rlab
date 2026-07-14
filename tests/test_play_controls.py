from __future__ import annotations

from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest

from rlab.env import EnvConfig
from rlab.play import (
    optional_fast_env_frames,
    playback_model_observation,
    playback_runtime_config,
    playback_step_indices,
    vector_env_frame,
)
from rlab.play_debug import DebugCommandError, parse_debug_command


def test_debug_commands_use_one_non_overlapping_step_grammar() -> None:
    assert parse_debug_command("").count == 1
    assert parse_debug_command("step 4").count == 4
    assert parse_debug_command("continue", ("life_loss",)).target is None
    assert parse_debug_command("continue life_loss", ("life_loss",)).target == "life_loss"
    assert parse_debug_command("show policy").target == "policy"
    assert parse_debug_command("reset 123").seed == 123


@pytest.mark.parametrize("line", ["step 0", "step 101", "run 2", "show wat"])
def test_invalid_debug_commands_never_parse_as_steps(line: str) -> None:
    with pytest.raises(DebugCommandError):
        parse_debug_command(line, ("life_loss",))


def test_playback_runtime_config_preserves_selected_boundary_policy() -> None:
    original = EnvConfig(
        game="SuperMarioBros-Nes-v0",
        task={"termination": {"failure": [], "success": []}},
    )

    configured = playback_runtime_config(original)

    assert configured is original
    assert configured.task["termination"]["success"] == []
    assert original.task["termination"]["success"] == []


def test_generic_vector_observation_reaches_policy_unchanged() -> None:
    observation = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    result = playback_model_observation(
        type("Model", (), {"observation_space": object()})(),
        observation,
        EnvConfig(game="CartPole-v1"),
        active_task_state=None,
        active_info_value=None,
    )

    assert result is observation
    assert optional_fast_env_frames(result) is None


def test_zero_max_episode_steps_keeps_playback_running() -> None:
    assert list(islice(playback_step_indices(0), 4)) == [0, 1, 2, 3]


def test_positive_max_episode_steps_caps_playback() -> None:
    assert list(playback_step_indices(3)) == [0, 1, 2]


def test_vector_env_frame_returns_owned_lane_frame() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    env = SimpleNamespace(get_images=lambda: [source])

    frame = vector_env_frame(env)
    source.fill(0)

    assert frame.shape == (2, 2, 3)
    assert frame.any()
