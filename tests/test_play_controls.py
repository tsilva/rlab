from __future__ import annotations

from itertools import islice
from types import SimpleNamespace

import numpy as np

from rlab.env import EnvConfig
from rlab.play import (
    StepOverControls,
    playback_runtime_config,
    playback_step_indices,
    vector_env_frame,
)


def test_step_over_controls_advance_once_on_space_press() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=32), keydown_type=1, keyup_type=2, step_key=32
    )
    controls.handle_event(
        SimpleNamespace(type=2, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert controls.consume_step()
    assert not controls.consume_step()


def test_step_over_controls_keep_advancing_while_space_is_held() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert controls.consume_step()
    assert controls.consume_step()

    controls.handle_event(
        SimpleNamespace(type=2, key=32), keydown_type=1, keyup_type=2, step_key=32
    )

    assert not controls.consume_step()


def test_step_over_controls_ignore_other_keys() -> None:
    controls = StepOverControls()

    controls.handle_event(
        SimpleNamespace(type=1, key=13), keydown_type=1, keyup_type=2, step_key=32
    )

    assert not controls.consume_step()


def test_playback_runtime_config_requests_clean_mario_completion_records() -> None:
    original = EnvConfig(
        game="SuperMarioBros-Nes-v0",
        task={"termination": {"failure": [], "success": []}},
    )

    configured = playback_runtime_config(original)

    assert configured.task["termination"]["success"] == ["level_change"]
    assert original.task["termination"]["success"] == []


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
