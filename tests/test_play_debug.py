from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from rlab.play_debug import (
    format_policy_detail,
    format_raw,
    format_model_input,
    inspect_policy,
    sample_policy_decision,
)


def make_model(action_space: gym.Space, **policy_kwargs):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    policy = ActorCriticPolicy(
        observation_space,
        action_space,
        lambda _progress: 1e-3,
        net_arch=[8],
        **policy_kwargs,
    )
    return SimpleNamespace(policy=policy), np.zeros((1, 4), dtype=np.float32)


@pytest.mark.parametrize(
    "action_space",
    [
        gym.spaces.Discrete(4),
        gym.spaces.MultiDiscrete([2, 3]),
        gym.spaces.MultiBinary(3),
        gym.spaces.Box(-0.1, 0.1, shape=(2,), dtype=np.float32),
    ],
)
def test_sampled_decision_executes_the_same_action_as_sb3_predict(action_space) -> None:
    model, observation = make_model(action_space)
    torch.manual_seed(1234)
    rng_state = torch.random.get_rng_state()

    decision = sample_policy_decision(model, observation)
    torch.random.set_rng_state(rng_state)
    expected, _state = model.policy.predict(observation, deterministic=False)

    np.testing.assert_allclose(decision.executed_action, expected[0])
    assert np.isfinite(decision.value)
    assert np.isfinite(decision.log_probability)


def test_squashed_box_decision_matches_sb3_predict() -> None:
    action_space = gym.spaces.Box(-2.0, 3.0, shape=(2,), dtype=np.float32)
    model, observation = make_model(action_space, use_sde=True, squash_output=True)
    torch.manual_seed(55)
    rng_state = torch.random.get_rng_state()

    decision = sample_policy_decision(model, observation)
    torch.random.set_rng_state(rng_state)
    expected, _state = model.policy.predict(observation, deterministic=False)

    np.testing.assert_allclose(decision.executed_action, expected[0])
    assert action_space.contains(decision.executed_action.astype(np.float32))
    assert decision.entropy is None


def test_policy_inspection_does_not_sample_or_change_rng() -> None:
    model, observation = make_model(gym.spaces.Discrete(3))
    before = torch.random.get_rng_state().clone()

    decision = inspect_policy(model, observation)

    assert not decision.sampled
    assert torch.equal(before, torch.random.get_rng_state())
    assert decision.selected_discrete_action == int(decision.mode)


def test_large_model_input_is_summarized_with_content_hash() -> None:
    text = format_model_input(np.arange(128, dtype=np.float32).reshape(1, 128))

    assert "shape=(1, 128)" in text
    assert "sha256=" in text
    assert "values=" not in text


def test_policy_detail_uses_a_color_optional_terminal_card(monkeypatch) -> None:
    monkeypatch.setenv("RLAB_NO_COLOR", "1")
    model, observation = make_model(gym.spaces.Discrete(3))

    text = format_policy_detail(
        sample_policy_decision(model, observation),
        ("noop", "right_a", "left"),
    )

    assert "╭─ POLICY INSPECTOR" in text
    assert "🎲  POLICY DISTRIBUTION" in text
    assert "█" in text
    assert "\033[" not in text


def test_policy_inspector_labels_unsampled_action_as_the_mode(monkeypatch) -> None:
    monkeypatch.setenv("RLAB_NO_COLOR", "1")
    model, observation = make_model(gym.spaces.Discrete(3))

    text = format_policy_detail(inspect_policy(model, observation))

    assert "policy mode" in text
    assert "sampled action" not in text


def test_raw_output_groups_fields_instead_of_dumping_a_python_dict(monkeypatch) -> None:
    monkeypatch.setenv("RLAB_NO_COLOR", "1")

    text = format_raw(
        {
            "runtime": {"provider_reward": 0.0, "events": ("life_loss",)},
            "terminal_observation": ["observation: shape=(4, 84, 84) dtype=uint8"],
        }
    )

    assert "⚙  RUNTIME" in text
    assert "provider reward" in text
    assert "▧  TERMINAL OBSERVATION" in text
    assert "{'runtime':" not in text
