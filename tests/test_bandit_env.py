from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from rlab.bandit_env import BanditVectorEnv
from rlab.env import EnvConfig, make_vec_envs
from rlab.env_registry import resolve_env_id, resolve_env_provider
from rlab.recipe_documents import load_recipe_document
from rlab.recipe_schema import validate_materialized_train_recipe


def _native_env(num_envs: int = 3) -> BanditVectorEnv:
    return BanditVectorEnv(
        "Bandit-v0",
        num_envs,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def _config() -> EnvConfig:
    return EnvConfig(
        env_provider="rlab",
        game="Bandit-v0",
        env_args={"autoreset_mode": "disabled"},
        task={
            "id": "identity",
            "action": {"set": "native"},
            "signals": {},
            "events": {},
            "termination": {
                "success": [],
                "failure": [],
                "timeout": [],
                "max_episode_steps": 1,
            },
            "reward": {"reward_mode": "native"},
        },
        state="",
        frame_skip=1,
        max_pool_frames=False,
        observation_size=0,
        obs_crop=(0, 0, 0, 0),
    )


def test_bandit_spaces_rewards_and_manual_reset() -> None:
    env = _native_env()
    assert env.single_observation_space == gym.spaces.Box(
        0.0, 0.0, shape=(1,), dtype=np.float32
    )
    assert env.single_action_space == gym.spaces.Discrete(2)
    assert env.metadata["autoreset_mode"] is gym.vector.AutoresetMode.DISABLED

    with pytest.raises(RuntimeError, match="require reset"):
        env.step(np.asarray([0, 1, 1], dtype=np.int64))

    observations, reset_infos = env.reset(seed=[1, 2, 3])
    assert observations.shape == (3, 1)
    assert observations.dtype == np.float32
    assert reset_infos == {}

    observations, rewards, terminated, truncated, infos = env.step(
        np.asarray([0, 1, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(observations, np.zeros((3, 1), dtype=np.float32))
    np.testing.assert_array_equal(rewards, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(terminated, [True, True, True])
    np.testing.assert_array_equal(truncated, [False, False, False])
    np.testing.assert_array_equal(infos["chosen_arm"], [0, 1, 1])
    np.testing.assert_array_equal(infos["optimal_arm"], [1, 1, 1])
    np.testing.assert_array_equal(infos["is_optimal"], [False, True, True])

    with pytest.raises(RuntimeError, match="require reset"):
        env.step(np.asarray([0, 1, 1], dtype=np.int64))


def test_bandit_masked_reset_preserves_unselected_lane_lifecycle() -> None:
    env = _native_env()
    env.reset()
    env.step(np.asarray([0, 1, 0], dtype=np.int64))

    selected = np.asarray([True, False, True], dtype=np.bool_)
    env.reset(seed=[4, None, 6], options={"reset_mask": selected})
    with pytest.raises(RuntimeError, match=r"\[1\]"):
        env.step(np.asarray([1, 1, 1], dtype=np.int64))

    env.reset(seed=[None, 5, None], options={"reset_mask": ~selected})
    _observations, rewards, terminated, truncated, _infos = env.step(
        np.asarray([1, 1, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(rewards, np.ones(3, dtype=np.float32))
    assert terminated.all()
    assert not truncated.any()


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"reset_mask": [True, False, True]}, TypeError),
        ({"reset_mask": np.ones(2, dtype=np.bool_)}, ValueError),
        ({"reset_mask": np.ones(3, dtype=np.int8)}, TypeError),
        ({"reset_mask": np.zeros(3, dtype=np.bool_)}, ValueError),
        ({"unknown": True}, ValueError),
    ],
)
def test_bandit_rejects_invalid_reset_options(options, error) -> None:
    env = _native_env()
    with pytest.raises(error):
        env.reset(options=options)


@pytest.mark.parametrize(
    ("actions", "error"),
    [
        (np.asarray([[0], [1], [0]], dtype=np.int64), ValueError),
        (np.asarray([0.0, 1.0, 0.0], dtype=np.float32), TypeError),
        (np.asarray([0, 2, 0], dtype=np.int64), ValueError),
    ],
)
def test_bandit_rejects_invalid_actions(actions, error) -> None:
    env = _native_env()
    env.reset()
    with pytest.raises(error):
        env.step(actions)


def test_rlab_provider_is_fixed_and_rejects_unknown_environment() -> None:
    provider = resolve_env_provider("rlab")
    assert provider.env_ids == ("Bandit-v0",)
    assert provider.supports_states is False
    assert provider.constructor_contract is not None
    assert provider.constructor_contract.required_values == {"autoreset_mode": "disabled"}
    assert resolve_env_id("rlab:Bandit-v0").provider_env_id == "Bandit-v0"
    assert "Bandit-v0" not in gym.registry
    with pytest.raises(ValueError, match="does not register environment"):
        resolve_env_id("rlab:Unknown-v0")

    with pytest.raises(ValueError, match="does not support state"):
        make_vec_envs(replace(_config(), state="Start"), n_envs=1, seed=1)


def test_rlab_facade_same_step_resets_bandit_lanes() -> None:
    env = make_vec_envs(_config(), n_envs=3, seed=7)
    try:
        observations = env.reset()
        np.testing.assert_array_equal(observations, np.zeros((3, 1), dtype=np.float32))

        next_observations, rewards, dones, infos = env.step(
            np.asarray([0, 1, 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(next_observations, np.zeros((3, 1), dtype=np.float32))
        np.testing.assert_array_equal(rewards, [0.0, 1.0, 1.0])
        assert dones.all()
        assert all("terminal_observation" in info for info in infos)

        _next_observations, rewards, dones, _infos = env.step(
            np.asarray([1, 1, 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(rewards, np.ones(3, dtype=np.float32))
        assert dones.all()
    finally:
        env.close()


def test_bandit_recipe_materializes_fixed_train_and_eval_contracts() -> None:
    document = load_recipe_document(
        Path("experiments/goals/rlab__bandit/recipes/base.yaml")
    )
    validate_materialized_train_recipe(document)

    train_config = document["train_config"]
    assert train_config["env_provider"] == "rlab"
    assert train_config["game"] == "Bandit-v0"
    assert train_config["n_envs"] == 8
    assert train_config["timesteps"] == 256
    assert train_config["checkpoint_eval_backend"] == "modal"
    assert train_config["post_train_eval_episodes"] == 256
    assert train_config["checkpoint_eval_n_envs"] == 32
    assert train_config["env_args"] == {"autoreset_mode": "disabled"}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bandit_ppo_converges_under_stochastic_evaluation(seed: int) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    train_env = make_vec_envs(_config(), n_envs=8, seed=seed)
    eval_env = None
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=8,
            batch_size=64,
            n_epochs=4,
            learning_rate=0.01,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            clip_range=0.2,
            vf_coef=0.5,
            normalize_advantage=True,
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=256)

        eval_env = make_vec_envs(_config(), n_envs=32, seed=10_000 + seed)
        observations = eval_env.reset()
        rewards: list[float] = []
        for _ in range(32):
            actions, _state = model.predict(observations, deterministic=False)
            observations, batch_rewards, dones, _infos = eval_env.step(actions)
            assert dones.all()
            rewards.extend(float(value) for value in batch_rewards)
        assert len(rewards) == 1_024
        assert np.mean(rewards) >= 0.95
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        torch.set_num_threads(previous_threads)
