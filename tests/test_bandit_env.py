from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import A2C, PPO

from rlab.bandit_env import BanditVectorEnv
from rlab.env import EnvConfig, make_vec_envs
from rlab.env_registry import resolve_env_id, resolve_env_provider
from rlab.metric_store import MetricStore
from rlab.policy_bundle import build_recipe_document, write_canonical_json
from rlab.recipe_documents import compose_train_document
from rlab.recipe_schema import validate_materialized_train_recipe
from rlab.sb3_models import load_sb3_model
from rlab.train import main as train_main


BANDIT_GOAL = Path("experiments/goals/rlab__bandit/_goal.yaml")
BANDIT_RECIPE = Path("experiments/goals/rlab__bandit/recipes/ppo.yaml")


def _bandit_recipe_document():
    return compose_train_document(BANDIT_GOAL, BANDIT_RECIPE)


def _write_versioned_recipe(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "recipe.json"
    return write_canonical_json(
        path,
        build_recipe_document(
            document,
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            run_description="ROM-free backend boundary smoke.",
            runtime_image_ref="docker:ghcr.io/tsilva/rlab-runtime@sha256:" + "b" * 64,
        ),
    )


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
    assert env.single_observation_space == gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)
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

        next_observations, rewards, dones, infos = env.step(np.asarray([0, 1, 1], dtype=np.int64))
        np.testing.assert_array_equal(next_observations, np.zeros((3, 1), dtype=np.float32))
        np.testing.assert_array_equal(rewards, [0.0, 1.0, 1.0])
        assert dones.all()
        assert all("terminal_observation" in info for info in infos)

        _next_observations, rewards, dones, _infos = env.step(np.asarray([1, 1, 1], dtype=np.int64))
        np.testing.assert_array_equal(rewards, np.ones(3, dtype=np.float32))
        assert dones.all()
    finally:
        env.close()


def test_bandit_recipe_materializes_fixed_train_and_eval_contracts() -> None:
    document = _bandit_recipe_document()
    validate_materialized_train_recipe(document)

    train_config = document["train_config"]
    assert train_config["env_provider"] == "rlab"
    assert train_config["game"] == "Bandit-v0"
    assert train_config["n_envs"] == 8
    assert train_config["timesteps"] == 256
    assert train_config["checkpoint_eval_backend"] == "modal"
    assert train_config["post_train_eval_episodes"] == 256
    assert train_config["checkpoint_eval_n_envs"] == 32
    assert train_config["stop_on_acceptance"] is True
    assert train_config["checkpoint_eval_acceptance"] == [
        {
            "metric": "eval/full/episode/return/mean",
            "operator": ">=",
            "threshold": 0.9,
        }
    ]
    assert train_config["env_args"] == {"autoreset_mode": "disabled"}
    assert train_config["training_backend"]["id"] == "sb3.ppo"


def test_bandit_runs_through_sb3_backend_and_records_backend_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    document = _bandit_recipe_document()
    recipe_path = _write_versioned_recipe(tmp_path, document)
    config = dict(document["train_config"])
    config.update(
        {
            "run_name": "backend-smoke",
            "run_description": "ROM-free backend boundary smoke.",
            "runs_dir": str(tmp_path),
            "timesteps": 64,
            "checkpoint_freq": 0,
            "checkpoint_eval_backend": "none",
            "checkpoint_eval_stages": [],
            "early_stop": None,
            "post_train_eval_episodes": 0,
            "wandb": False,
            "wandb_mode": "disabled",
            "recipe_json_path": str(recipe_path),
        }
    )
    backend = dict(config["training_backend"])
    backend_config = dict(backend["config"])
    backend_config.update({"device": "cpu", "n_epochs": 1})
    backend["config"] = backend_config
    config["training_backend"] = backend
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv("RLAB_INTERNAL_LEARNER", "1")
    assert train_main(["--train-config-json", str(path)]) == 0

    run_dir = tmp_path / "backend-smoke"
    assert (run_dir / "learner_ready.json").is_file()
    assert (run_dir / "final_model.zip").is_file()
    metadata = json.loads((run_dir / "final_model.metadata.json").read_text())
    assert metadata["metadata_version"] == 7
    assert metadata["training_backend_id"] == "sb3.ppo"
    assert len(metadata["training_backend_config_hash"]) == 64


def test_bandit_runs_through_a2c_backend_and_round_trips_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    document = _bandit_recipe_document()
    recipe_path = _write_versioned_recipe(tmp_path, document)
    config = dict(document["train_config"])
    config.update(
        {
            "run_name": "a2c-backend-smoke",
            "run_description": "ROM-free A2C backend boundary smoke.",
            "runs_dir": str(tmp_path),
            "timesteps": 64,
            "checkpoint_freq": 0,
            "checkpoint_eval_backend": "none",
            "checkpoint_eval_stages": [],
            "early_stop": None,
            "post_train_eval_episodes": 0,
            "wandb": False,
            "wandb_mode": "disabled",
            "recipe_json_path": str(recipe_path),
            "training_backend": {
                "id": "sb3.a2c",
                "config": {
                    "device": "cpu",
                    "n_steps": 8,
                    "learning_rate": 0.01,
                    "gamma": 1.0,
                },
            },
        }
    )
    path = tmp_path / "a2c-train.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv("RLAB_INTERNAL_LEARNER", "1")
    assert train_main(["--train-config-json", str(path)]) == 0

    run_dir = tmp_path / "a2c-backend-smoke"
    model_path = run_dir / "final_model.zip"
    metadata = json.loads((run_dir / "final_model.metadata.json").read_text())
    assert metadata["training_backend_id"] == "sb3.a2c"
    assert metadata["algorithm_id"] == "a2c"
    assert metadata["model_class"] == "stable_baselines3.a2c.a2c.A2C"
    metric_store = MetricStore(run_dir / "rlab.sqlite")
    assert metric_store.latest_metric("train/algorithm/a2c/update/value_loss") is not None
    assert metric_store.latest_metric("train/algorithm/ppo/update/value_loss") is None
    from rlab.trusted_inputs import approve_internal_model

    with approve_internal_model(model_path, execution_id="test-bandit") as approved:
        assert isinstance(load_sb3_model(approved, device="cpu"), A2C)


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
