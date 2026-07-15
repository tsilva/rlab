from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from rlab.jerk import JerkPolicy, JerkSearch
from rlab.policy_models import load_policy_model, resolve_policy_algorithm
from rlab.task_kernels import Outcome


ACTIONS = ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left")


def test_jerk_search_retains_successful_full_sequence() -> None:
    search = JerkSearch(
        n_envs=1,
        seed=7,
        total_timesteps=10,
        action_names=ACTIONS,
        forward_action="right_b",
        jump_action="right_a_b",
        backtrack_action="left",
        fallback_action="noop",
        forward_steps=100,
        backtrack_steps=70,
        jump_probability=0.0,
        jump_repeat=4,
        exploit_bias=0.25,
        retained_limit=8,
    )
    first = search.next_actions()
    search.observe([1.0], [False])
    second = search.next_actions()
    record = SimpleNamespace(
        outcome=Outcome.SUCCESS,
        metrics={"level_complete": True, "max_x_pos": 3161},
    )
    search.observe([2.0], [True], {0: record})

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.completed is True
    assert candidate.progress == 3161
    assert candidate.actions == (int(first[0]), int(second[0]))
    assert candidate.mean_return == 3.0


def test_jerk_policy_round_trip_and_lane_resets(tmp_path) -> None:
    path = tmp_path / "model.zip"
    policy = JerkPolicy(
        action_names=ACTIONS,
        action_sequence=(2, 4),
        fallback_action=0,
    )
    policy.save(path)
    loaded = JerkPolicy.load(path)
    loaded.bind_action_space(gym.spaces.Discrete(len(ACTIONS)))

    obs = np.zeros((2, 1), dtype=np.float32)
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 2]
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [4, 4]
    loaded.reset_lanes([True, False])
    assert loaded.predict(obs, deterministic=False)[0].tolist() == [2, 0]


def test_generic_policy_loader_dispatches_jerk(tmp_path) -> None:
    path = tmp_path / "model.zip"
    JerkPolicy(action_names=ACTIONS, action_sequence=(2,), fallback_action=0).save(path)
    metadata = {
        "training_backend_id": "rlab.jerk",
        "algorithm_id": "jerk",
        "model_class": "rlab.jerk.JerkPolicy",
    }

    assert resolve_policy_algorithm(metadata) == "jerk"
    loaded = load_policy_model(path, device="cpu", metadata=metadata)
    assert isinstance(loaded, JerkPolicy)
