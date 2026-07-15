from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from rlab.batch_runtime import EpisodeRecord
from rlab.jerk import JerkPolicy, JerkSearch, RetainedSequence
from rlab.policy_models import load_policy_model, resolve_policy_algorithm
from rlab.task_kernels import Outcome
from rlab.training import jerk as jerk_training


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
        max_exploit_probability=0.9,
        mutation_window_steps=0,
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


def test_jerk_replays_retained_prefix_then_extends_it() -> None:
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
        exploit_bias=1.0,
        max_exploit_probability=1.0,
        mutation_window_steps=0,
        retained_limit=8,
    )
    prefix = (ACTIONS.index("right_b"),)
    search._retained[prefix] = RetainedSequence(
        actions=prefix,
        returns=[1.0],
        progress=100.0,
    )
    search._start_lane(0)

    assert search.next_actions().tolist() == list(prefix)
    search.observe([1.0], [False])
    assert search.next_actions().tolist() == [ACTIONS.index("right_b")]
    record = SimpleNamespace(outcome=Outcome.NEUTRAL, metrics={"max_x_pos": 200})
    search.observe([1.0], [True], {0: record})

    candidate = search.best_candidate()
    assert candidate is not None
    assert candidate.actions == (*prefix, ACTIONS.index("right_b"))
    assert candidate.progress == 200


def test_jerk_rewinds_retained_tail_before_exploring() -> None:
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
        exploit_bias=1.0,
        max_exploit_probability=1.0,
        mutation_window_steps=3,
        retained_limit=8,
    )
    prefix = tuple(range(4))
    search._retained[prefix] = RetainedSequence(
        actions=prefix,
        returns=[1.0],
        progress=100.0,
    )
    search._start_lane(0)

    replay_limit = search._lanes[0].replay_limit
    assert 1 <= replay_limit <= len(prefix)
    assert [int(search.next_actions()[0]) for _ in range(replay_limit)] == list(
        prefix[:replay_limit]
    )
    assert int(search.next_actions()[0]) == ACTIONS.index("right_b")


def test_jerk_preserves_root_exploration_floor() -> None:
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
        max_exploit_probability=0.9,
        mutation_window_steps=0,
        retained_limit=8,
    )
    search.global_step = search.total_timesteps

    assert search.exploit_probability == 0.9


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


class _FakeJerkEnv:
    def __init__(self, *, success: bool) -> None:
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.success = success
        self.steps = 0
        self.closed = False

    def reset(self):
        return np.zeros((1, 1), dtype=np.float32)

    def step(self, actions):
        del actions
        self.steps += 1
        done = self.success and self.steps == 1
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([done]),
            [{}],
        )

    def drain_records(self):
        if not self.success or self.steps != 1:
            return []
        return [
            EpisodeRecord(
                lane=0,
                episode_index=0,
                start_id="Level1-1",
                episode_return=1.0,
                episode_length=1,
                terminated=True,
                truncated=False,
                outcome=Outcome.SUCCESS,
                events=("level_change",),
                metrics={"level_complete": True},
            )
        ]

    def close(self):
        self.closed = True


class _FakeMetricStore:
    def __init__(self) -> None:
        self.payloads = []
        self.checkpoints = []

    def append_metrics(self, payload, **kwargs):
        self.payloads.append((dict(payload), kwargs))

    def record_checkpoint(self, **kwargs):
        self.checkpoints.append(dict(kwargs))
        return len(self.checkpoints)


def _jerk_context(tmp_path, *, timesteps: int):
    args = SimpleNamespace(
        resolved_n_envs=1,
        seed=7,
        timesteps=timesteps,
        acceptance_mode="first_training_success",
        forward_action="right_b",
        jump_action="right_a_b",
        backtrack_action="left",
        fallback_action="noop",
        forward_steps=100,
        backtrack_steps=70,
        jump_probability=0.0,
        jump_repeat=4,
        exploit_bias=0.25,
        max_exploit_probability=0.9,
        mutation_window_steps=0,
        retained_limit=8,
        log_interval_steps=10,
        checkpoint_freq=100,
        early_stop=None,
        checkpoint_eval_backend="none",
        run_name="test-jerk",
    )
    return SimpleNamespace(
        args=args,
        environment=SimpleNamespace(game="SuperMarioBros-Nes-v0", state="Level1-1", states=()),
        checkpoint_dir=tmp_path / "checkpoints",
        run_dir=tmp_path,
        metric_store=_FakeMetricStore(),
        wandb_enabled=False,
        stop_flag=SimpleNamespace(requested=False),
        mark_ready=lambda: None,
    )


def test_first_training_success_saves_playable_checkpoint_and_stops(tmp_path) -> None:
    env = _FakeJerkEnv(success=True)
    context = _jerk_context(tmp_path, timesteps=10)

    with (
        mock.patch.object(jerk_training, "make_training_vec_env", return_value=env),
        mock.patch.object(jerk_training, "task_action_set", return_value="simple"),
        mock.patch.object(
            jerk_training,
            "target_for_game",
            return_value=SimpleNamespace(action_names_for_set=lambda _action_set: ACTIONS),
        ),
        mock.patch.object(
            jerk_training,
            "write_model_metadata",
            side_effect=lambda path, *_args, **_kwargs: path.with_suffix(".metadata.json"),
        ),
    ):
        jerk_training.run_jerk(context)

    assert env.steps == 1
    assert env.closed is True
    assert [
        (checkpoint["kind"], checkpoint["step"]) for checkpoint in context.metric_store.checkpoints
    ] == [("checkpoint", 1), ("final", 1)]
    checkpoint_path = context.metric_store.checkpoints[0]["path"]
    assert isinstance(JerkPolicy.load(checkpoint_path), JerkPolicy)
    final_metrics = context.metric_store.payloads[-1][0]
    assert final_metrics["train/outcome/success/from/Level1-1/count"] == 1


def test_first_training_success_budget_exhaustion_is_unsuccessful(tmp_path) -> None:
    env = _FakeJerkEnv(success=False)
    context = _jerk_context(tmp_path, timesteps=2)

    with (
        mock.patch.object(jerk_training, "make_training_vec_env", return_value=env),
        mock.patch.object(jerk_training, "task_action_set", return_value="simple"),
        mock.patch.object(
            jerk_training,
            "target_for_game",
            return_value=SimpleNamespace(action_names_for_set=lambda _action_set: ACTIONS),
        ),
        mock.patch.object(
            jerk_training,
            "write_model_metadata",
            side_effect=lambda path, *_args, **_kwargs: path.with_suffix(".metadata.json"),
        ),
        pytest.raises(RuntimeError, match="exhausted 2 transitions"),
    ):
        jerk_training.run_jerk(context)

    assert env.steps == 2
    assert env.closed is True
    assert [
        (checkpoint["kind"], checkpoint["step"]) for checkpoint in context.metric_store.checkpoints
    ] == [("final", 2)]
