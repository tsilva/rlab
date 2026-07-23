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


def _search(
    *,
    seed: int = 7,
    n_envs: int = 1,
    initial_probability: float = 0.25,
    max_probability: float = 0.9,
    protected_prefix_steps: int = 128,
    max_prefix_shorten_steps: int = 128,
    retained_limit: int = 8,
) -> JerkSearch:
    return JerkSearch(
        n_envs=n_envs,
        seed=seed,
        total_timesteps=10,
        action_names=ACTIONS,
        fallback_action="noop",
        archive_replay_probability_initial=initial_probability,
        archive_replay_probability_max=max_probability,
        protected_prefix_steps=protected_prefix_steps,
        max_prefix_shorten_steps=max_prefix_shorten_steps,
        retained_limit=retained_limit,
    )


def _candidate(
    actions: tuple[int, ...],
    mean_return: float,
    *,
    completed: bool = False,
    progress: float = 0.0,
) -> RetainedSequence:
    return RetainedSequence(
        actions=actions,
        return_sum=mean_return,
        return_count=1,
        completed=completed,
        progress=progress,
    )


def test_jerk_search_retains_successful_full_sequence() -> None:
    search = _search()
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


def test_jerk_uniform_sampling_is_seeded_and_covers_action_set() -> None:
    first = _search(initial_probability=0.0, max_probability=0.0)
    second = _search(initial_probability=0.0, max_probability=0.0)

    first_actions = [int(first.next_actions()[0]) for _ in range(256)]
    second_actions = [int(second.next_actions()[0]) for _ in range(256)]

    assert first_actions == second_actions
    assert set(first_actions) == set(range(len(ACTIONS)))


def test_jerk_replays_short_retained_prefix_then_samples_uniform_suffix() -> None:
    search = _search(initial_probability=1.0, max_probability=1.0)
    prefix = (1, 2, 3)
    search._retained[prefix] = _candidate(prefix, 1.0, progress=100.0)
    search._start_lane(0)

    replayed = [int(search.next_actions()[0]) for _ in prefix]
    sampled_suffix = int(search.next_actions()[0])

    assert replayed == list(prefix)
    assert sampled_suffix in range(len(ACTIONS))
    assert search._lanes[0].mode == "explore"


@pytest.mark.parametrize("length", [1, 128])
def test_jerk_paths_at_or_below_protected_length_replay_fully(length: int) -> None:
    search = _search(initial_probability=1.0, max_probability=1.0)
    path = tuple(index % len(ACTIONS) for index in range(length))
    search._retained[path] = _candidate(path, 1.0)

    search._start_lane(0)

    assert search._lanes[0].replay_limit == length
    assert [int(search.next_actions()[0]) for _ in range(length)] == list(path)


def test_jerk_long_retained_path_always_shortens_within_protected_bounds() -> None:
    search = _search(
        initial_probability=1.0,
        max_probability=1.0,
        protected_prefix_steps=128,
        max_prefix_shorten_steps=128,
    )
    path = tuple(index % len(ACTIONS) for index in range(300))
    search._retained[path] = _candidate(path, 1.0, progress=100.0)
    search._start_lane(0)

    replay_limit = search._lanes[0].replay_limit
    assert 172 <= replay_limit <= 299
    assert replay_limit >= search.protected_prefix_steps
    assert len(path) - replay_limit <= search.max_prefix_shorten_steps
    assert [int(search.next_actions()[0]) for _ in range(replay_limit)] == list(path[:replay_limit])
    assert int(search.next_actions()[0]) in range(len(ACTIONS))
    assert search.archive_selected_prefix_return_mean == 1.0


def test_jerk_shortened_attempt_is_retained_by_its_executed_path_identity() -> None:
    search = _search(
        initial_probability=1.0,
        max_probability=1.0,
        protected_prefix_steps=1,
        max_prefix_shorten_steps=1,
    )
    source_path = (1, 2, 3)
    search._retained[source_path] = _candidate(source_path, 10.0)
    search._start_lane(0)
    replay_limit = search._lanes[0].replay_limit
    assert replay_limit == 2

    for step in range(replay_limit):
        search.next_actions()
        done = step == replay_limit - 1
        record = SimpleNamespace(outcome=Outcome.NEUTRAL, metrics={}) if done else None
        search.observe([1.0], [done], {0: record} if record is not None else None)

    executed_path = source_path[:replay_limit]
    assert set(search._retained) == {source_path, executed_path}
    assert search._retained[source_path].return_count == 1
    assert search._retained[executed_path].mean_return == 2.0


def test_jerk_archive_distribution_is_sorted_shifted_and_strictly_positive() -> None:
    search = _search()
    search._retained[(2,)] = _candidate((2,), 5.0)
    search._retained[(0,)] = _candidate((0,), -5.0)
    search._retained[(1,)] = _candidate((1,), 0.0)

    candidates, probabilities = search._retained_distribution()

    assert [candidate.actions for candidate in candidates] == [(0,), (1,), (2,)]
    assert np.all(probabilities > 0.0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(1.0 / 3.0)
    assert probabilities[2] == pytest.approx(2.0 / 3.0)


def test_jerk_equal_archive_returns_sample_uniformly() -> None:
    search = _search()
    for action in range(3):
        search._retained[(action,)] = _candidate((action,), -2.0)

    _candidates, probabilities = search._retained_distribution()

    assert probabilities.tolist() == pytest.approx([1.0 / 3.0] * 3)


def test_jerk_duplicate_path_updates_running_stats_without_growing_archive() -> None:
    search = _search(retained_limit=2)
    path = (1, 2)

    search._upsert_retained(path, score_return=2.0, completed=False, progress=10.0)
    search._upsert_retained(path, score_return=4.0, completed=True, progress=20.0)

    assert search.retained_count == 1
    candidate = search._retained[path]
    assert candidate.return_sum == 6.0
    assert candidate.return_count == 2
    assert candidate.mean_return == 3.0
    assert candidate.completed is True
    assert candidate.progress == 20.0


def test_jerk_only_unique_insertions_trigger_worst_path_eviction() -> None:
    search = _search(retained_limit=2)
    search._upsert_retained((0,), score_return=1.0, completed=False, progress=10.0)
    search._upsert_retained((1,), score_return=2.0, completed=False, progress=20.0)
    search._upsert_retained((1,), score_return=4.0, completed=False, progress=20.0)
    assert search.retained_count == 2

    search._upsert_retained((2,), score_return=3.0, completed=False, progress=30.0)

    assert search.retained_count == 2
    assert set(search._retained) == {(1,), (2,)}


def test_jerk_preserves_root_exploration_floor() -> None:
    search = _search()
    search.global_step = search.total_timesteps

    assert search.archive_replay_probability == 0.9


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
    from rlab.trusted_inputs import approve_internal_model

    with approve_internal_model(path, execution_id="test-jerk") as approved:
        loaded = load_policy_model(approved, device="cpu", metadata=metadata)
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
        fallback_action="noop",
        archive_replay_probability_initial=0.25,
        archive_replay_probability_max=0.9,
        protected_prefix_steps=128,
        max_prefix_shorten_steps=128,
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


def _install_test_bundle(model_path, *, save_checkpoint, **_kwargs):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text("{}\n", encoding="utf-8")
    return model_path, metadata_path


def test_first_training_success_saves_playable_checkpoint_and_stops(tmp_path) -> None:
    env = _FakeJerkEnv(success=True)
    context = _jerk_context(tmp_path, timesteps=10)

    with (
        mock.patch.object(jerk_training, "make_training_vec_env", return_value=env),
        mock.patch.object(
            jerk_training,
            "configured_action_meanings",
            return_value=ACTIONS,
        ),
        mock.patch.object(
            jerk_training,
            "install_model_bundle",
            side_effect=_install_test_bundle,
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
        mock.patch.object(
            jerk_training,
            "configured_action_meanings",
            return_value=ACTIONS,
        ),
        mock.patch.object(
            jerk_training,
            "install_model_bundle",
            side_effect=_install_test_bundle,
        ),
        pytest.raises(RuntimeError, match="exhausted 2 transitions"),
    ):
        jerk_training.run_jerk(context)

    assert env.steps == 2
    assert env.closed is True
    assert [
        (checkpoint["kind"], checkpoint["step"]) for checkpoint in context.metric_store.checkpoints
    ] == [("final", 2)]
