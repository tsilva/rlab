from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import fields
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from rlab.batch_runtime import BatchRuntime, ProviderDescriptor
from rlab.runtime_contract import train_config_contract_payload, train_config_contract_sha256
from rlab.task_kernels import IdentityTaskDefinition
from rlab.train import main as train_main
from rlab.train_config import materialized_train_args, validate_and_normalize_train_config
from rlab.training_backend import BackendContext, BackendUnavailableError


def backend_config(backend_id: str = "sb3.ppo", **config) -> dict[str, object]:
    return {"training_backend": {"id": backend_id, "config": config}}


def test_core_backend_contract_does_not_import_sb3() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import rlab.train, rlab.batch_runtime, "
                "rlab.training_backend; from rlab.runtime_contract import "
                "train_config_contract_sha256; train_config_contract_sha256(); "
                "assert 'stable_baselines3' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_learner_backend_context_is_wandb_blind() -> None:
    names = {field.name for field in fields(BackendContext)}
    assert "wandb_run" not in names
    assert "external_wandb_publisher" not in names
    assert "wandb_enabled" in names


def test_direct_learner_entrypoint_is_guarded() -> None:
    with (
        mock.patch.dict("os.environ", {}, clear=True),
        pytest.raises(RuntimeError, match="use `rlab train`"),
    ):
        train_main([])


def test_sb3_backend_schema_is_strict_and_materializes_backend_defaults() -> None:
    normalized = validate_and_normalize_train_config(
        {"timesteps": 10, **backend_config(n_steps=4)},
        required_keys=("training_backend",),
    )
    config = normalized["training_backend"]["config"]
    assert config["n_steps"] == 4
    assert config["batch_size"] == 256
    assert "learning_rate" not in normalized

    with pytest.raises(ValueError, match="unexpected fields.*invented"):
        validate_and_normalize_train_config(backend_config(invented=True))
    with pytest.raises(ValueError, match="unknown training backend"):
        validate_and_normalize_train_config(backend_config("unknown.algo"))


def test_sb3_a2c_schema_is_strict_and_rejects_ppo_only_fields() -> None:
    normalized = validate_and_normalize_train_config(
        {"timesteps": 10, **backend_config("sb3.a2c", n_steps=8)},
        required_keys=("training_backend",),
    )
    config = normalized["training_backend"]["config"]
    assert config["n_steps"] == 8
    assert config["learning_rate"] == 7e-4
    assert config["use_rms_prop"] is True

    with pytest.raises(ValueError, match="unexpected fields.*clip_range"):
        validate_and_normalize_train_config(backend_config("sb3.a2c", clip_range=0.2))


def test_jerk_backend_schema_is_strict_and_available() -> None:
    normalized = validate_and_normalize_train_config(
        {"timesteps": 100, **backend_config("rlab.jerk", jump_probability=0.2)},
        required_keys=("training_backend",),
    )
    config = normalized["training_backend"]["config"]
    assert config["jump_probability"] == 0.2
    assert config["forward_action"] == "right_b"
    assert config["acceptance_mode"] == "checkpoint_eval"

    with pytest.raises(ValueError, match="jump_probability must be in"):
        validate_and_normalize_train_config(backend_config("rlab.jerk", jump_probability=1.1))
    with pytest.raises(ValueError, match="requires checkpoint_eval_backend=none"):
        validate_and_normalize_train_config(
            backend_config("rlab.jerk", acceptance_mode="first_training_success")
        )

    accepted = validate_and_normalize_train_config(
        {
            "checkpoint_eval_backend": "none",
            "early_stop": None,
            "checkpoint_eval_stages": [],
            **backend_config("rlab.jerk", acceptance_mode="first_training_success"),
        }
    )
    assert accepted["training_backend"]["config"]["acceptance_mode"] == ("first_training_success")


@pytest.mark.parametrize("backend_id", ["rlab.ppo", "rlab.a2c"])
def test_planned_backends_and_optional_components_fail_preflight(
    backend_id: str,
) -> None:
    payload = backend_config(
        backend_id,
        model={"id": "model2", "config": {}},
        value_normalization={"id": "popart", "config": {}},
        intrinsic_rewards=[
            {"id": "rnd", "config": {}},
            {"id": "phash", "config": {}},
        ],
    )
    with pytest.raises(BackendUnavailableError, match="future milestone"):
        validate_and_normalize_train_config(payload)


def test_unavailable_backend_fails_before_run_resources_are_created() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_path = root / "train.json"
        config_path.write_text(
            json.dumps(
                {
                    "game": "Bandit-v0",
                    "env_provider": "rlab",
                    "runs_dir": str(root / "runs"),
                    "run_name": "must-not-exist",
                    "timesteps": 1,
                    **backend_config("rlab.ppo"),
                }
            ),
            encoding="utf-8",
        )
        with (
            mock.patch.dict("os.environ", {"RLAB_INTERNAL_LEARNER": "1"}),
            pytest.raises(BackendUnavailableError),
        ):
            train_main(["--train-config-json", str(config_path)])
        assert not (root / "runs" / "must-not-exist").exists()


def test_materialized_args_are_a_temporary_flat_backend_view() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.json"
        path.write_text(
            json.dumps({"timesteps": 20, **backend_config(n_steps=7)}),
            encoding="utf-8",
        )
        args = materialized_train_args(path)
    assert args.timesteps == 20
    assert args.n_steps == 7
    assert args.training_backend_id == "sb3.ppo"
    assert args.training_backend["config"]["n_steps"] == 7


def test_backend_schemas_participate_in_runtime_contract_hash(monkeypatch) -> None:
    before = train_config_contract_sha256()
    from rlab.training import sb3_ppo

    monkeypatch.setitem(sb3_ppo.DEFAULT_CONFIG, "test_contract_field", 1)
    after = train_config_contract_sha256()
    assert before != after
    assert "training_backends" in train_config_contract_payload()


def test_on_policy_backends_can_reuse_one_backend_owned_collector() -> None:
    shared_collector: list[object] = []

    class TestOnlyOnPolicyBackend:
        def __init__(self, collector) -> None:
            self.collector = collector

    ppo = TestOnlyOnPolicyBackend(shared_collector)
    a2c = TestOnlyOnPolicyBackend(shared_collector)
    assert ppo.collector is a2c.collector


class TextVectorProvider:
    num_envs = 2
    single_observation_space = gym.spaces.Dict(
        {
            "prompt": gym.spaces.Text(max_length=64),
            "state": gym.spaces.Box(0, 10, shape=(2,), dtype=np.int32),
        }
    )
    single_action_space = gym.spaces.Dict({"answer": gym.spaces.Text(max_length=16)})

    def __init__(self) -> None:
        self.last_actions = None
        self.reset_seeds: list[list[int | None]] = []

    def reset(self, *, seed=None, options=None):
        del options
        self.reset_seeds.append(list(seed))
        return {
            "prompt": np.asarray(["choose", "choose"], dtype=object),
            "state": np.zeros((2, 2), dtype=np.int32),
        }, {}

    def step(self, actions):
        self.last_actions = actions
        return (
            {
                "prompt": np.asarray(["done-a", "done-b"], dtype=object),
                "state": np.ones((2, 2), dtype=np.int32),
            },
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([True, True]),
            np.asarray([False, False]),
            {"tokens": np.asarray([3, 5], dtype=np.int64)},
        )

    def close(self) -> None:
        pass


def text_runtime(*, global_lane_ids=(0, 1)) -> tuple[TextVectorProvider, BatchRuntime]:
    provider = TextVectorProvider()
    descriptor = ProviderDescriptor(
        provider_id="text",
        native_observation_space=provider.single_observation_space,
        native_action_space=provider.single_action_space,
    )
    kernel = IdentityTaskDefinition(max_episode_steps=0).bind(descriptor, provider.num_envs)
    return provider, BatchRuntime(
        provider,
        descriptor,
        kernel,
        run_seed=91,
        global_lane_ids=global_lane_ids,
    )


def test_neutral_runtime_preserves_text_multimodal_observations_and_actions() -> None:
    provider, runtime = text_runtime()
    observations = runtime.reset()
    assert observations["prompt"].tolist() == ["choose", "choose"]
    actions = {"answer": np.asarray(["yes", "no"], dtype=object)}
    step = runtime.step(actions)
    assert provider.last_actions["answer"].tolist() == ["yes", "no"]
    assert step.observations["prompt"].tolist() == ["choose", "choose"]
    assert step.final_observations["prompt"].tolist() == ["done-a", "done-b"]
    assert step.transition_info["tokens"].tolist() == [3, 5]


def test_global_lane_ids_make_seeds_invariant_to_actor_grouping() -> None:
    full_provider, full = text_runtime(global_lane_ids=(0, 1))
    full.reset()
    actor_provider, actor = text_runtime(global_lane_ids=(2, 3))
    actor.reset()
    regrouped_provider, regrouped = text_runtime(global_lane_ids=(2, 3))
    regrouped.reset()
    assert set(full_provider.reset_seeds[0]).isdisjoint(actor_provider.reset_seeds[0])
    assert actor_provider.reset_seeds[0] == regrouped_provider.reset_seeds[0]


@pytest.mark.parametrize(
    "private_state",
    [
        {"rollout_collector": []},
        {"replay": deque(maxlen=32)},
        {"apex_priorities": np.ones(8)},
        {"impala_fragments": deque(maxlen=2)},
        {"muzero_search": {"nodes": []}},
        {"jerk_retained_knowledge": {"facts": []}},
        {"llm_tokens": [], "inference_scheduler": object()},
    ],
)
def test_disparate_algorithm_state_can_remain_backend_private(private_state) -> None:
    class TestOnlyBackend:
        def __init__(self, state) -> None:
            self.private_state = state

        def validate(self, common_config, backend_config) -> None:
            del common_config, backend_config

        def run(self, context) -> None:
            context["coordinator_saw"] = self.private_state

    context: dict[str, object] = {}
    backend = TestOnlyBackend(private_state)
    backend.run(context)
    assert context["coordinator_saw"] is private_state
