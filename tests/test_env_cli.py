from __future__ import annotations

import io
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from rlab.batch_runtime import ProviderDescriptor
from rlab.env import bind_native_provider, make_native_provider, resolve_mixed_state_config
from rlab.env_cli import _finish_report, main as env_main
from rlab.env_config import env_config_from_mapping
from rlab.recipe_documents import compose_train_document


MARIO_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
MARIO_RECIPE = Path("experiments/recipes/mario/single/ppo.yaml")


class FakeNativeProvider:
    def __init__(self, n_envs: int = 16, *, break_masked_reset: bool = False):
        self.num_envs = n_envs
        self.single_observation_space = gym.spaces.Box(
            0,
            255,
            shape=(2,),
            dtype=np.uint8,
        )
        self.single_action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            n_envs,
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n_envs)
        self._observations = np.zeros((n_envs, 2), dtype=np.uint8)
        self._starts = np.asarray(["Level1-1"] * n_envs, dtype=object)
        self.break_masked_reset = break_masked_reset
        self.close_calls = 0

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = np.asarray(
            options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
            dtype=np.bool_,
        )
        effective_mask = np.ones_like(mask) if self.break_masked_reset else mask
        self._observations[effective_mask] = 0
        starts = options.get("start_ids")
        if starts is not None:
            for lane in np.flatnonzero(mask):
                if starts[lane] is not None:
                    self._starts[lane] = starts[lane]
        return self._observations, {
            "start_id": self._starts.copy(),
            "_start_id": mask.copy(),
        }

    def step(self, actions):
        del actions
        self._observations += 1
        return (
            self._observations,
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )

    def close(self):
        self.close_calls += 1


class FakeBatchRuntime:
    def __init__(self, native: FakeNativeProvider):
        self.provider = native
        self.num_envs = native.num_envs
        self.observation_space = native.single_observation_space
        self.action_space = gym.spaces.Discrete(2)
        self.global_lane_ids = tuple(range(native.num_envs))
        self.reset_infos = [{} for _ in range(native.num_envs)]

    def reset(self, *, seed=None, options_by_lane=None):
        del seed, options_by_lane
        return np.zeros((self.num_envs, 2), dtype=np.uint8)

    def step(self, actions):
        del actions
        return SimpleNamespace(
            observations=np.zeros((self.num_envs, 2), dtype=np.uint8),
            rewards=np.ones(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=np.bool_),
            truncated=np.zeros(self.num_envs, dtype=np.bool_),
            diagnostics=None,
        )

    def close(self):
        self.provider.close()


class MalformedStepProvider(FakeNativeProvider):
    def __init__(self, field: str):
        super().__init__()
        self.field = field

    def step(self, actions):
        result = list(super().step(actions))
        replacements = {
            "observations": np.zeros((self.num_envs, 3), dtype=np.uint8),
            "rewards": np.asarray(["bad"] * self.num_envs, dtype=object),
            "terminated": np.zeros(self.num_envs, dtype=np.int8),
            "truncated": np.zeros((self.num_envs, 1), dtype=np.bool_),
            "infos": {"bad": 1},
        }
        result[
            {"observations": 0, "rewards": 1, "terminated": 2, "truncated": 3, "infos": 4}[
                self.field
            ]
        ] = replacements[self.field]
        return tuple(result)


class WrongStartProvider(FakeNativeProvider):
    def reset(self, *, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        mask = np.asarray((options or {}).get("reset_mask"), dtype=np.bool_)
        if mask.any() and not mask.all():
            infos["start_id"][mask] = "wrong-start"
        return observations, infos


def _descriptor(native: FakeNativeProvider) -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="supermariobrosnes-turbo",
        native_observation_space=native.single_observation_space,
        native_action_space=native.single_action_space,
        start_catalog=("Level1-1",),
        lane_start_ids=("Level1-1",) * native.num_envs,
        render_support=("rgb_array",),
    )


def _distribution(root: Path):
    return SimpleNamespace(version="0.2.25", locate_file=lambda _value: root)


def _run_fake_check(
    tmp_path: Path,
    native: FakeNativeProvider,
    *,
    distribution_error: Exception | None = None,
    runtime_error: Exception | None = None,
    recipe_overrides: list[str] | None = None,
):
    package_root = tmp_path / "site-packages"
    module_path = package_root / "supermariobrosnes_turbo" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    module = SimpleNamespace(__file__=str(module_path))
    real_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "supermariobrosnes_turbo":
            return module
        return real_import_module(name, package)

    def bind_provider(*_args, **kwargs):
        return FakeBatchRuntime(kwargs["native_env"])

    stdout = io.StringIO()
    stderr = io.StringIO()
    distribution_kwargs = (
        {"side_effect": distribution_error}
        if distribution_error is not None
        else {"return_value": _distribution(package_root)}
    )
    with (
        patch("rlab.env.assert_provider_runtime_available", side_effect=runtime_error),
        patch(
            "rlab.env.make_native_provider",
            side_effect=lambda *_args, **_kwargs: (
                print("provider noise"),
                (native, _descriptor(native)),
            )[1],
        ),
        patch("rlab.env.bind_native_provider", side_effect=bind_provider),
        patch("importlib.metadata.distribution", **distribution_kwargs),
        patch("sys.stdout", stdout),
        patch("sys.stderr", stderr),
        patch("importlib.import_module", side_effect=import_module),
    ):
        argv = [
            "check",
            "--goal-file",
            str(MARIO_GOAL),
            "--recipe-file",
            str(MARIO_RECIPE),
            "--json",
        ]
        for override in recipe_overrides or []:
            argv.extend(("--set", override))
        exit_code = env_main(argv)
    return exit_code, json.loads(stdout.getvalue()), stderr.getvalue()


def test_static_cli_does_not_import_runtime_provider_modules() -> None:
    command = """
import json, sys
from rlab.env_cli import main
assert 'stable_retro' not in sys.modules
assert 'supermariobrosnes_turbo' not in sys.modules
assert 'breakout_turbo_env' not in sys.modules
assert main(['list', '--json']) == 0
assert 'stable_retro' not in sys.modules
assert 'supermariobrosnes_turbo' not in sys.modules
assert 'breakout_turbo_env' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert {item["provider_id"] for item in payload["providers"]} == {
        "ale-py",
        "breakout-turbo-env",
        "gymnasium",
        "rlab",
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
    }


def test_env_check_requires_goal_file() -> None:
    with pytest.raises(SystemExit):
        env_main(["check", "--recipe-file", str(MARIO_RECIPE)])


def test_inspect_reports_dynamic_gymnasium_contract() -> None:
    stdout = io.StringIO()
    with patch("sys.stdout", stdout):
        exit_code = env_main(["inspect", "gymnasium:CustomNativeVector-v0", "--json"])

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["environment"]["qualified_env_id"] == ("gymnasium:CustomNativeVector-v0")
    assert payload["environment"]["constructor_contract"] == {"kind": "dynamic"}


def test_check_keeps_json_clean_and_closes_provider(tmp_path: Path) -> None:
    native = FakeNativeProvider()

    exit_code, report, stderr = _run_fake_check(tmp_path, native)

    assert exit_code == 0
    assert report["summary"]["ok"] is True
    assert report["resolved"]["n_envs"] == 16
    assert "provider noise" in stderr
    assert native.close_calls == 1
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["visible_masked_reset"] == "passed"
    assert statuses["hidden_masked_reset_state"] == "not_observable"


def test_check_rejects_visible_masked_reset_corruption(tmp_path: Path) -> None:
    native = FakeNativeProvider(break_masked_reset=True)

    exit_code, report, _stderr = _run_fake_check(tmp_path, native)

    assert exit_code == 1
    assert report["summary"]["ok"] is False
    assert "visible_masked_reset" in report["summary"]["blocking_checks"]
    assert native.close_calls == 1


@pytest.mark.parametrize(
    "field",
    ("observations", "rewards", "terminated", "truncated", "infos"),
)
def test_check_rejects_malformed_native_step_outputs(tmp_path: Path, field: str) -> None:
    native = MalformedStepProvider(field)

    exit_code, report, _stderr = _run_fake_check(tmp_path, native)

    assert exit_code == 1
    assert report["summary"]["blocking_checks"] == ["native_step"]
    assert native.close_calls == 1


def test_check_rejects_masked_reset_reporting_the_wrong_start(tmp_path: Path) -> None:
    native = WrongStartProvider()

    exit_code, report, _stderr = _run_fake_check(tmp_path, native)

    assert exit_code == 1
    assert report["summary"]["blocking_checks"] == ["visible_masked_reset"]
    assert native.close_calls == 1


def test_check_accepts_one_lane_with_fixed_contract_evidence(tmp_path: Path) -> None:
    native = FakeNativeProvider(n_envs=1)

    exit_code, report, _stderr = _run_fake_check(
        tmp_path,
        native,
        recipe_overrides=["train.environment.env_config.n_envs=1"],
    )

    assert exit_code == 0
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["visible_masked_reset"]["status"] == "not_observable"
    assert checks["visible_masked_reset"]["evidence"] == "pinned_provider_contract"
    assert native.close_calls == 1


def test_check_reports_missing_provider_package_without_constructing(tmp_path: Path) -> None:
    native = FakeNativeProvider()

    exit_code, report, _stderr = _run_fake_check(
        tmp_path,
        native,
        distribution_error=importlib.metadata.PackageNotFoundError("missing provider"),
    )

    assert exit_code == 1
    assert report["summary"]["blocking_checks"] == ["provider_import"]
    assert native.close_calls == 0


def test_check_reports_missing_runtime_assets_without_constructing(tmp_path: Path) -> None:
    native = FakeNativeProvider()

    exit_code, report, _stderr = _run_fake_check(
        tmp_path,
        native,
        runtime_error=FileNotFoundError("missing ROM"),
    )

    assert exit_code == 1
    assert report["summary"]["blocking_checks"] == ["runtime_availability"]
    assert native.close_calls == 0


def test_required_inconclusive_dynamic_evidence_blocks_success() -> None:
    report = {"checks": [], "summary": {}}
    report["checks"].append(
        {
            "name": "visible_masked_reset",
            "required": True,
            "status": "not_observable",
            "evidence": "runtime",
            "detail": "one lane",
        }
    )

    assert _finish_report(report) is False
    assert report["summary"]["blocking_checks"] == ["visible_masked_reset"]


def test_native_construction_closes_provider_when_description_fails() -> None:
    document = compose_train_document(MARIO_GOAL, MARIO_RECIPE)
    config = resolve_mixed_state_config(
        env_config_from_mapping(document["train_config"]),
        n_envs=16,
    )
    native = FakeNativeProvider()

    with (
        patch("rlab.env.provider_native_vec_kwargs", return_value={}),
        patch("rlab.env.make_provider_vec_env", return_value=native),
        patch("rlab.env._provider_descriptor", side_effect=ValueError("bad descriptor")),
        pytest.raises(ValueError, match="bad descriptor"),
    ):
        make_native_provider(config, 16)

    assert native.close_calls == 1


def test_task_binding_failure_closes_unowned_native_provider() -> None:
    document = compose_train_document(MARIO_GOAL, MARIO_RECIPE)
    config = resolve_mixed_state_config(
        env_config_from_mapping(document["train_config"]),
        n_envs=16,
    )
    native = FakeNativeProvider()

    with (
        patch("rlab.env._bound_task_kernel", side_effect=ValueError("bad task")),
        pytest.raises(ValueError, match="bad task"),
    ):
        bind_native_provider(
            config,
            n_envs=16,
            seed=0,
            native_env=native,
            descriptor=_descriptor(native),
        )

    assert native.close_calls == 1
