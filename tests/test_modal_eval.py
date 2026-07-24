from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlab.eval_backend import EvalHandle
from rlab.modal_eval_backend import ModalEvalBackend
from rlab.modal_eval_config import load_modal_eval_config, modal_app_name
from rlab.modal_eval_protocol import SEED_PROTOCOL, build_execution_contract
from rlab.modal_eval_storage import file_sha256, write_downloaded_file
from rlab.modal_eval_worker import execute_attempt
from rlab.r2_store import PUBLIC_OBJECT_USER_AGENT


ROOT = Path(__file__).resolve().parents[1]


def _contract(model: Path) -> dict:
    return build_execution_contract(
        checkpoint_sha256=file_sha256(model),
        runtime_image_ref="docker:example.invalid/rlab@sha256:" + "b" * 64,
        eval_environment={"env_provider": "rlab", "game": "Bandit-v0", "task": {}},
        episodes=2,
        n_envs=2,
        max_steps=100,
        seed=10_000,
        seed_protocol=SEED_PROTOCOL,
        asset_manifest=None,
        recipe_sha256="c" * 64,
        recipe_format_version=1,
        evaluation_contract_sha256="d" * 64,
    )


def test_checked_in_modal_contract_is_cold_and_cost_bounded() -> None:
    config = load_modal_eval_config(ROOT / "experiments" / "modal_eval.yaml")

    assert config.cpu == 8
    assert config.memory_mib == 4096
    assert config.min_containers == 0
    assert config.buffer_containers == 0
    assert config.max_containers == 10
    assert config.max_attempts == 2
    assert config.alert_per_run_usd == 5


def test_modal_app_name_is_immutable_per_source() -> None:
    assert modal_app_name("rlab-eval-v2", "a" * 40) == "rlab-eval-v2-aaaaaaaaaaaa"
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        modal_app_name("rlab-eval-v2", "main")


def test_modal_download_uses_explicit_rlab_user_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    class Response:
        def __init__(self):
            self.payload = b"checkpoint"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int = -1) -> bytes:
            payload, self.payload = self.payload, b""
            return payload

    def urlopen(request, *, timeout):
        assert timeout == 60
        assert isinstance(request, urllib.request.Request)
        observed.append(str(request.get_header("User-agent")))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    target = write_downloaded_file(
        "https://models.example.test/model.zip",
        tmp_path / "model.zip",
    )

    assert target.read_bytes() == b"checkpoint"
    assert observed == [PUBLIC_OBJECT_USER_AGENT]


def test_backend_uses_spawn_poll_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class Function:
        @staticmethod
        def from_name(app, function, *, environment_name):
            calls.append(("from_name", (app, function, environment_name)))
            return SimpleNamespace(
                spawn=lambda payload: (
                    calls.append(("spawn", payload)) or SimpleNamespace(object_id="fc-1")
                )
            )

    class FunctionCall:
        @staticmethod
        def from_id(call_id):
            calls.append(("from_id", call_id))
            return SimpleNamespace(
                get=lambda *, timeout: {"ok": timeout == 0},
                cancel=lambda: calls.append(("cancel", call_id)),
            )

    fake_modal = SimpleNamespace(Function=Function, FunctionCall=FunctionCall)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    backend = ModalEvalBackend(
        app_name="rlab-eval-v2-aaaaaaaaaaaa",
        environment_name="rlab-eval",
    )

    handle = backend.submit({"intent": "value"})
    assert handle == EvalHandle(provider="modal", call_id="fc-1")
    assert backend.poll(handle).provider_result == {"ok": True}
    backend.cancel(handle)
    assert ("spawn", {"intent": "value"}) in calls
    assert ("cancel", "fc-1") in calls


def test_expired_attempt_persists_create_only_result_before_download(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"checkpoint")
    result = tmp_path / "result.json"
    payload = {
        "attempt_id": "attempt-1",
        "contract": _contract(model),
        "expires_at": time.time() - 1,
        "child_timeout_seconds": 10,
        "model_get_url": (tmp_path / "missing.zip").as_uri(),
        "result_uri": result.as_uri(),
        "result_put_url": result.as_uri(),
    }

    returned = execute_attempt(payload, cache_root=tmp_path / "cache")
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["status"] == "expired"
    assert returned["result_uri"] == result.as_uri()

    with pytest.raises(RuntimeError, match="different content"):
        execute_attempt({**payload, "attempt_id": "attempt-2"}, cache_root=tmp_path / "cache")
