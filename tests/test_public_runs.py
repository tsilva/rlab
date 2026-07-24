from __future__ import annotations

import urllib.request
from pathlib import Path

from rlab.model_sources import _public_json, download_public_run_source
from rlab.policy_bundle import (
    build_model_document,
    build_recipe_document,
    model_document_path,
    recipe_document_path,
    write_canonical_json,
)
from rlab.r2_store import BucketConfig, RunStorageConfig
from rlab.r2_store import PUBLIC_OBJECT_USER_AGENT
from rlab.recipe_documents import compose_train_document
from rlab.run_authority import RunAuthority
from rlab.run_contracts import PromotionReceipt, new_run_id, utc_now
from rlab.training_backend import training_backend_config_hash


GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = GOAL.parent / "recipes" / "ppo.yaml"
RUNTIME = "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "b" * 64


class _HttpResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _size: int = -1) -> bytes:
        return self.payload


def test_public_json_uses_explicit_rlab_user_agent(monkeypatch) -> None:
    observed: list[str] = []

    def urlopen(request, *, timeout):
        assert timeout == 30
        assert isinstance(request, urllib.request.Request)
        observed.append(str(request.get_header("User-agent")))
        return _HttpResponse(b'{"ok":true}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert _public_json("https://models.example.test/index.json") == {"ok": True}
    assert observed == [PUBLIC_OBJECT_USER_AGENT]


def _policy_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "model.zip"
    checkpoint.write_bytes(b"public checkpoint bytes")
    recipe = build_recipe_document(
        compose_train_document(GOAL, RECIPE),
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="public playback test",
        seed=123,
        runtime_image_ref=RUNTIME,
    )
    canonical_recipe = write_canonical_json(root / "recipe.json", recipe)
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 250_000,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_backend_id": "sb3.ppo",
        "training_backend_config_hash": training_backend_config_hash(
            recipe["recipe"]["train_config"]
        ),
    }
    write_canonical_json(
        model_document_path(checkpoint),
        build_model_document(checkpoint, canonical_recipe, metadata),
    )
    recipe_document_path(checkpoint).write_bytes(canonical_recipe.read_bytes())
    return checkpoint


def test_public_run_resolves_promoted_bundle_without_private_credentials(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    storage = RunStorageConfig(
        control=BucketConfig((tmp_path / "control").resolve().as_uri()),
        evaluation=BucketConfig((tmp_path / "eval").resolve().as_uri()),
        models=BucketConfig(
            models.resolve().as_uri(),
            public_base_url=models.resolve().as_uri(),
        ),
    )
    authority = RunAuthority(storage)
    run_id = new_run_id()
    checkpoint = _policy_checkpoint(tmp_path / "source")
    manifest = authority.publish_checkpoint(
        run_id=run_id,
        model_path=checkpoint,
        step=250_000,
        purpose="periodic",
        contract_hashes={
            "goal_sha256": "1" * 64,
            "recipe_sha256": "2" * 64,
            "environment_sha256": "3" * 64,
            "evaluation_contract_sha256": "4" * 64,
        },
        recovery_sidecar={"attempt_id": "attempt-" + "a" * 16},
    )
    authority.create_promotion(
        PromotionReceipt(
            run_id=run_id,
            checkpoint_id=manifest.checkpoint_id,
            checkpoint_step=manifest.step,
            eval_idempotency_key="5" * 64,
            eval_result_sha256="6" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )
    )

    resolved = download_public_run_source(
        run_id,
        root=tmp_path / "download",
        public_base_url=models.resolve().as_uri(),
    )

    assert resolved.model_path.read_bytes() == b"public checkpoint bytes"
    assert resolved.checkpoint_step == 250_000
    assert resolved.bundle is not None
    assert resolved.artifact_name is not None
    assert resolved.artifact_name.endswith("/manifest.json")
