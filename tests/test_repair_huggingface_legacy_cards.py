from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from huggingface_hub import ModelCard


SCRIPT = Path(__file__).parents[1] / "scripts/repair_huggingface_legacy_cards.py"
SPEC = importlib.util.spec_from_file_location("repair_huggingface_legacy_cards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def legacy_metadata() -> dict:
    preprocessing = {
        "obs_resize": [84, 84],
        "obs_crop": [32, 0, 0, 0],
        "obs_grayscale": True,
        "frame_stack": 4,
        "policy_observation_layout": "channel_first",
    }
    return {
        "metadata_version": 2,
        "goal_slug": "Level1-1",
        "run_name": "Level1-1_base_s7_20260704T204541Z",
        "wandb_run_id": "run123",
        "wandb_run_path": "tsilva/SuperMarioBros-Nes-v0/run123",
        "recipe_slug": "base",
        "checkpoint_step": 4_000_000,
        "training_metadata": {
            "environment_hash": "sha256:environment",
            "environment": {
                "env_id": "supermariobrosnes-turbo:SuperMarioBros-Nes-v0",
                "action": {"action_set": "simple"},
            },
            "preprocessing": preprocessing,
        },
    }


def legacy_manifest() -> dict:
    return {
        "metrics": {
            "deterministic": True,
            "checkpoint_step": 4_000_000,
            "checkpoint_artifact": "tsilva/project/model:step-4000000",
            "episodes": 100,
            "completion_count": 99,
            "completion_rate": 0.99,
            "reward_mean": 3147.3,
            "max_x_max": 3129,
        }
    }


def test_legacy_repair_card_uses_canonical_renderer_and_marks_evidence() -> None:
    repo_id = (
        "tsilva/NES-SuperMarioBros_Level1-1_"
        "gray84-hudcrop-stack4-simple_ppo"
    )
    manifest, upgraded = MODULE._legacy_card_inputs(
        repo_id,
        legacy_metadata(),
        legacy_manifest(),
        youtube_url="https://www.youtube.com/watch?v=example",
    )

    card = MODULE.render_model_card(manifest, upgraded, legacy=True)

    ModelCard(card).validate(repo_type="model")
    assert "legacy-deterministic" in card.lower()
    assert "deterministic action selection" in card.lower()
    assert "99.0%" in card
    assert "supermariobrosnes-turbo" in card
    assert manifest["source"]["seed"] == 7
    assert hashlib.sha256(card.encode()).hexdigest() == (
        "abf75cd1f933c05a0472de2791d29ab117431ae92269cb6130f0ff17b1df74b7"
    )


def test_repair_plan_digest_ignores_embedded_readme_but_hashes_public_plan() -> None:
    plan = {
        "owner": "tsilva",
        "collection": {"title": "Policies", "slug": None, "action": "create"},
        "items": [
            {
                "repo_id": "tsilva/model",
                "new_readme_sha256": "abc",
                "_readme": "large generated body",
            }
        ],
    }
    changed_private = {
        **plan,
        "items": [{**plan["items"][0], "_readme": "different body"}],
    }
    changed_hash = {
        **plan,
        "items": [{**plan["items"][0], "new_readme_sha256": "def"}],
    }

    assert MODULE._plan_digest(plan) == MODULE._plan_digest(changed_private)
    assert MODULE._plan_digest(plan) != MODULE._plan_digest(changed_hash)
