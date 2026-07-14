from __future__ import annotations

import pytest

from rlab.sb3_models import resolve_sb3_algorithm


def test_metadata_less_checkpoints_default_to_historical_ppo() -> None:
    assert resolve_sb3_algorithm({}) == "ppo"


def test_a2c_checkpoint_identity_resolves_consistently() -> None:
    assert (
        resolve_sb3_algorithm(
            {
                "training_backend_id": "sb3.a2c",
                "algorithm_id": "a2c",
                "model_class": "stable_baselines3.a2c.a2c.A2C",
            }
        )
        == "a2c"
    )


def test_checkpoint_identity_rejects_conflicting_metadata() -> None:
    with pytest.raises(ValueError, match="metadata disagree"):
        resolve_sb3_algorithm(
            {
                "training_backend_id": "sb3.a2c",
                "algorithm_id": "ppo",
            }
        )


def test_checkpoint_identity_rejects_unknown_model_class() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint model class"):
        resolve_sb3_algorithm({"model_class": "example.Unknown"})
