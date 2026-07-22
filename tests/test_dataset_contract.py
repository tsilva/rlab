from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from rlab.dataset_contract import (
    COMMON_FIELDS,
    DATASET_FORMAT_VERSION,
    ENVIRONMENT_DOCUMENT_FILENAME,
    IMAGE_FIELDS,
    canonical_json_bytes,
    episode_content_fingerprint,
    features_append_compatible,
    validate_contract_artifacts,
    validate_v3,
)


class Rows(list):
    @property
    def column_names(self):
        return list(self[0])


def _digest_document(document):
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _environment_document():
    return {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": "stable-retro-turbo",
        "provider_contract_version": 1,
        "environment_id": "Airstriker-Genesis-v0",
        "declared_config": {},
        "effective_config": {"state": "Level1"},
        "provenance": {
            "distribution": "stable-retro-turbo",
            "version": "1.0.1.post32",
            "assets": {"rom_sha256": "a" * 64},
        },
        "action_space": {"type": "Discrete", "n": 2},
        "observation_space": {"type": "Box", "shape": [2, 2, 3]},
        "control_profile": "default",
        "fps": 60.0,
    }


def _rows(*, collector_terminated=False, provider_ended=True):
    episode_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    environment_id = _digest_document(_environment_document())
    common = {
        "episode_id": episode_id,
        "seed": 7,
        "session_id": session_id,
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "collector": "random",
        "gymrec_version": "0.1.1",
        "storage_format": "images",
        "provider_id": "stable-retro-turbo",
        "env_id": "Airstriker-Genesis-v0",
        "environment_contract_id": environment_id,
        "collector_contract_id": None,
        "policy_mode": None,
        "policy_seed": 7,
    }
    transition = {
        **common,
        "step_index": 0,
        "actions": 1,
        "policy_actions": None,
        "rewards": 1.0,
        "terminations": provider_ended,
        "truncations": False,
        "infos": json.dumps({"score": 1}),
        "collector_terminated": False,
        "observations": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    terminal = {
        **common,
        "step_index": 1,
        "actions": None,
        "policy_actions": None,
        "rewards": None,
        "terminations": None,
        "truncations": None,
        "infos": None,
        "collector_terminated": collector_terminated,
        "observations": np.ones((2, 2, 3), dtype=np.uint8),
    }
    order = [field.name for field in (*COMMON_FIELDS, *IMAGE_FIELDS)]
    return Rows([{key: row[key] for key in order} for row in (transition, terminal)])


def test_current_v3_accepts_nullable_transition_policy_action() -> None:
    summary = validate_v3(_rows())
    assert summary.episodes == 1
    assert summary.transitions == 1


def test_collector_boundary_must_match_provider_boundary() -> None:
    with pytest.raises(ValueError, match="collector_terminated"):
        validate_v3(_rows(collector_terminated=True, provider_ended=True))
    validate_v3(_rows(collector_terminated=True, provider_ended=False))


def test_terminal_transition_values_must_all_be_null() -> None:
    rows = _rows()
    rows[-1]["policy_actions"] = 2
    with pytest.raises(ValueError, match="final row has transition values"):
        validate_v3(rows)


def test_environment_documents_are_content_addressed() -> None:
    rows = _rows()
    summary = validate_v3(rows)
    document = _environment_document()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = (
            root / "environments" / summary.environment_contracts[0] / ENVIRONMENT_DOCUMENT_FILENAME
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(canonical_json_bytes(document))
        documents = validate_contract_artifacts(root, summary)
    assert list(documents) == [f"environment:{summary.environment_contracts[0]}"]


def test_only_policy_actions_null_feature_can_promote() -> None:
    null = {
        "actions": {"_type": "Value", "dtype": "int64"},
        "policy_actions": {"_type": "Value", "dtype": "null"},
    }
    concrete = {
        "actions": {"_type": "Value", "dtype": "int64"},
        "policy_actions": {"_type": "Value", "dtype": "int64"},
    }
    compatible, promotion = features_append_compatible(null, concrete)
    assert compatible and promotion
    assert features_append_compatible(concrete, null) == (True, False)


def test_episode_fingerprint_uses_float_bits_and_decoded_rgb() -> None:
    rows = _rows()
    first = episode_content_fingerprint(
        rows,
        frame_loader=lambda row: row["observations"],
        contract_documents={"environment": b"{}"},
    )
    rows[0]["rewards"] = np.nextafter(1.0, 2.0)
    second = episode_content_fingerprint(
        rows,
        frame_loader=lambda row: row["observations"],
        contract_documents={"environment": b"{}"},
    )
    assert first != second
