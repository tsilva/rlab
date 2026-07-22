from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

from rlab.dataset_contract import (  # noqa: E402
    ENVIRONMENT_DOCUMENT_FILENAME,
    canonical_column_order,
    canonical_json_bytes,
)
from rlab.dataset_store import adopt_tree, collection_paths, open_source  # noqa: E402


def _environment_document() -> dict:
    return {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": "rlab",
        "provider_contract_version": 1,
        "environment_id": "fixture-v0",
        "declared_config": {},
        "effective_config": {},
        "provenance": {"distribution": "rlab", "version": "0.1.0", "assets": {}},
        "action_space": {"type": "Discrete", "n": 2, "start": 0},
        "observation_space": {
            "type": "Box",
            "shape": [3, 4, 3],
            "dtype": "uint8",
            "low": 0,
            "high": 255,
        },
        "control_profile": {"type": "discrete", "actions": [0, 1]},
        "fps": 30,
    }


def _write_tree(path: Path, *, episode_id: str | None = None, reward: float = 1.0) -> str:
    episode_id = episode_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    environment = _environment_document()
    contract_id = hashlib.sha256(canonical_json_bytes(environment)).hexdigest()
    common = {
        "episode_id": episode_id,
        "seed": 7,
        "session_id": session_id,
        "dataset_format_version": 3,
        "collector": "random",
        "gymrec_version": "rlab:0.1.0",
        "storage_format": "images",
        "provider_id": "rlab",
        "env_id": "fixture-v0",
        "environment_contract_id": contract_id,
        "collector_contract_id": None,
        "policy_mode": "random",
        "policy_seed": 7,
    }
    frame0 = np.zeros((3, 4, 3), dtype=np.uint8)
    frame1 = np.full((3, 4, 3), 17, dtype=np.uint8)
    rows = [
        {
            **common,
            "step_index": 0,
            "actions": 1,
            "policy_actions": None,
            "rewards": reward,
            "terminations": True,
            "truncations": False,
            "infos": "{}",
            "collector_terminated": False,
            "observations": frame0,
        },
        {
            **common,
            "step_index": 1,
            "actions": None,
            "policy_actions": None,
            "rewards": None,
            "terminations": None,
            "truncations": None,
            "infos": None,
            "collector_terminated": False,
            "observations": frame1,
        },
    ]
    data = {name: [row[name] for row in rows] for name in canonical_column_order("images")}
    dataset = datasets.Dataset.from_dict(data)
    dataset = dataset.cast_column("observations", datasets.Image())
    dataset.save_to_disk(str(path))
    environment_path = path / "environments" / contract_id / ENVIRONMENT_DOCUMENT_FILENAME
    environment_path.parent.mkdir(parents=True)
    environment_path.write_bytes(canonical_json_bytes(environment))
    return episode_id


def test_adopt_is_transactional_and_idempotent(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "root"
    episode_id = _write_tree(source)

    first = adopt_tree(source, "fixture", root=root)
    second = adopt_tree(source, "fixture", root=root)

    assert first.collection_fingerprint == second.collection_fingerprint
    paths = collection_paths("fixture", root)
    assert paths.current.is_dir()
    assert not paths.next.exists()
    assert not paths.backup.exists()
    assert not paths.phase.exists()
    with open_source("fixture", root=root) as loaded:
        assert list(loaded.validation.episode_fingerprints) == [episode_id]


def test_adopt_rejects_duplicate_uuid_with_different_content(tmp_path):
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    root = tmp_path / "root"
    episode_id = _write_tree(first_source)
    _write_tree(second_source, episode_id=episode_id, reward=2.0)
    adopt_tree(first_source, "fixture", root=root)

    with pytest.raises(ValueError, match="episode UUID conflict"):
        adopt_tree(second_source, "fixture", root=root)


def test_reader_recovers_rename_boundary(tmp_path):
    source = tmp_path / "source"
    root = tmp_path / "root"
    _write_tree(source)
    adopted = adopt_tree(source, "fixture", root=root)
    paths = collection_paths("fixture", root)
    paths.current.rename(paths.backup)
    source.rename(paths.next)
    paths.phase.write_text(
        json.dumps(
            {
                "version": 1,
                "transaction_id": str(uuid.uuid4()),
                "had_current": True,
                "old_fingerprint": adopted.collection_fingerprint,
                "new_fingerprint": adopted.collection_fingerprint,
                "phase": "prepared",
            }
        ),
        encoding="utf-8",
    )
    with open_source("fixture", root=root) as loaded:
        assert loaded.validation.collection_fingerprint == adopted.collection_fingerprint
    assert paths.current.exists()
    assert not paths.backup.exists()
    assert not paths.phase.exists()


def test_adopt_rejects_symlinked_source_entry(tmp_path):
    source = tmp_path / "source"
    _write_tree(source)
    outside = tmp_path / "outside.json"
    outside.write_text("secret", encoding="utf-8")
    (source / "linked.json").symlink_to(outside)

    with pytest.raises(ValueError, match="unsafe entry"):
        adopt_tree(source, "fixture", root=tmp_path / "root")
