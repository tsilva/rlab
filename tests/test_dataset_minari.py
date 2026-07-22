from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("minari")

from rlab.dataset_minari import _materialize  # noqa: E402


class RowsDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _validation(tmp_path: Path, *, collector_terminated: bool = False):
    contract_id = "a" * 64
    environment = {
        "action_space": {"type": "Discrete", "dtype": "int64", "n": 3, "start": 0},
        # Regression: provider space differs from the canonical recorded RGB frame.
        "observation_space": {"type": "Box", "shape": [200, 256, 3], "dtype": "uint8"},
    }
    path = tmp_path / "environments" / contract_id
    path.mkdir(parents=True)
    (path / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    context = {
        "episode_id": "d0e75436-e660-4253-b43e-5b315b7e4a43",
        "seed": 3,
        "environment_contract_id": contract_id,
    }
    rows = [
        {
            **context,
            "step_index": 0,
            "actions": 2,
            "rewards": 1.0,
            "terminations": True,
            "truncations": False,
            "infos": '{"score":1}',
            "collector_terminated": False,
        },
        {
            **context,
            "step_index": 1,
            "actions": None,
            "rewards": None,
            "terminations": None,
            "truncations": None,
            "infos": None,
            "collector_terminated": collector_terminated,
        },
    ]
    return SimpleNamespace(path=tmp_path, dataset=RowsDataset(rows))


def test_minari_observation_space_comes_from_recorded_rgb_not_provider_space(tmp_path, monkeypatch):
    frames = [
        np.zeros((224, 256, 3), dtype=np.uint8),
        np.ones((224, 256, 3), dtype=np.uint8),
    ]
    monkeypatch.setattr("rlab.dataset_minari.iter_episode_frames", lambda rows, root: iter(frames))

    buffers, observation_space, action_space = _materialize(_validation(tmp_path))

    assert observation_space.shape == (224, 256, 3)
    assert action_space.contains(2)
    assert len(buffers) == 1
    assert len(buffers[0].observations) == 2


def test_minari_rejects_collector_boundary(tmp_path):
    with pytest.raises(ValueError, match="collector boundary"):
        _materialize(_validation(tmp_path, collector_terminated=True))
