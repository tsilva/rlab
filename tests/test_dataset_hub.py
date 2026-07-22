from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlab.dataset_hub import HubAppendSession


class FakeFeatures(dict):
    def to_dict(self):
        return dict(self)


class FakeDataset:
    def __init__(self, episode_ids, *, session_id="53a118b3-b31e-4f83-8ad4-b47cf1b05b33"):
        self.episode_ids = list(episode_ids)
        self.features = FakeFeatures(
            {
                "episode_id": {"dtype": "string", "_type": "Value"},
                "policy_actions": {"dtype": "null", "_type": "Value"},
            }
        )
        self.session_id = session_id

    def __getitem__(self, key):
        if key == "episode_id":
            return self.episode_ids
        if key == 0:
            return {"session_id": self.session_id}
        raise KeyError(key)

    def select(self, indices):
        return FakeDataset(
            [self.episode_ids[index] for index in indices], session_id=self.session_id
        )

    def cast(self, features):
        self.features = features
        return self

    def to_parquet(self, path):
        Path(path).write_bytes(b"parquet")


@dataclass
class FakeValidation:
    path: Path
    dataset: FakeDataset
    episode_fingerprints: dict[str, str]
    summary: object


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(oid=f"head-{len(self.calls)}")


def _session(api):
    session = HubAppendSession.__new__(HubAppendSession)
    session.target = "owner/repository"
    session.revision = "main"
    session.api = api
    session.head = "head-0"
    session.features = None
    session.episode_fingerprints = {}
    session.known_artifacts = {}
    session.sessions = []
    return session


def test_append_session_advances_expected_parent_without_rescanning(tmp_path):
    episode_id = "8f48fdb2-e6b2-49f8-97e3-b58f34f93d88"
    incoming = FakeValidation(
        path=tmp_path,
        dataset=FakeDataset([episode_id]),
        episode_fingerprints={episode_id: "a" * 64},
        summary=SimpleNamespace(environment_contracts=("b" * 64,), collector_contracts=()),
    )
    api = FakeApi()
    session = _session(api)

    with (
        patch("rlab.dataset_store.validate_tree", return_value=incoming),
        patch("rlab.dataset_hub._secret_egress_gate"),
    ):
        first_head = session.append(tmp_path)
        no_op_head = session.append(tmp_path)

    assert first_head == "head-1"
    assert no_op_head == "head-1"
    assert len(api.calls) == 1
    assert api.calls[0]["parent_commit"] == "head-0"
    assert session.episode_fingerprints == {episode_id: "a" * 64}
    assert len(session.sessions) == 1
