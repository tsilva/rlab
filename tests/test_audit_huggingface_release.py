from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlab.publication import HUGGINGFACE_RELEASE_FILES


SCRIPT = Path(__file__).parents[1] / "scripts/audit_huggingface_release.py"
SPEC = importlib.util.spec_from_file_location("audit_huggingface_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeApi:
    def __init__(self, *, main_sha: str = "a" * 40, tag_sha: str = "a" * 40) -> None:
        self.main_sha = main_sha
        self.tag_sha = tag_sha

    def model_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        sha = self.tag_sha if kwargs.get("revision") else self.main_sha
        return SimpleNamespace(sha=sha, private=False)

    def list_repo_refs(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            tags=[SimpleNamespace(name="v1", target_commit=self.tag_sha)]
        )

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        return sorted(HUGGINGFACE_RELEASE_FILES)

    def list_collections(self, **kwargs: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                title="NES-SuperMarioBros Policies",
                slug="tsilva/nes-supermariobros-policies-id",
            )
        ]

    def get_collection(self, slug: str) -> SimpleNamespace:
        return SimpleNamespace(
                title="NES-SuperMarioBros Policies",
                slug=slug,
                private=False,
                items=[SimpleNamespace(item_id="tsilva/model", item_type="model")],
        )


def test_remote_release_audit_checks_commit_files_collection_and_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for filename in HUGGINGFACE_RELEASE_FILES:
        (tmp_path / filename).write_bytes(filename.encode())
    monkeypatch.setattr(
        MODULE,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: str(tmp_path / filename),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_release_bundle",
        lambda root: {
            "manifest_version": 1,
            "repository": {
                "repo_id": "tsilva/model",
                "game_family": "NES-SuperMarioBros",
            },
            "release": {"version": "v1"},
        },
    )
    monkeypatch.setattr(
        MODULE,
        "verify_replay",
        lambda path: {"codec_name": "h264", "frames": 100},
    )

    result = MODULE.audit_huggingface_release("tsilva/model", "v1", api=FakeApi())

    assert result["status"] == "passed"
    assert result["collection"] == "tsilva/nes-supermariobros-policies-id"


def test_remote_release_audit_rejects_main_tag_drift() -> None:
    with pytest.raises(ValueError, match="do not point to the same commit"):
        MODULE.audit_huggingface_release(
            "tsilva/model",
            "v1",
            api=FakeApi(main_sha="a" * 40, tag_sha="b" * 40),
        )
