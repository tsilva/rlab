from __future__ import annotations

import json
import os
import tempfile
import zipfile
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlab.trusted_inputs import (
    ModelApprovalError,
    SOURCE_ALLOWLIST_ENV,
    approve_staged_model,
    stage_model_input,
)
from rlab.policy_models import load_pinned_remote_policy_model


def _checkpoint(root: Path) -> Path:
    checkpoint = root / "model.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", "safe fixture")
    (root / "model.json").write_text(json.dumps({"document_type": "rlab.model"}))
    (root / "recipe.json").write_text(json.dumps({"document_type": "rlab.recipe"}))
    return checkpoint


def test_staging_copies_known_bundle_closure_with_private_permissions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        staged = stage_model_input(_checkpoint(Path(temporary)))
        try:
            assert {entry.path for entry in staged.manifest} == {
                "model.json",
                "model.zip",
                "recipe.json",
            }
            assert staged.model_path.read_bytes() != b""
            assert os.stat(staged.root).st_mode & 0o777 == 0o700
            assert all(
                os.stat(staged.root / entry.path).st_mode & 0o777 == 0o600
                for entry in staged.manifest
            )
        finally:
            staged.cleanup()


def test_staging_uses_current_metadata_precedence_and_ignores_lower_sidecar() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = _checkpoint(Path(temporary))
        checkpoint.with_suffix(".metadata.json").write_text('{"ignored": true}')
        staged = stage_model_input(checkpoint)
        try:
            assert "model.metadata.json" not in {entry.path for entry in staged.manifest}
        finally:
            staged.cleanup()


def test_release_manifest_missing_bound_file_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = _checkpoint(root)
        (root / "release_manifest.json").write_text(
            json.dumps({"artifacts": {"missing.bin": {"sha256": "a" * 64}}})
        )
        with pytest.raises(ValueError, match="binds missing"):
            stage_model_input(checkpoint)


def test_noninteractive_external_model_requires_exact_manifest_hash() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        staged = stage_model_input(_checkpoint(Path(temporary)))
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelApprovalError, match="approval is required"):
                approve_staged_model(staged, interactive=False)
        approved = approve_staged_model(
            staged,
            expected_hash=staged.manifest_hash,
            interactive=False,
        )
        approved.verify()
        approved.cleanup()


def test_source_allowlist_bypasses_interactive_approval_for_matching_source() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source = "hf://tsilva/policy@" + "a" * 40
        staged = stage_model_input(_checkpoint(Path(temporary)), source_identity=source)
        with patch.dict(
            os.environ,
            {SOURCE_ALLOWLIST_ENV: "hf://someone-else/*, hf://tsilva/*"},
            clear=True,
        ):
            approved = approve_staged_model(staged, interactive=False)
        approved.verify()
        approved.cleanup()


def test_source_allowlist_loads_from_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint_root = root / "checkpoint"
        checkpoint_root.mkdir()
        source = "hf://tsilva/policy@" + "a" * 40
        staged = stage_model_input(_checkpoint(checkpoint_root), source_identity=source)
        (root / ".env").write_text(
            f'{SOURCE_ALLOWLIST_ENV}="hf://tsilva/*"\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(root)
        monkeypatch.delenv(SOURCE_ALLOWLIST_ENV, raising=False)
        approved = approve_staged_model(staged, interactive=False)
        approved.verify()
        approved.cleanup()


def test_source_allowlist_does_not_override_a_wrong_approval_hash() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        source = "hf://tsilva/policy@" + "a" * 40
        staged = stage_model_input(_checkpoint(Path(temporary)), source_identity=source)
        with (
            patch.dict(
                os.environ,
                {SOURCE_ALLOWLIST_ENV: "hf://tsilva/*"},
                clear=True,
            ),
            pytest.raises(ModelApprovalError, match="approval hash mismatch"),
        ):
            approve_staged_model(
                staged,
                expected_hash="0" * 64,
                interactive=False,
            )
        staged.cleanup()


def test_changed_staged_bytes_fail_before_loader() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        staged = stage_model_input(_checkpoint(Path(temporary)))
        approved = approve_staged_model(staged, expected_hash=staged.manifest_hash)
        approved.model_path.write_bytes(b"changed")
        with pytest.raises(ModelApprovalError, match="changed"):
            approved.verify()
        approved.cleanup()


def test_source_symlink_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = _checkpoint(root)
        alias = root / "alias.zip"
        alias.symlink_to(checkpoint)
        with pytest.raises(ValueError, match="symlink"):
            stage_model_input(alias)


def test_pinned_remote_worker_reverifies_queued_manifest_before_load() -> None:
    source = "hf://owner/repository@" + "a" * 40 + "/model.zip"
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = _checkpoint(Path(temporary))
        staged = stage_model_input(checkpoint, source_identity=source)
        manifest = [entry.as_dict() for entry in staged.manifest]
        approval_hash = staged.manifest_hash
        staged.cleanup()
        sentinel = object()
        with (
            patch(
                "rlab.model_sources.download_remote_model_source",
                return_value=SimpleNamespace(model_path=checkpoint, artifact_name=source),
            ),
            patch("rlab.policy_models.load_policy_model", return_value=sentinel) as loader,
        ):
            loaded = load_pinned_remote_policy_model(
                source,
                download_root=Path(temporary) / "download",
                approval_hash=approval_hash,
                manifest=manifest,
                device="cpu",
            )

        assert loaded is sentinel
        loader.assert_called_once()


def test_sb3_deserialization_exists_only_behind_approved_loader() -> None:
    source_root = Path(__file__).parents[1] / "src" / "rlab"
    offenders = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr == "load"
                and isinstance(owner, ast.Name)
                and owner.id in {"PPO", "A2C"}
                and path.name != "sb3_models.py"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
