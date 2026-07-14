from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/migrate_huggingface_legacy_repos.py"
)
SPEC = importlib.util.spec_from_file_location("migrate_legacy_repos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_migrated_readme_removes_deleted_file_rows() -> None:
    source = """---
pipeline_tag: reinforcement-learning
---

rlab play hf://tsilva/old

| File | Purpose |
|---|---|
| `model.zip` | Policy |
| `candidate_eval_result.json` | Staging evidence |
| `verify_remote/replay.mp4` | Staging copy |
| `replay.mp4` | Preview |
"""

    result = MODULE._migrated_readme(
        source,
        source_id="tsilva/old",
        destination_id="tsilva/new",
        deleted_files=["candidate_eval_result.json", "verify_remote/replay.mp4"],
    )

    assert "hf://tsilva/new" in result
    assert "candidate_eval_result.json" not in result
    assert "verify_remote/replay.mp4" not in result
    assert "`model.zip`" in result
    assert "`replay.mp4`" in result
