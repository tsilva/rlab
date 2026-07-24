from __future__ import annotations

import re
from pathlib import Path


def safe_model_stem(value: str, fallback: str = "model") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return stem or fallback


def model_metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".metadata.json")


def model_zip_from_download(
    path: Path,
    *,
    expected_filename: str | None = None,
) -> Path:
    if path.is_file():
        if path.suffix != ".zip":
            raise ValueError(f"downloaded model is not a ZIP checkpoint: {path}")
        return path
    expected = path / expected_filename if expected_filename else None
    if expected is not None and expected.is_file():
        return expected
    candidates = sorted(path.rglob("*.zip"))
    if len(candidates) != 1:
        raise ValueError(
            f"downloaded model directory must contain exactly one ZIP: {path}"
        )
    return candidates[0]
