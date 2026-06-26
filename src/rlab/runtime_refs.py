from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DIGEST_IMAGE_REF_RE = re.compile(
    r"^docker:[^\s@]+@sha256:(?P<digest>[0-9a-fA-F]{64})$"
)


def normalize_runtime_image_ref(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("runtime image ref is required")
    if not DIGEST_IMAGE_REF_RE.fullmatch(text):
        raise ValueError(
            "runtime image ref must be an immutable docker digest ref like "
            "docker:ghcr.io/owner/image@sha256:<64-hex-digest>"
        )
    return text


def runtime_image_ref_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"runtime image ref file is empty: {path}")
    if text.startswith("{"):
        payload: Any = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"runtime image ref file must contain a JSON object: {path}")
        for key in ("runtime_image_ref", "image_ref", "docker_runtime_image_ref"):
            value = payload.get(key)
            if value:
                return normalize_runtime_image_ref(str(value))
        image = str(payload.get("image") or payload.get("image_name") or "").strip()
        digest = str(payload.get("digest") or "").strip()
        if image and digest:
            if not digest.startswith("sha256:"):
                digest = f"sha256:{digest}"
            return normalize_runtime_image_ref(f"docker:{image}@{digest}")
        raise ValueError(
            f"runtime image ref JSON must include runtime_image_ref or image + digest: {path}"
        )
    return normalize_runtime_image_ref(text)


def runtime_image_digest(value: str) -> str:
    match = DIGEST_IMAGE_REF_RE.fullmatch(normalize_runtime_image_ref(value))
    assert match is not None
    return match.group("digest").lower()


def runtime_image_digest_slug(value: str, *, length: int = 12) -> str:
    return runtime_image_digest(value)[:length]
