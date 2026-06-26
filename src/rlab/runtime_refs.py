from __future__ import annotations

import re


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


def runtime_image_digest(value: str) -> str:
    match = DIGEST_IMAGE_REF_RE.fullmatch(normalize_runtime_image_ref(value))
    assert match is not None
    return match.group("digest").lower()


def runtime_image_digest_slug(value: str, *, length: int = 12) -> str:
    return runtime_image_digest(value)[:length]
