from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_WORKFLOW = "rlab train image"
DEFAULT_IMAGE_BRANCH = "main"
DEFAULT_IMAGE_ARTIFACT = "rlab-train-image"
DEFAULT_IMAGE_ARTIFACT_FILE = "rlab-train-image.json"

DIGEST_IMAGE_REF_RE = re.compile(
    r"^docker:[^\s@]+@sha256:(?P<digest>[0-9a-fA-F]{64})$"
)


@dataclass(frozen=True)
class RuntimeImageInfo:
    runtime_image_ref: str
    source_sha: str
    commit_message: str
    published_at: str
    workflow_run_id: str


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


def runtime_image_payload_from_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"runtime image ref file is empty: {path}")
    if text.startswith("{"):
        payload: Any = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"runtime image ref file must contain a JSON object: {path}")
        return dict(payload)
    return {"runtime_image_ref": text}


def runtime_image_ref_from_payload(payload: Mapping[str, Any], *, label: str = "runtime image ref JSON") -> str:
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
    raise ValueError(f"{label} must include runtime_image_ref or image + digest")


def runtime_image_ref_from_file(path: Path) -> str:
    payload = runtime_image_payload_from_file(path)
    return runtime_image_ref_from_payload(payload, label=f"runtime image ref JSON in {path}")


def _download_runtime_image_payload(run_id: str, artifact_name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rlab-train-image-") as tmp:
        result = subprocess.run(
            [
                "gh",
                "run",
                "download",
                run_id,
                "--name",
                artifact_name,
                "--dir",
                tmp,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
            raise RuntimeError(f"failed to download artifact {artifact_name!r} from run {run_id}\n{output}")
        return runtime_image_payload_from_file(Path(tmp) / DEFAULT_IMAGE_ARTIFACT_FILE)


def recent_runtime_images(
    *,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str = DEFAULT_IMAGE_BRANCH,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
    limit: int = 3,
) -> tuple[RuntimeImageInfo, ...]:
    if limit <= 0:
        return ()
    runs = _run_gh_json(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--status",
            "success",
            "--limit",
            str(limit),
            "--json",
            "databaseId,headSha,displayTitle,createdAt,updatedAt",
        ]
    )
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"no successful {workflow!r} runs found on branch {branch!r}")
    images: list[RuntimeImageInfo] = []
    for run in runs[:limit]:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("databaseId") or "").strip()
        if not run_id:
            raise RuntimeError(f"{workflow!r} run did not include a databaseId")
        payload = _download_runtime_image_payload(run_id, artifact_name)
        runtime_image_ref = runtime_image_ref_from_payload(
            payload,
            label=f"runtime image artifact {artifact_name!r} from run {run_id}",
        )
        source_sha = str(payload.get("source_sha") or run.get("headSha") or "").strip()
        published_at = str(
            payload.get("published_at")
            or payload.get("created_at")
            or payload.get("publishedAt")
            or run.get("updatedAt")
            or run.get("createdAt")
            or ""
        ).strip()
        images.append(
            RuntimeImageInfo(
                runtime_image_ref=runtime_image_ref,
                source_sha=source_sha,
                commit_message=str(run.get("displayTitle") or payload.get("commit_message") or "").strip(),
                published_at=published_at,
                workflow_run_id=str(payload.get("workflow_run_id") or run_id).strip(),
            )
        )
    return tuple(images)


def _run_gh_json(command: Sequence[str]) -> Any:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to resolve the latest runtime image") from exc
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{output}")
    return json.loads(result.stdout or "null")


def latest_runtime_image_ref(
    *,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str = DEFAULT_IMAGE_BRANCH,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
) -> str:
    return recent_runtime_images(
        workflow=workflow,
        branch=branch,
        artifact_name=artifact_name,
        limit=1,
    )[0].runtime_image_ref


def runtime_image_ref_from_args(
    args: Any,
    *,
    default_latest: bool = False,
) -> str | None:
    ref_file = getattr(args, "runtime_image_ref_file", None)
    if ref_file:
        return runtime_image_ref_from_file(Path(ref_file))
    value = getattr(args, "runtime_image_ref", None)
    if value:
        return normalize_runtime_image_ref(value)
    if not default_latest:
        return None
    return latest_runtime_image_ref(
        workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
        branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
        artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
    )


def runtime_image_digest(value: str) -> str:
    match = DIGEST_IMAGE_REF_RE.fullmatch(normalize_runtime_image_ref(value))
    assert match is not None
    return match.group("digest").lower()


def runtime_image_digest_slug(value: str, *, length: int = 12) -> str:
    return runtime_image_digest(value)[:length]
