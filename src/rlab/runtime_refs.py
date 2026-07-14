from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.runtime_contract import (
    RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
    train_config_contract_sha256,
)


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
    schema_version: int = 0
    train_config_contract_sha256: str = ""
    modal_app_name: str = ""
    startup_probe: dict[str, Any] | None = None


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


def docker_image_ref(runtime_image_ref: str) -> str:
    """Return Docker's image spelling for an immutable rlab runtime ref."""

    return normalize_runtime_image_ref(runtime_image_ref).removeprefix("docker:")


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
    value = payload.get("runtime_image_ref")
    if not value:
        raise ValueError(f"{label} must include runtime_image_ref")
    return normalize_runtime_image_ref(str(value))


def runtime_image_ref_from_file(path: Path) -> str:
    payload = runtime_image_payload_from_file(path)
    return runtime_image_ref_from_payload(payload, label=f"runtime image ref JSON in {path}")


def runtime_release_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_source_sha: str,
) -> RuntimeImageInfo:
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version != RUNTIME_DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be {RUNTIME_DESCRIPTOR_SCHEMA_VERSION}; "
            "legacy runtime artifacts cannot launch new training jobs"
        )
    runtime_image_ref = runtime_image_ref_from_payload(payload, label=label)
    source_sha = str(payload.get("source_sha") or "").strip()
    if source_sha != expected_source_sha:
        raise ValueError(
            f"{label} source_sha mismatch: expected {expected_source_sha}, got {source_sha or 'missing'}"
        )
    contract_sha = str(payload.get("train_config_contract_sha256") or "").strip()
    expected_contract_sha = train_config_contract_sha256()
    if contract_sha != expected_contract_sha:
        raise ValueError(
            f"{label} train-config contract mismatch: expected {expected_contract_sha}, "
            f"got {contract_sha or 'missing'}"
        )
    modal_app_name = str(payload.get("modal_app_name") or "").strip()
    startup_probe = payload.get("startup_probe")
    if not modal_app_name or not isinstance(startup_probe, Mapping):
        raise ValueError(f"{label} must include Modal deployment and startup-probe evidence")
    expected_probe = {
        "runtime_image_ref": runtime_image_ref,
        "source_sha": source_sha,
        "train_config_contract_sha256": contract_sha,
        "app_name": modal_app_name,
    }
    for key, expected in expected_probe.items():
        if startup_probe.get(key) != expected:
            raise ValueError(f"{label} startup_probe.{key} does not match the release descriptor")
    return RuntimeImageInfo(
        runtime_image_ref=runtime_image_ref,
        source_sha=source_sha,
        commit_message=str(payload.get("commit_message") or "").strip(),
        published_at=str(
            payload.get("published_at")
            or payload.get("created_at")
            or payload.get("publishedAt")
            or ""
        ).strip(),
        workflow_run_id=str(payload.get("workflow_run_id") or "").strip(),
        schema_version=schema_version,
        train_config_contract_sha256=contract_sha,
        modal_app_name=modal_app_name,
        startup_probe=dict(startup_probe),
    )


def clean_git_source_sha(repo_root: Path | str = ".") -> str:
    root = Path(repo_root)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "failed to inspect git worktree")
    if status.stdout.strip():
        raise RuntimeError(
            "queue-backed training requires a clean worktree so source and runtime are exact; "
            "commit or isolate local changes first"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError(revision.stderr.strip() or "failed to resolve git HEAD")
    return revision.stdout.strip()


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


def runtime_release_for_source(
    *,
    source_sha: str,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str = DEFAULT_IMAGE_BRANCH,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
) -> RuntimeImageInfo:
    runs = _run_gh_json(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--limit",
            "50",
            "--json",
            "databaseId,headSha,displayTitle,createdAt,updatedAt,status,conclusion,url",
        ]
    )
    matching = [
        run
        for run in (runs if isinstance(runs, list) else [])
        if isinstance(run, dict) and str(run.get("headSha") or "") == source_sha
    ]
    successful = [run for run in matching if str(run.get("conclusion") or "") == "success"]
    errors: list[str] = []
    for run in successful:
        run_id = str(run.get("databaseId") or "").strip()
        try:
            payload = _download_runtime_image_payload(run_id, artifact_name)
            info = runtime_release_from_payload(
                payload,
                label=f"runtime release {run_id}",
                expected_source_sha=source_sha,
            )
            return RuntimeImageInfo(
                **{
                    **info.__dict__,
                    "commit_message": str(run.get("displayTitle") or "").strip(),
                    "published_at": str(
                        run.get("updatedAt") or run.get("createdAt") or info.published_at
                    ).strip(),
                    "workflow_run_id": info.workflow_run_id or run_id,
                }
            )
        except Exception as exc:
            errors.append(f"run {run_id}: {exc}")
    latest = matching[0] if matching else None
    if latest:
        detail = (
            f"latest workflow conclusion={latest.get('conclusion') or latest.get('status')} "
            f"url={latest.get('url')}"
        )
    else:
        detail = f"no {workflow!r} workflow run exists for {source_sha}"
    if errors:
        detail += "; invalid releases: " + " | ".join(errors)
    raise RuntimeError(
        f"no release-complete runtime exists for source {source_sha}; {detail}. "
        "Wait for or repair the exact-source train-image workflow; older runtimes are not used."
    )


def runtime_release_from_args(args: Any, *, repo_root: Path | str = ".") -> RuntimeImageInfo:
    source_sha = clean_git_source_sha(repo_root)
    workflow = getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW)
    branch = getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH)
    artifact_name = getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT)
    ref_file = getattr(args, "runtime_image_ref_file", None)
    if ref_file:
        payload = runtime_image_payload_from_file(Path(ref_file))
        return runtime_release_from_payload(
            payload,
            label=f"runtime image descriptor {ref_file}",
            expected_source_sha=source_sha,
        )
    release = runtime_release_for_source(
        source_sha=source_sha,
        workflow=workflow,
        branch=branch,
        artifact_name=artifact_name,
    )
    explicit_ref = getattr(args, "runtime_image_ref", None)
    if explicit_ref:
        normalized = normalize_runtime_image_ref(explicit_ref)
        if normalized != release.runtime_image_ref:
            raise ValueError(
                "explicit runtime image does not match the release-complete descriptor for "
                f"source {source_sha}: expected {release.runtime_image_ref}, got {normalized}"
            )
    return release


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
