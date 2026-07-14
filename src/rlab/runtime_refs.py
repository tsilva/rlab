from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
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
DEFAULT_MODAL_WORKFLOW = "rlab Modal eval deployment"
DEFAULT_MODAL_ARTIFACT = "rlab-modal-eval-readiness"
DEFAULT_MODAL_ARTIFACT_FILE = "rlab-modal-eval-readiness.json"
MODAL_READINESS_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS = 20 * 60

DIGEST_IMAGE_REF_RE = re.compile(
    r"^docker:[^\s@]+@sha256:(?P<digest>[0-9a-fA-F]{64})$"
)
ACTIVE_WORKFLOW_STATUSES = frozenset(
    {"queued", "in_progress", "pending", "requested", "waiting"}
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


@dataclass(frozen=True)
class ModalReadinessInfo:
    runtime_image_ref: str
    source_sha: str
    modal_app_name: str
    startup_probe: dict[str, Any]
    workflow_run_id: str
    schema_version: int = MODAL_READINESS_SCHEMA_VERSION


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


def runtime_image_ref_from_payload(
    payload: Mapping[str, Any], *, label: str = "runtime image ref JSON"
) -> str:
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
            "legacy combined runtime artifacts cannot launch new training jobs; "
            "rebuild the exact source revision"
        )
    runtime_image_ref = runtime_image_ref_from_payload(payload, label=label)
    source_sha = str(payload.get("source_sha") or "").strip()
    if source_sha != expected_source_sha:
        raise ValueError(
            f"{label} source_sha mismatch: expected {expected_source_sha}, "
            f"got {source_sha or 'missing'}"
        )
    digest = str(payload.get("digest") or "").strip().removeprefix("sha256:")
    if digest and digest.lower() != runtime_image_digest(runtime_image_ref):
        raise ValueError(f"{label} digest does not match runtime_image_ref")
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
        train_config_contract_sha256=train_config_contract_sha256(),
    )


def modal_readiness_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_source_sha: str,
    expected_runtime_image_ref: str,
) -> ModalReadinessInfo:
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version != MODAL_READINESS_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be {MODAL_READINESS_SCHEMA_VERSION}"
        )
    source_sha = str(payload.get("source_sha") or "").strip()
    if source_sha != expected_source_sha:
        raise ValueError(
            f"{label} source_sha mismatch: expected {expected_source_sha}, "
            f"got {source_sha or 'missing'}"
        )
    runtime_image_ref = runtime_image_ref_from_payload(payload, label=label)
    expected_runtime_image_ref = normalize_runtime_image_ref(expected_runtime_image_ref)
    if runtime_image_ref != expected_runtime_image_ref:
        raise ValueError(f"{label} runtime image does not match the image receipt")
    modal_app_name = str(payload.get("modal_app_name") or "").strip()
    startup_probe = payload.get("startup_probe")
    if not modal_app_name or not isinstance(startup_probe, Mapping):
        raise ValueError(f"{label} must include Modal app and startup-probe evidence")
    expected_probe = {
        "runtime_image_ref": runtime_image_ref,
        "source_sha": source_sha,
        "train_config_contract_sha256": train_config_contract_sha256(),
        "app_name": modal_app_name,
    }
    for key, expected in expected_probe.items():
        if startup_probe.get(key) != expected:
            raise ValueError(f"{label} startup_probe.{key} does not match readiness")
    return ModalReadinessInfo(
        runtime_image_ref=runtime_image_ref,
        source_sha=source_sha,
        modal_app_name=modal_app_name,
        startup_probe=dict(startup_probe),
        workflow_run_id=str(payload.get("workflow_run_id") or "").strip(),
        schema_version=schema_version,
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


def current_git_branch(repo_root: Path | str = ".") -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=Path(repo_root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    branch = result.stdout.strip() if result.returncode == 0 else ""
    if not branch:
        raise RuntimeError("automatic runtime builds require a named current Git branch")
    return branch


def require_remote_source(
    source_sha: str, *, branch: str, repo_root: Path | str = "."
) -> None:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
        cwd=Path(repo_root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    remote_sha = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
    if result.returncode != 0 or remote_sha != source_sha:
        detail = result.stderr.strip()
        raise RuntimeError(
            f"exact source {source_sha} is not the pushed head of origin/{branch}; "
            f"push the commit before running queue-backed training"
            + (f" ({detail})" if detail else "")
        )


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


def _run_gh_bytes(command: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to download runtime artifacts") from exc
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{error}")
    return bytes(result.stdout)


def _run_gh(command: Sequence[str]) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to dispatch the runtime workflow") from exc
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{output}")


@lru_cache(maxsize=1)
def _repository_name() -> str:
    payload = _run_gh_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    name = str(payload.get("nameWithOwner") or "") if isinstance(payload, Mapping) else ""
    if not name:
        raise RuntimeError("gh repo view did not return nameWithOwner")
    return name


def _artifact_payload_for_run(
    run_id: str,
    artifact_name: str,
    artifact_file: str,
) -> dict[str, Any] | None:
    repository = _repository_name()
    listing = _run_gh_json(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    artifacts = listing.get("artifacts", []) if isinstance(listing, Mapping) else []
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, Mapping)
            and str(item.get("name") or "") == artifact_name
            and not bool(item.get("expired"))
        ),
        None,
    )
    if artifact is None:
        return None
    artifact_id = str(artifact.get("id") or "").strip()
    if not artifact_id:
        raise RuntimeError(f"artifact {artifact_name!r} from run {run_id} has no id")
    archive = _run_gh_bytes(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"]
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        try:
            text = bundle.read(artifact_file).decode("utf-8")
        except KeyError as exc:
            raise RuntimeError(
                f"artifact {artifact_name!r} from run {run_id} lacks {artifact_file}"
            ) from exc
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {artifact_name!r} from run {run_id} must contain an object")
    return dict(payload)


def _workflow_runs(
    *, workflow: str, branch: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    command = [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        str(limit),
        "--json",
        "databaseId,headSha,displayTitle,createdAt,updatedAt,status,conclusion,url",
    ]
    if branch:
        command[5:5] = ["--branch", branch]
    payload = _run_gh_json(command)
    return [dict(row) for row in payload if isinstance(row, Mapping)] if isinstance(payload, list) else []


def _matching_runs(
    *, source_sha: str, workflow: str, branch: str | None = None
) -> list[dict[str, Any]]:
    return [
        row
        for row in _workflow_runs(workflow=workflow, branch=branch)
        if str(row.get("headSha") or "") == source_sha
    ]


def recent_runtime_images(
    *,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str = DEFAULT_IMAGE_BRANCH,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
    limit: int = 3,
) -> tuple[RuntimeImageInfo, ...]:
    if limit <= 0:
        return ()
    images: list[RuntimeImageInfo] = []
    for run in _workflow_runs(workflow=workflow, branch=branch, limit=max(limit * 3, limit)):
        run_id = str(run.get("databaseId") or "").strip()
        if not run_id:
            continue
        payload = _artifact_payload_for_run(
            run_id, artifact_name, DEFAULT_IMAGE_ARTIFACT_FILE
        )
        if payload is None:
            continue
        source_sha = str(payload.get("source_sha") or run.get("headSha") or "").strip()
        info = runtime_release_from_payload(
            payload,
            label=f"runtime image receipt {run_id}",
            expected_source_sha=source_sha,
        )
        images.append(
            replace(
                info,
                commit_message=str(run.get("displayTitle") or info.commit_message).strip(),
                published_at=str(
                    run.get("updatedAt") or run.get("createdAt") or info.published_at
                ).strip(),
                workflow_run_id=info.workflow_run_id or run_id,
            )
        )
        if len(images) >= limit:
            break
    if not images:
        raise RuntimeError(f"no usable {workflow!r} image receipts found on branch {branch!r}")
    return tuple(images)


def runtime_release_for_source(
    *,
    source_sha: str,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str | None = None,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
) -> RuntimeImageInfo:
    errors: list[str] = []
    runs = _matching_runs(source_sha=source_sha, workflow=workflow, branch=branch)
    for run in runs:
        run_id = str(run.get("databaseId") or "").strip()
        if not run_id:
            continue
        try:
            payload = _artifact_payload_for_run(
                run_id, artifact_name, DEFAULT_IMAGE_ARTIFACT_FILE
            )
            if payload is None:
                continue
            info = runtime_release_from_payload(
                payload,
                label=f"runtime image receipt {run_id}",
                expected_source_sha=source_sha,
            )
            return replace(
                info,
                commit_message=str(run.get("displayTitle") or info.commit_message).strip(),
                published_at=str(
                    run.get("updatedAt") or run.get("createdAt") or info.published_at
                ).strip(),
                workflow_run_id=info.workflow_run_id or run_id,
            )
        except Exception as exc:
            errors.append(f"run {run_id}: {exc}")
    detail = f"no {workflow!r} workflow run exists for {source_sha}"
    if runs:
        latest = runs[0]
        detail = (
            f"latest workflow status={latest.get('conclusion') or latest.get('status')} "
            f"url={latest.get('url')}"
        )
    if errors:
        detail += "; invalid receipts: " + " | ".join(errors)
    raise RuntimeError(f"no exact-source runtime image receipt exists for {source_sha}; {detail}")


def _workflow_status_detail(runs: Sequence[Mapping[str, Any]]) -> str:
    if not runs:
        return "no workflow run is visible yet"
    latest = runs[0]
    return (
        f"status={latest.get('conclusion') or latest.get('status')} "
        f"url={latest.get('url')}"
    )


def wait_for_runtime_release(
    *,
    source_sha: str,
    workflow: str,
    branch: str,
    artifact_name: str,
    timeout: float,
    repo_root: Path | str = ".",
    poll_seconds: float = 5.0,
) -> RuntimeImageInfo:
    deadline = time.monotonic() + max(timeout, 0.0)
    dispatched = False
    run_ids_before_dispatch: set[str] = set()
    watched_run_ids: set[str] = set()
    while True:
        try:
            return runtime_release_for_source(
                source_sha=source_sha,
                workflow=workflow,
                branch=None,
                artifact_name=artifact_name,
            )
        except RuntimeError as receipt_error:
            runs = _matching_runs(source_sha=source_sha, workflow=workflow)
            active_runs = [
                row
                for row in runs
                if str(row.get("status") or "") in ACTIVE_WORKFLOW_STATUSES
            ]
            active = bool(active_runs)
            watched_run_ids.update(
                str(row.get("databaseId") or "") for row in active_runs
            )
            current_run_ids = {str(row.get("databaseId") or "") for row in runs}
            watched_terminal = bool(watched_run_ids & current_run_ids) and not active
            if watched_terminal:
                raise RuntimeError(
                    f"exact-source runtime workflow completed without a usable image receipt; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            if not active and not dispatched:
                run_ids_before_dispatch = {
                    str(row.get("databaseId") or "") for row in runs
                }
                require_remote_source(source_sha, branch=branch, repo_root=repo_root)
                _run_gh(
                    [
                        "gh",
                        "workflow",
                        "run",
                        workflow,
                        "--ref",
                        branch,
                        "-f",
                        f"source_sha={source_sha}",
                    ]
                )
                dispatched = True
                print(
                    f"runtime image missing; dispatched {workflow!r} for {source_sha}",
                    file=sys.stderr,
                    flush=True,
                )
            dispatched_terminal = bool(current_run_ids - run_ids_before_dispatch)
            if not active and dispatched and dispatched_terminal:
                raise RuntimeError(
                    f"exact-source runtime workflow completed without a usable image receipt; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for exact-source runtime image after {timeout:g}s; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            time.sleep(max(poll_seconds, 0.1))


def modal_readiness_for_release(
    release: RuntimeImageInfo,
    *,
    artifact_name: str = DEFAULT_MODAL_ARTIFACT,
    image_workflow: str = DEFAULT_IMAGE_WORKFLOW,
) -> ModalReadinessInfo:
    run_ids = [release.workflow_run_id] if release.workflow_run_id else []
    for workflow in (image_workflow, DEFAULT_MODAL_WORKFLOW):
        for row in _matching_runs(source_sha=release.source_sha, workflow=workflow):
            run_id = str(row.get("databaseId") or "").strip()
            if run_id and run_id not in run_ids:
                run_ids.append(run_id)
    errors: list[str] = []
    for run_id in run_ids:
        try:
            payload = _artifact_payload_for_run(
                run_id, artifact_name, DEFAULT_MODAL_ARTIFACT_FILE
            )
            if payload is None:
                continue
            return modal_readiness_from_payload(
                payload,
                label=f"Modal readiness receipt {run_id}",
                expected_source_sha=release.source_sha,
                expected_runtime_image_ref=release.runtime_image_ref,
            )
        except Exception as exc:
            errors.append(f"run {run_id}: {exc}")
    detail = "; ".join(errors) if errors else "readiness artifact is not available"
    raise RuntimeError(f"Modal is not ready for {release.runtime_image_ref}: {detail}")


def wait_for_modal_readiness(
    release: RuntimeImageInfo,
    *,
    timeout: float,
    image_workflow: str = DEFAULT_IMAGE_WORKFLOW,
    poll_seconds: float = 5.0,
) -> RuntimeImageInfo:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            readiness = modal_readiness_for_release(
                release,
                image_workflow=image_workflow,
            )
            return replace(
                release,
                modal_app_name=readiness.modal_app_name,
                startup_probe=readiness.startup_probe,
            )
        except RuntimeError as readiness_error:
            image_runs = _matching_runs(
                source_sha=release.source_sha, workflow=image_workflow
            )
            modal_runs = _matching_runs(
                source_sha=release.source_sha, workflow=DEFAULT_MODAL_WORKFLOW
            )
            runs = [*image_runs, *modal_runs]
            active = any(
                str(row.get("status") or "") in ACTIVE_WORKFLOW_STATUSES for row in runs
            )
            if runs and not active:
                raise RuntimeError(
                    f"Modal deployment completed without valid readiness for "
                    f"{release.runtime_image_ref}; {_workflow_status_detail(runs)}"
                ) from readiness_error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for Modal readiness after {timeout:g}s; "
                    f"{_workflow_status_detail(runs)}"
                ) from readiness_error
            time.sleep(max(poll_seconds, 0.1))


def runtime_release_from_args(
    args: Any,
    *,
    repo_root: Path | str = ".",
    checkpoint_eval_backend: str | None = None,
) -> RuntimeImageInfo:
    source_sha = clean_git_source_sha(repo_root)
    workflow = getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW)
    branch = getattr(args, "image_branch", None) or current_git_branch(repo_root)
    artifact_name = getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT)
    timeout = float(
        getattr(
            args,
            "runtime_readiness_timeout",
            DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
        )
    )
    ref_file = getattr(args, "runtime_image_ref_file", None)
    readiness_started = time.monotonic()
    if ref_file:
        payload = runtime_image_payload_from_file(Path(ref_file))
        release = runtime_release_from_payload(
            payload,
            label=f"runtime image descriptor {ref_file}",
            expected_source_sha=source_sha,
        )
    else:
        release = wait_for_runtime_release(
            source_sha=source_sha,
            workflow=workflow,
            branch=branch,
            artifact_name=artifact_name,
            timeout=timeout,
            repo_root=repo_root,
        )
    explicit_ref = getattr(args, "runtime_image_ref", None)
    if explicit_ref:
        normalized = normalize_runtime_image_ref(explicit_ref)
        if normalized != release.runtime_image_ref:
            raise ValueError(
                "explicit runtime image does not match the exact-source image receipt for "
                f"source {source_sha}: expected {release.runtime_image_ref}, got {normalized}"
            )
    if str(checkpoint_eval_backend or "") == "modal":
        remaining = max(timeout - (time.monotonic() - readiness_started), 0.0)
        release = wait_for_modal_readiness(
            release,
            timeout=remaining,
            image_workflow=workflow,
        )
    return release


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
        branch=getattr(args, "image_branch", None) or DEFAULT_IMAGE_BRANCH,
        artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
    )


def runtime_image_digest(value: str) -> str:
    match = DIGEST_IMAGE_REF_RE.fullmatch(normalize_runtime_image_ref(value))
    assert match is not None
    return match.group("digest").lower()
