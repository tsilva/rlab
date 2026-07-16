from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from rlab.policy_bundle import canonical_json_bytes, sha256_file
from rlab.publication import HUGGINGFACE_RELEASE_FILES, validate_release_bundle
from rlab.wandb_utils import load_wandb_env


MARIO_ENVIRONMENT = "SuperMarioBros-Nes-v0"
LEGACY_REQUIRED_FILES = {"model.zip", "model_metadata.json"}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _next_release_tag(api: HfApi, repo_id: str) -> str:
    refs = api.list_repo_refs(repo_id, repo_type="model")
    versions = [
        int(match.group(1))
        for ref in refs.tags
        if (match := re.fullmatch(r"v([1-9][0-9]*)", str(ref.name)))
    ]
    return f"v{max(versions, default=0) + 1}"


def _wandb_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(metadata.get("wandb_run_id") or "").strip()
    project = str(metadata.get("wandb_project") or "").strip()
    if not run_id or not project:
        return {"status": "missing", "run_id": run_id, "project": project}
    try:
        load_wandb_env()
        import wandb

        run = wandb.Api().run(f"tsilva/{project}/{run_id}")
        config = dict(getattr(run, "config", {}) or {})
        summary = dict(getattr(run, "summary", {}) or {})
    except Exception as exc:
        return {
            "status": "unavailable",
            "run_id": run_id,
            "project": project,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "available",
        "run_id": run_id,
        "project": project,
        "runtime_image_ref": config.get("runtime_image_ref"),
        "source_sha": config.get("source_sha") or config.get("repo_git_commit"),
        "environment_hash": config.get("environment_hash"),
        "recipe_document": config.get("recipe_document"),
        "evaluation_evidence": summary.get("evaluation_evidence"),
    }


def _authoritative_recoverability(
    metadata: Mapping[str, Any], wandb: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    source_commit = str(
        metadata.get("repo_git_commit") or wandb.get("source_sha") or ""
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        blockers.append("missing exact 40-character source commit")
    runtime_image = str(
        metadata.get("runtime_image_ref") or wandb.get("runtime_image_ref") or ""
    ).strip()
    if not re.fullmatch(r"docker:[^\s]+@sha256:[0-9a-f]{64}", runtime_image):
        blockers.append("missing immutable training runtime image digest")
    recipe_document = wandb.get("recipe_document")
    if not (
        isinstance(recipe_document, Mapping)
        and recipe_document.get("document_type") == "rlab.recipe"
        and recipe_document.get("format_version") == 1
    ):
        blockers.append("no authoritative complete versioned recipe document")
    evaluation = wandb.get("evaluation_evidence")
    if not (isinstance(evaluation, Mapping) and evaluation.get("exact_contract") is True):
        blockers.append("no exact-contract stochastic evaluation evidence")
    return not blockers, blockers


def build_repair_audit(
    *, namespace: str, cache_root: Path, api: HfApi | None = None
) -> dict[str, Any]:
    api = api or HfApi()
    repositories: list[dict[str, Any]] = []
    for model in sorted(api.list_models(author=namespace), key=lambda item: str(item.id)):
        repo_id = str(model.id)
        info = api.model_info(repo_id, files_metadata=True)
        revision = str(info.sha)
        files = set(api.list_repo_files(repo_id, repo_type="model", revision=revision))
        if not LEGACY_REQUIRED_FILES.issubset(files):
            continue
        metadata_path = Path(
            hf_hub_download(
                repo_id,
                "model_metadata.json",
                repo_type="model",
                revision=revision,
                cache_dir=cache_root,
            )
        )
        metadata = _load_object(metadata_path)
        training = metadata.get("training_metadata")
        environment = training.get("environment") if isinstance(training, Mapping) else None
        env_id = str(environment.get("env_id") or "") if isinstance(environment, Mapping) else ""
        if MARIO_ENVIRONMENT not in env_id:
            continue
        checkpoint_path = Path(
            hf_hub_download(
                repo_id,
                "model.zip",
                repo_type="model",
                revision=revision,
                cache_dir=cache_root,
            )
        )
        wandb = _wandb_evidence(metadata)
        recoverable, blockers = _authoritative_recoverability(metadata, wandb)
        repositories.append(
            {
                "repo_id": repo_id,
                "remote_parent_commit": revision,
                "files": sorted(files),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "provider": env_id.split(":", 1)[0] if ":" in env_id else None,
                "environment_id": env_id,
                "wandb": wandb,
                "runtime_identity": {
                    "image_ref": metadata.get("runtime_image_ref")
                    or wandb.get("runtime_image_ref"),
                    "packages": (
                        training.get("runtime")
                        if isinstance(training, Mapping)
                        else None
                    ),
                },
                "contract_evidence": {
                    "environment_hash": (
                        training.get("environment_hash")
                        if isinstance(training, Mapping)
                        else None
                    ),
                    "source_commit": metadata.get("repo_git_commit")
                    or wandb.get("source_sha"),
                    "versioned_recipe": isinstance(wandb.get("recipe_document"), Mapping),
                    "exact_evaluation": isinstance(
                        wandb.get("evaluation_evidence"), Mapping
                    ),
                },
                "recoverable": recoverable,
                "blockers": blockers,
                "planned_changes": (
                    {
                        "preserve_checkpoint_bytes": True,
                        "replace_model_metadata_with_model_json": True,
                        "add_recipe_json": True,
                        "release_tag": _next_release_tag(api, repo_id),
                    }
                    if recoverable
                    else None
                ),
            }
        )
    audit: dict[str, Any] = {
        "audit_version": 1,
        "namespace": namespace,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "repositories": repositories,
    }
    digest_value = dict(audit)
    digest_value.pop("generated_at")
    audit["plan_digest"] = hashlib.sha256(canonical_json_bytes(digest_value)).hexdigest()
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the immutable pre-mutation audit for Mario policy repairs."
    )
    parser.add_argument("--namespace", default="tsilva")
    parser.add_argument("--cache-root", type=Path, default=Path("runs/hf_repair/cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-digest")
    parser.add_argument("--staging-root", type=Path, default=Path("runs/hf_repair/staged"))
    parser.add_argument("--apply", action="store_true")
    return parser


def apply_repair_audit(
    audit: Mapping[str, Any], *, staging_root: Path, api: HfApi
) -> list[dict[str, str]]:
    rows = list(audit.get("repositories") or [])
    blocked = [str(row["repo_id"]) for row in rows if not row.get("recoverable")]
    if blocked:
        raise RuntimeError(
            "repair is blocked by unrecoverable semantic evidence: " + ", ".join(blocked)
        )
    staged: list[tuple[Mapping[str, Any], Path]] = []
    for row in rows:
        repo_id = str(row["repo_id"])
        parent = str(row["remote_parent_commit"])
        if str(api.model_info(repo_id).sha) != parent:
            raise RuntimeError(f"remote parent changed for {repo_id}; rerun the audit")
        root = staging_root / repo_id.replace("/", "__")
        manifest = validate_release_bundle(root)
        if str(manifest["repository"]["repo_id"]) != repo_id:
            raise ValueError(f"staged repository identity mismatch for {repo_id}")
        if str(manifest["release"]["version"]) != str(
            row["planned_changes"]["release_tag"]
        ):
            raise ValueError(f"staged release tag mismatch for {repo_id}")
        if sha256_file(root / "model.zip") != str(row["checkpoint_sha256"]):
            raise ValueError(f"staged checkpoint bytes changed for {repo_id}")
        staged.append((row, root))

    results: list[dict[str, str]] = []
    for row, root in staged:
        repo_id = str(row["repo_id"])
        parent = str(row["remote_parent_commit"])
        existing = set(row["files"])
        operations = [
            CommitOperationDelete(path_in_repo=filename)
            for filename in sorted(existing - HUGGINGFACE_RELEASE_FILES)
        ]
        operations.extend(
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=root / filename)
            for filename in sorted(HUGGINGFACE_RELEASE_FILES)
        )
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=(
                "Publish versioned rlab policy bundle "
                + str(row["planned_changes"]["release_tag"])
            ),
            parent_commit=parent,
        )
        revision = str(getattr(commit, "oid", "") or "")
        if not revision:
            raise RuntimeError(f"Hugging Face did not return a commit for {repo_id}")
        tag = str(row["planned_changes"]["release_tag"])
        api.create_tag(repo_id, tag=tag, revision=revision, repo_type="model")
        results.append({"repo_id": repo_id, "commit": revision, "tag": tag})
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_repair_audit(namespace=args.namespace, cache_root=args.cache_root)
    if args.expected_plan_digest and args.expected_plan_digest != audit["plan_digest"]:
        raise ValueError("repair audit digest changed; rerun and review before mutation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(audit))
    if args.apply:
        if not args.expected_plan_digest:
            raise ValueError("--apply requires --expected-plan-digest")
        results = apply_repair_audit(
            audit,
            staging_root=args.staging_root,
            api=HfApi(),
        )
        print(json.dumps({"plan_digest": audit["plan_digest"], "repairs": results}, indent=2))
        return 0
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
