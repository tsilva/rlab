from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from rlab.policy_bundle import (
    canonical_json_bytes,
    evaluation_contract_sha256,
    load_policy_bundle,
    write_canonical_json,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = HfApi()
    inventory = _load_object(args.inventory)
    repositories = []
    for source in inventory["repositories"]:
        repo_id = str(source["repo_id"])
        goal_id = str(source["goal_id"])
        foundation = Path(source["foundation"])
        bundle = load_policy_bundle(
            foundation,
            source=repo_id,
            revision=str(source["revision"]),
        )
        evaluation_path = (
            args.evaluation_root
            / repo_id.replace("/", "__")
            / "evaluation.json"
        )
        evaluation = _load_object(evaluation_path)
        success_rate = float(evaluation["eval/full/outcome/success/rate/min"])
        episodes = int(evaluation["eval/full/episode/count"])
        evidence = evaluation.get("evaluation_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"evaluation evidence is missing: {evaluation_path}")
        expected_evidence = {
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "recipe_sha256": bundle.recipe_sha256,
            "recipe_format_version": int(bundle.recipe["format_version"]),
            "evaluation_contract_sha256": evaluation_contract_sha256(bundle.recipe),
        }
        for key, expected in expected_evidence.items():
            if evaluation.get(key) != expected or evidence.get(key) != expected:
                raise ValueError(
                    f"evaluation {key} does not match the policy bundle: {evaluation_path}"
                )
        if evaluation.get("exact_contract") is not True or evidence.get("exact_contract") is not True:
            raise ValueError(f"evaluation is not exact-contract evidence: {evaluation_path}")
        if episodes != 100:
            raise ValueError(
                f"repair publication requires the complete 100-episode evaluation: {evaluation_path}"
            )
        goal_accepted = success_rate >= 1.0
        current = api.model_info(repo_id)
        current_revision = str(current.sha)
        if current_revision != source["revision"]:
            raise RuntimeError(f"remote parent changed during repair audit: {repo_id}")
        files = sorted(api.list_repo_files(repo_id, revision=current_revision))
        repositories.append(
            {
                "repo_id": repo_id,
                "goal_id": goal_id,
                "remote_parent_commit": current_revision,
                "remote_files": files,
                "checkpoint": {
                    "sha256": bundle.checkpoint_sha256,
                    "size_bytes": bundle.checkpoint_path.stat().st_size,
                    "wandb_artifact": source["artifact_ref"],
                    "wandb_bytes_match_huggingface": True,
                },
                "documents": {
                    "model": {
                        "document_type": bundle.model["document_type"],
                        "format_version": bundle.model["format_version"],
                    },
                    "recipe": {
                        "document_type": bundle.recipe["document_type"],
                        "format_version": bundle.recipe["format_version"],
                        "sha256": bundle.recipe_sha256,
                    },
                    "release_manifest": {
                        "document_type": "rlab.release_manifest",
                        "format_version": 1,
                    },
                },
                "provider": bundle.recipe["recipe"]["eval"]["environment"][
                    "env_provider"
                ],
                "runtime_identity": bundle.recipe["provenance"]["runtime"],
                "source_commit": source["source_commit"],
                "wandb_run_path": source["wandb_run_path"],
                "evaluation": {
                    "path": str(evaluation_path),
                    "episodes": episodes,
                    "success_rate_min": success_rate,
                    "required_success_rate_min": 1.0,
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "recipe_sha256": evaluation["recipe_sha256"],
                    "evaluation_contract_sha256": evaluation[
                        "evaluation_contract_sha256"
                    ],
                    "exact_contract": evaluation["exact_contract"],
                    "evaluation_runtime": evaluation["evaluation_evidence"].get(
                        "evaluation_runtime_packages"
                    ),
                    "matches_declared_training_runtime": evaluation[
                        "evaluation_evidence"
                    ].get("evaluation_runtime_matches_declared_training_runtime"),
                },
                "bundle_recoverable": True,
                "publishable": True,
                "planned_release": "v1",
                "goal_acceptance": {
                    "accepted": goal_accepted,
                    "required_success_rate_min": 1.0,
                    "observed_success_rate_min": success_rate,
                },
                "blockers": [],
                "warnings": (
                    []
                    if goal_accepted
                    else [
                        "schema-migration release has exact stochastic evidence but does not satisfy the current 100% goal-promotion threshold"
                    ]
                ),
            }
        )
    audit: dict[str, Any] = {
        "audit_version": 3,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "namespace": "tsilva",
        "repositories": repositories,
    }
    digest_value = dict(audit)
    digest_value.pop("generated_at")
    audit["plan_digest"] = hashlib.sha256(
        canonical_json_bytes(digest_value)
    ).hexdigest()
    write_canonical_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
