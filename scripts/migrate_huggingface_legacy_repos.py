from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from rlab.publication import (
    LEGACY_HUGGINGFACE_RELEASE_FILES,
    HUGGINGFACE_NAMESPACE,
    MIT_LICENSE_TEXT,
    assert_unique_repo_ids,
    build_model_repo_id,
    publication_identity_from_model_metadata,
    upgrade_legacy_model_metadata_for_publication,
)


LEGACY_MODEL_CLASS = "stable_baselines3.ppo.ppo.PPO"
LEGACY_REPO_PATTERN = re.compile(r"^SuperMarioBros-Nes-v0_(Level[0-9]+-[0-9]+)$")
LEGACY_TAG = "legacy-deterministic"
LEGACY_NOTICE = """## Release Status

This repository currently preserves a legacy release whose published evaluation used a
deterministic policy. It is not a schema-v1 `v1` release. A current `v1` will only be tagged
after stochastic reevaluation and representative replay validation.

## Licensing

The rlab-authored policy weights and publication material are licensed under the MIT License
in `LICENSE`. Stable Baselines3, emulator/runtime software, and Super Mario Bros game assets
are third-party works governed by their own licenses and terms. This repository does not
redistribute a game ROM.
"""


def _load_json(repo_id: str, filename: str) -> dict:
    path = Path(hf_hub_download(repo_id, filename, repo_type="model"))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{repo_id}/{filename} must contain a JSON object")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or apply the one-time migration of legacy Mario policy repositories."
    )
    parser.add_argument("--source-owner", default="tsilva")
    parser.add_argument("--target-namespace", default=HUGGINGFACE_NAMESPACE)
    parser.add_argument("--expected-count", type=int, default=9)
    parser.add_argument("--legacy-algorithm", required=True, choices=("ppo",))
    parser.add_argument("--legacy-crop-mode", required=True, choices=("mask", "remove"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--mapping-sha256",
        help="Required with --apply; copy the digest printed by a reviewed dry run.",
    )
    return parser


def _organization_names(api: HfApi) -> set[str]:
    whoami = api.whoami()
    organizations = whoami.get("orgs") if isinstance(whoami, Mapping) else None
    if not isinstance(organizations, list):
        return set()
    return {
        str(org.get("name") or "")
        for org in organizations
        if isinstance(org, Mapping) and org.get("name")
    }


def _mapping_digest(mapping: list[dict[str, object]]) -> str:
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _migrated_readme(
    text: str,
    *,
    source_id: str,
    destination_id: str,
    deleted_files: list[str],
) -> str:
    result = text.replace(source_id, destination_id)
    result = "\n".join(
        line
        for line in result.splitlines()
        if not (
            line.lstrip().startswith("|")
            and any(f"`{filename}`" in line for filename in deleted_files)
        )
    )
    if result.startswith("---\n"):
        closing = result.find("\n---\n", 4)
        if closing < 0:
            raise ValueError(f"{source_id}/README.md has invalid YAML front matter")
        front_matter = result[4:closing]
        if not re.search(r"(?m)^license:\s*", front_matter):
            front_matter = f"{front_matter.rstrip()}\nlicense: mit"
            result = f"---\n{front_matter}\n---\n{result[closing + 5:]}"
    else:
        raise ValueError(f"{source_id}/README.md is missing YAML front matter")
    if "## Release Status" not in result:
        result = f"{result.rstrip()}\n\n{LEGACY_NOTICE}"
    return result.rstrip() + "\n"


def _migration_plan(
    api: HfApi,
    *,
    source_owner: str,
    target_namespace: str,
    expected_count: int,
    legacy_algorithm: str,
    legacy_crop_mode: str,
) -> list[dict[str, object]]:
    candidates = []
    for info in api.list_models(author=source_owner, limit=500, full=True):
        repo_id = str(info.id)
        repo_name = repo_id.split("/", 1)[-1]
        match = LEGACY_REPO_PATTERN.fullmatch(repo_name)
        if match:
            candidates.append((repo_id, match.group(1), info))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) != expected_count:
        raise ValueError(
            f"expected {expected_count} legacy Mario repositories, found {len(candidates)}"
        )

    identities = []
    mapping = []
    for source_id, name_goal, info in candidates:
        metadata = _load_json(source_id, "model_metadata.json")
        metadata_goal = str(metadata.get("goal_slug") or "")
        if metadata_goal != name_goal:
            raise ValueError(
                f"{source_id} name goal {name_goal!r} disagrees with metadata {metadata_goal!r}"
            )
        tags = set(info.tags or [])
        if legacy_algorithm not in tags:
            raise ValueError(
                f"{source_id} does not carry the expected {legacy_algorithm!r} model tag"
            )
        upgraded = upgrade_legacy_model_metadata_for_publication(
            metadata,
            algorithm_id=legacy_algorithm,
            model_class=LEGACY_MODEL_CLASS,
            crop_mode=legacy_crop_mode,
        )
        identity = publication_identity_from_model_metadata(metadata_goal, upgraded)
        identities.append(identity)
        generated_id = build_model_repo_id(identity)
        destination_id = f"{target_namespace}/{generated_id.split('/', 1)[1]}"
        files = sorted(api.list_repo_files(source_id, repo_type="model"))
        planned_deletes = sorted(set(files) - LEGACY_HUGGINGFACE_RELEASE_FILES)
        mapping.append(
            {
                "source": source_id,
                "source_revision": str(info.sha or ""),
                "destination": destination_id,
                "goal": metadata_goal,
                "game_family": identity.game_family,
                "policy_variant": identity.policy_variant,
                "algorithm": identity.algorithm,
                "legacy_evaluation": "deterministic",
                "legacy_tag": LEGACY_TAG,
                "source_files": files,
                "planned_delete_files": planned_deletes,
                "planned_add_or_update_files": ["LICENSE", "README.md"],
                "result_files": sorted(LEGACY_HUGGINGFACE_RELEASE_FILES),
            }
        )
    assert_unique_repo_ids(identities)
    destinations = [entry["destination"] for entry in mapping]
    if len(destinations) != len(set(destinations)):
        raise ValueError("legacy migration destinations are not unique")
    return mapping


def _apply(api: HfApi, mapping: list[dict[str, object]], *, target_namespace: str) -> None:
    organizations = _organization_names(api)
    if target_namespace not in organizations:
        raise ValueError(
            f"authenticated account is not a member of target organization {target_namespace!r}"
        )
    existing = [
        str(item["destination"])
        for item in mapping
        if api.repo_exists(str(item["destination"]), repo_type="model")
    ]
    if existing:
        raise ValueError(f"migration destinations already exist: {existing}")

    # Tag every source before the first move so historical deterministic evidence
    # remains addressable even if a later move is interrupted.
    for item in mapping:
        api.create_tag(
            str(item["source"]),
            tag=LEGACY_TAG,
            tag_message="Legacy deterministic evaluation release before schema v1 migration",
            revision=str(item["source_revision"]),
            repo_type="model",
            exist_ok=True,
        )
    for item in mapping:
        api.move_repo(
            str(item["source"]),
            str(item["destination"]),
            repo_type="model",
        )
    for item in mapping:
        source_id = str(item["source"])
        destination_id = str(item["destination"])
        readme_path = Path(
            hf_hub_download(destination_id, "README.md", repo_type="model", force_download=True)
        )
        readme = _migrated_readme(
            readme_path.read_text(encoding="utf-8"),
            source_id=source_id,
            destination_id=destination_id,
            deleted_files=[str(filename) for filename in item["planned_delete_files"]],
        )
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()),
            CommitOperationAdd(
                path_in_repo="LICENSE", path_or_fileobj=MIT_LICENSE_TEXT.encode()
            ),
            *[
                CommitOperationDelete(path_in_repo=str(filename))
                for filename in item["planned_delete_files"]
            ],
        ]
        api.create_commit(
            destination_id,
            repo_type="model",
            operations=operations,
            commit_message="Migrate repository to deterministic publication identity",
        )

    families = sorted({str(item["game_family"]) for item in mapping})
    for family in families:
        collection = api.create_collection(
            f"{family} Policies",
            namespace=target_namespace,
            description=f"Promoted rlab reinforcement-learning policies for {family}.",
            private=False,
            exists_ok=True,
        )
        family_repos = [
            str(item["destination"]) for item in mapping if item["game_family"] == family
        ]
        for repo_id in family_repos:
            api.add_collection_item(
                collection.slug,
                repo_id,
                "model",
                exists_ok=True,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.target_namespace != HUGGINGFACE_NAMESPACE:
        raise ValueError(
            f"target namespace is code-owned and must be {HUGGINGFACE_NAMESPACE!r}"
        )
    api = HfApi()
    mapping = _migration_plan(
        api,
        source_owner=args.source_owner,
        target_namespace=args.target_namespace,
        expected_count=args.expected_count,
        legacy_algorithm=args.legacy_algorithm,
        legacy_crop_mode=args.legacy_crop_mode,
    )
    digest = _mapping_digest(mapping)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "mapping_sha256": digest,
        "count": len(mapping),
        "mapping": mapping,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.apply:
        return 0
    if args.mapping_sha256 != digest:
        raise ValueError(
            "--apply requires the exact --mapping-sha256 from the reviewed dry run"
        )
    _apply(api, mapping, target_namespace=args.target_namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
