from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from rlab.publication import (
    HUGGINGFACE_RELEASE_FILES,
    validate_release_bundle,
    verify_replay,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one immutable rlab model release on the Hugging Face Hub."
    )
    parser.add_argument("repo_id")
    parser.add_argument("--revision", required=True, help="Sequential release tag, such as v1.")
    return parser


def _collection_with_title(collections: Iterable[Any], *, expected_title: str) -> Any:
    matches = [
        collection
        for collection in collections
        if str(collection.title) == expected_title
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one collection titled {expected_title!r}, found {len(matches)}"
        )
    return matches[0]


def audit_huggingface_release(
    repo_id: str,
    revision: str,
    *,
    api: HfApi | None = None,
) -> dict[str, Any]:
    if not revision.startswith("v") or not revision[1:].isdigit() or revision == "v0":
        raise ValueError("revision must be a sequential release tag such as v1 or v2")
    api = api or HfApi()
    main = api.model_info(repo_id, files_metadata=True)
    tagged = api.model_info(repo_id, revision=revision, files_metadata=True)
    if bool(main.private) or bool(tagged.private):
        raise ValueError(f"{repo_id} must be public")
    if str(main.sha) != str(tagged.sha):
        raise ValueError(
            f"main ({main.sha}) and {revision} ({tagged.sha}) do not point to the same commit"
        )

    refs = api.list_repo_refs(repo_id, repo_type="model")
    tags = {str(tag.name): str(tag.target_commit) for tag in refs.tags}
    if tags.get(revision) != str(tagged.sha):
        raise ValueError(f"tag {revision!r} is missing or does not point to {tagged.sha}")

    expected_files = set(HUGGINGFACE_RELEASE_FILES)
    main_files = set(api.list_repo_files(repo_id, repo_type="model"))
    tagged_files = set(api.list_repo_files(repo_id, revision=revision, repo_type="model"))
    if main_files != expected_files:
        raise ValueError(
            f"main file set mismatch; missing={sorted(expected_files - main_files)}, "
            f"extra={sorted(main_files - expected_files)}"
        )
    if tagged_files != expected_files:
        raise ValueError(
            f"{revision} file set mismatch; missing={sorted(expected_files - tagged_files)}, "
            f"extra={sorted(tagged_files - expected_files)}"
        )

    with tempfile.TemporaryDirectory(prefix="rlab-hf-audit-") as temporary:
        root = Path(temporary)
        for filename in sorted(HUGGINGFACE_RELEASE_FILES):
            cached = hf_hub_download(
                repo_id,
                filename,
                repo_type="model",
                revision=revision,
            )
            shutil.copy2(cached, root / filename)
        manifest = validate_release_bundle(root)
        replay = verify_replay(root / "replay.mp4")

    manifest_repo = str(manifest["repository"]["repo_id"])
    if manifest_repo != repo_id:
        raise ValueError(
            f"release manifest repository {manifest_repo!r} disagrees with {repo_id!r}"
        )
    manifest_version = str(manifest["release"]["version"])
    if manifest_version != revision:
        raise ValueError(
            f"release manifest version {manifest_version!r} disagrees with {revision!r}"
        )
    family = str(manifest["repository"]["game_family"])
    expected_title = f"{family} Policies"
    owner = repo_id.split("/", 1)[0]
    listed_collection = _collection_with_title(
        api.list_collections(owner=owner, limit=100),
        expected_title=expected_title,
    )
    collection = api.get_collection(str(listed_collection.slug))
    if bool(collection.private):
        raise ValueError(f"collection {collection.slug} must be public")
    members = {
        str(item.item_id)
        for item in collection.items
        if str(item.item_type) == "model"
    }
    if repo_id not in members:
        raise ValueError(f"collection {collection.slug} does not contain {repo_id}")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "commit": str(tagged.sha),
        "files": sorted(expected_files),
        "manifest_version": manifest["manifest_version"],
        "collection": str(collection.slug),
        "replay": replay,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            audit_huggingface_release(args.repo_id, args.revision),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
