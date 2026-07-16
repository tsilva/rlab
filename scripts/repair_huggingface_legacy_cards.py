from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, ModelCard, hf_hub_download

from rlab.publication import (
    HUGGINGFACE_NAMESPACE,
    LEGACY_HUGGINGFACE_RELEASE_FILES,
    build_model_repo_id,
    normalize_publication_evaluation,
    publication_identity_from_model_metadata,
    render_model_card,
    upgrade_legacy_model_metadata_for_publication,
)


LEGACY_MODEL_CLASS = "stable_baselines3.ppo.ppo.PPO"
LEGACY_TAG = "legacy-deterministic"
LEGACY_REPO_PATTERN = re.compile(r"^NES-SuperMarioBros_(Level\d+-\d+)_.+_ppo$")
SEED_PATTERN = re.compile(r"(?:^|[_-])s(\d+)(?:[_-]|$)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply the one-time canonical README and collection repair for "
            "legacy deterministic Mario model repositories."
        )
    )
    parser.add_argument("--owner", default=HUGGINGFACE_NAMESPACE)
    parser.add_argument("--expected-count", type=int, default=9)
    parser.add_argument(
        "--youtube-results-root",
        type=Path,
        default=Path("runs/hf_upload"),
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--plan-sha256",
        help="Required with --apply; copy the digest printed by a reviewed dry run.",
    )
    return parser


def _load_json(repo_id: str, filename: str) -> dict[str, Any]:
    path = Path(hf_hub_download(repo_id, filename, repo_type="model"))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{repo_id}/{filename} must contain a JSON object")
    return dict(value)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _youtube_url_for_goal(results_root: Path, goal: str) -> str:
    path = results_root / f"SuperMarioBros-Nes-v0_{goal}" / "youtube_upload_result.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing YouTube upload result for {goal}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    youtube_url = str(value.get("youtube_url") or "").strip()
    parsed = urllib.parse.urlparse(youtube_url)
    if parsed.scheme != "https" or parsed.netloc not in {"www.youtube.com", "youtube.com"}:
        raise ValueError(f"{path} does not contain a direct HTTPS YouTube URL")
    query = urllib.parse.urlencode({"url": youtube_url, "format": "json"})
    request = urllib.request.Request(
        f"https://www.youtube.com/oembed?{query}",
        headers={"User-Agent": "rlab-huggingface-release-repair/1"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise ValueError(f"YouTube oEmbed rejected {youtube_url}: HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, Mapping) or not payload.get("title"):
        raise ValueError(f"YouTube oEmbed returned invalid metadata for {youtube_url}")
    return youtube_url


def _legacy_card_inputs(
    repo_id: str,
    metadata: Mapping[str, Any],
    legacy_manifest: Mapping[str, Any],
    *,
    youtube_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    goal = str(metadata.get("goal_slug") or "").strip()
    upgraded = upgrade_legacy_model_metadata_for_publication(
        metadata,
        algorithm_id="ppo",
        model_class=LEGACY_MODEL_CLASS,
        crop_mode="remove",
    )
    identity = publication_identity_from_model_metadata(goal, upgraded)
    if build_model_repo_id(identity) != repo_id:
        raise ValueError(f"{repo_id} does not match its generated publication identity")
    metrics = legacy_manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{repo_id}/release_manifest.json does not contain legacy metrics")
    if metrics.get("deterministic") is not True:
        raise ValueError(f"{repo_id} legacy evaluation is not explicitly deterministic")
    episodes = int(metrics.get("episodes") or 0)
    success_count = int(metrics.get("completion_count") or 0)
    success_rate = float(metrics.get("completion_rate"))
    evaluation = normalize_publication_evaluation(
        {
            "action_sampling": "deterministic",
            "protocol": "full",
            "checkpoint_step": metrics.get("checkpoint_step"),
            "checkpoint_artifact": metrics.get("checkpoint_artifact"),
            "episodes": episodes,
            "success_rate_min": success_rate,
            "success_rate_mean": success_rate,
            "return_mean": metrics.get("reward_mean"),
            "progress_max": metrics.get("max_x_max"),
            "by_start": [
                {
                    "start_id": goal,
                    "episodes": episodes,
                    "success_count": success_count,
                    "success_rate": success_rate,
                    "return_mean": metrics.get("reward_mean"),
                }
            ],
        },
        allow_deterministic=True,
    ).as_manifest_value()
    training = upgraded["training_metadata"]
    environment = training["environment"]
    run_name = str(metadata.get("run_name") or "")
    seed_match = SEED_PATTERN.search(run_name)
    run_path = str(metadata.get("wandb_run_path") or "")
    run_parts = run_path.split("/")
    source = {
        "repository": "https://github.com/tsilva/rlab",
        "run_id": metadata.get("wandb_run_id"),
        "run_name": run_name,
        "wandb_project": run_parts[-2] if len(run_parts) >= 3 else None,
        "recipe": metadata.get("recipe_slug") or metadata.get("recipe_path"),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "checkpoint_step": evaluation["checkpoint_step"],
        "checkpoint_artifact": evaluation["checkpoint_artifact"],
    }
    card_manifest = {
        "repository": {"repo_id": repo_id, **asdict(identity)},
        "release": {"youtube_url": youtube_url},
        "model": {
            "algorithm_id": upgraded["algorithm_id"],
            "model_class": upgraded["model_class"],
            "qualified_env_id": environment["env_id"],
            "environment_hash": training.get("environment_hash"),
            "preprocessing": training["preprocessing"],
            "action": environment["task"]["action"],
        },
        "source": source,
        "evaluation": evaluation,
        "artifacts": {},
    }
    return card_manifest, upgraded


def _collection_state(api: HfApi, *, owner: str, title: str) -> tuple[str | None, set[str]]:
    matches = [
        collection
        for collection in api.list_collections(owner=owner, limit=100)
        if str(collection.title) == title
    ]
    if len(matches) > 1:
        raise ValueError(f"found duplicate collections titled {title!r}")
    if not matches:
        return None, set()
    collection = api.get_collection(str(matches[0].slug))
    if bool(collection.private):
        raise ValueError(f"collection {collection.slug} must be public")
    members = {
        str(item.item_id)
        for item in collection.items
        if str(item.item_type) == "model"
    }
    return str(collection.slug), members


def build_repair_plan(
    api: HfApi,
    *,
    owner: str,
    expected_count: int,
    youtube_results_root: Path,
) -> dict[str, Any]:
    candidates = []
    for info in api.list_models(author=owner, limit=500, full=True):
        repo_id = str(info.id)
        name = repo_id.split("/", 1)[-1]
        match = LEGACY_REPO_PATTERN.fullmatch(name)
        if match:
            candidates.append((repo_id, match.group(1), info))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) != expected_count:
        raise ValueError(
            f"expected {expected_count} legacy Mario repositories, found {len(candidates)}"
        )
    collection_title = "NES-SuperMarioBros Policies"
    collection_slug, members = _collection_state(api, owner=owner, title=collection_title)
    items = []
    for repo_id, name_goal, info in candidates:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        if LEGACY_TAG not in {str(tag.name) for tag in refs.tags}:
            raise ValueError(f"{repo_id} does not have the {LEGACY_TAG!r} tag")
        files = set(api.list_repo_files(repo_id, repo_type="model"))
        if files != set(LEGACY_HUGGINGFACE_RELEASE_FILES):
            raise ValueError(f"{repo_id} does not have the exact legacy release file set")
        metadata = _load_json(repo_id, "model_metadata.json")
        if str(metadata.get("goal_slug") or "") != name_goal:
            raise ValueError(f"{repo_id} name and metadata goals disagree")
        legacy_manifest = _load_json(repo_id, "release_manifest.json")
        youtube_url = _youtube_url_for_goal(youtube_results_root, name_goal)
        card_manifest, upgraded = _legacy_card_inputs(
            repo_id,
            metadata,
            legacy_manifest,
            youtube_url=youtube_url,
        )
        readme = render_model_card(card_manifest, upgraded, legacy=True)
        ModelCard(readme).validate(repo_type="model")
        old_readme = Path(
            hf_hub_download(repo_id, "README.md", repo_type="model")
        ).read_text(encoding="utf-8")
        items.append(
            {
                "repo_id": repo_id,
                "source_revision": str(info.sha or ""),
                "goal": name_goal,
                "youtube_url": youtube_url,
                "old_readme_sha256": _sha256_text(old_readme),
                "new_readme_sha256": _sha256_text(readme),
                "readme_changed": old_readme != readme,
                "collection_action": "none" if repo_id in members else "add",
                "_readme": readme,
            }
        )
    return {
        "owner": owner,
        "collection": {
            "title": collection_title,
            "slug": collection_slug,
            "action": "none" if collection_slug else "create",
        },
        "items": items,
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owner": plan["owner"],
        "collection": plan["collection"],
        "items": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in plan["items"]
        ],
    }


def _plan_digest(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _public_plan(plan), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _authenticated_names(api: HfApi) -> set[str]:
    whoami = api.whoami()
    names = {str(whoami.get("name") or "")} if isinstance(whoami, Mapping) else set()
    organizations = whoami.get("orgs") if isinstance(whoami, Mapping) else None
    if isinstance(organizations, list):
        names.update(
            str(org.get("name"))
            for org in organizations
            if isinstance(org, Mapping) and org.get("name")
        )
    return names


def apply_repair_plan(api: HfApi, plan: Mapping[str, Any]) -> None:
    owner = str(plan["owner"])
    if owner not in _authenticated_names(api):
        raise ValueError(f"authenticated account cannot write to namespace {owner!r}")
    for item in plan["items"]:
        if item["readme_changed"]:
            api.create_commit(
                str(item["repo_id"]),
                repo_type="model",
                operations=[
                    CommitOperationAdd(
                        path_in_repo="README.md",
                        path_or_fileobj=str(item["_readme"]).encode(),
                    )
                ],
                commit_message="Normalize legacy deterministic model card",
                parent_commit=str(item["source_revision"]),
            )
    collection = plan["collection"]
    slug = collection["slug"]
    if slug is None:
        created = api.create_collection(
            str(collection["title"]),
            namespace=owner,
            description=(
                "Promoted rlab reinforcement-learning policies for NES-SuperMarioBros."
            ),
            private=False,
            exists_ok=True,
        )
        slug = str(created.slug)
    for item in plan["items"]:
        if item["collection_action"] == "add":
            api.add_collection_item(
                str(slug),
                str(item["repo_id"]),
                "model",
                exists_ok=True,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api = HfApi()
    plan = build_repair_plan(
        api,
        owner=args.owner,
        expected_count=args.expected_count,
        youtube_results_root=args.youtube_results_root,
    )
    digest = _plan_digest(plan)
    if args.apply:
        if not args.plan_sha256:
            raise ValueError("--apply requires --plan-sha256 from a reviewed dry run")
        if args.plan_sha256 != digest:
            raise ValueError(
                f"repair plan digest changed: expected {args.plan_sha256}, current {digest}"
            )
        apply_repair_plan(api, plan)
    print(
        json.dumps(
            {
                **_public_plan(plan),
                "plan_sha256": digest,
                "applied": bool(args.apply),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
