from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from rlab.eval_runner import evaluate_policy_bundle
from rlab.policy_bundle import load_policy_bundle, write_canonical_json


RUNTIME_PACKAGES = (
    "rlab",
    "stable-baselines3",
    "stable-retro-turbo",
    "supermariobrosnes-turbo",
)


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _add_publication_fields(
    summary: dict[str, Any],
    *,
    checkpoint_step: int,
    checkpoint_artifact: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for episode in summary.get("episode_results") or []:
        start = str(episode.get("start_state") or "unknown")
        grouped.setdefault(start, []).append(episode)
    by_start = []
    for start, episodes in sorted(grouped.items()):
        success_count = sum(bool(item.get("level_complete")) for item in episodes)
        by_start.append(
            {
                "start_id": start,
                "episodes": len(episodes),
                "success_count": success_count,
                "success_rate": success_count / len(episodes),
                "return_mean": sum(float(item["return"]) for item in episodes)
                / len(episodes),
            }
        )
    summary.update(
        {
            "action_sampling": "stochastic",
            "protocol": "full",
            "checkpoint_step": checkpoint_step,
            "checkpoint_artifact": checkpoint_artifact,
            "by_start": by_start,
        }
    )


def _add_runtime_evidence(
    evidence: dict[str, Any],
    *,
    declared_runtime: dict[str, Any],
    actual_runtime: dict[str, str],
) -> None:
    declared_packages = dict(declared_runtime.get("packages") or {})
    evidence["declared_training_runtime"] = declared_runtime
    evidence["evaluation_runtime_packages"] = actual_runtime
    evidence["evaluation_runtime_matches_declared_training_runtime"] = all(
        actual_runtime.get(package) == version
        for package, version in declared_packages.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/hf_repair/evaluation"))
    parser.add_argument("--only-goal")
    parser.add_argument("--preview-episodes", type=int, default=10)
    args = parser.parse_args()

    inventory = _load_object(args.inventory)
    actual_runtime = _runtime_versions()
    results = []
    for row in inventory["repositories"]:
        if args.only_goal and row["goal_id"] != args.only_goal:
            continue
        root = Path(row["foundation"])
        output = args.output_root / row["repo_id"].replace("/", "__")
        output.mkdir(parents=True, exist_ok=True)
        bundle = load_policy_bundle(
            root,
            source=row["repo_id"],
            revision=row["revision"],
        )
        summary, _ = evaluate_policy_bundle(bundle, progress=True)
        _add_publication_fields(
            summary,
            checkpoint_step=int(bundle.model["checkpoint"]["step"]),
            checkpoint_artifact=str(row["artifact_ref"]),
        )
        evidence = summary["evaluation_evidence"]
        _add_runtime_evidence(
            evidence,
            declared_runtime=dict(bundle.recipe["provenance"]["runtime"]),
            actual_runtime=actual_runtime,
        )
        write_canonical_json(output / "evaluation.json", summary)
        if float(summary["eval/full/outcome/success/rate/min"]) < 1.0:
            results.append(
                {
                    "repo_id": row["repo_id"],
                    "goal_id": row["goal_id"],
                    "evaluation": str(output / "evaluation.json"),
                    "status": "blocked_acceptance",
                }
            )
            continue

        preview, replay = evaluate_policy_bundle(
            bundle,
            episodes=args.preview_episodes,
            n_envs=1,
            progress=True,
            capture_best_video=True,
            video_path=output / "replay.mp4",
        )
        _add_publication_fields(
            preview,
            checkpoint_step=int(bundle.model["checkpoint"]["step"]),
            checkpoint_artifact=str(row["artifact_ref"]),
        )
        preview_evidence = preview["evaluation_evidence"]
        _add_runtime_evidence(
            preview_evidence,
            declared_runtime=dict(bundle.recipe["provenance"]["runtime"]),
            actual_runtime=actual_runtime,
        )
        write_canonical_json(output / "preview_evaluation.json", preview)
        if replay is None or not replay.is_file():
            raise RuntimeError(f"preview evaluation did not produce replay.mp4: {row['repo_id']}")
        results.append(
            {
                "repo_id": row["repo_id"],
                "goal_id": row["goal_id"],
                "evaluation": str(output / "evaluation.json"),
                "preview_evaluation": str(output / "preview_evaluation.json"),
                "replay": str(replay),
                "status": "accepted",
            }
        )
    print(json.dumps({"evaluations": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
