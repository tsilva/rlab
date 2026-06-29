from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate promotion of a trained checkpoint against a goal contract."
    )
    parser.add_argument("--goal", required=True, help="Goal slug or path to goal.json.")
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate run name, artifact ref, model path, or checkpoint label.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inspect promotion readiness without publishing. Publishing is not automated yet.",
    )
    return parser


def resolve_goal_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    return Path("experiments/goals") / value / "goal.json"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    goal_path = resolve_goal_path(args.goal)
    if not goal_path.is_file():
        raise SystemExit(f"goal contract not found: {goal_path}")

    goal = json.loads(goal_path.read_text(encoding="utf-8"))
    rank_order = goal.get("selection_policy", {}).get("rank_order", [])
    print(f"goal={goal.get('goal_slug') or args.goal}")
    print(f"goal_path={goal_path}")
    print(f"candidate={args.candidate}")
    if rank_order:
        print("rank_order=" + ", ".join(str(metric) for metric in rank_order))
    print("promotion_status=blocked")
    raise SystemExit(
        "rlab promote is a strict gate only right now: publication still uses the "
        "project upload-checkpoint workflow so HF/YouTube/model-card evidence is not "
        "silently skipped."
    )


if __name__ == "__main__":
    main()
