from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rlab import wandb_leaders
from rlab.metric_names import (
    EVAL_FULL_CHECKPOINT_ARTIFACT,
    EVAL_FULL_EPISODE_RETURN_BEST,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    EVAL_FULL_PROGRESS_X_MAX,
    EVAL_FULL_SUCCESS_RATE_MEAN,
    EVAL_FULL_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_BEST_RETURN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_PROGRESS_MAX,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_GOAL,
    LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_UPDATED_AT,
)
from rlab.config_validation import load_goal_contract
from rlab.ranking import RankCriterion, require_objective_rank
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH


PINNED_COLUMNS = [
    "run:name",
    "run:state",
    "run:group",
    "config:goal_slug.value",
    "config:recipe_slug.value",
    "config:seed.value",
    f"summary:{LEADER_CHECKPOINT_SUCCESS_RATE_MIN}",
    f"summary:{LEADER_CHECKPOINT_SUCCESS_RATE_MEAN}",
    f"summary:{LEADER_CHECKPOINT_STEPS_TO_GOAL}",
    f"summary:{LEADER_CHECKPOINT_RETURN_MEAN}",
    f"summary:{LEADER_CHECKPOINT_PROGRESS_MAX}",
    f"summary:{LEADER_CHECKPOINT_STEP}",
    f"summary:{LEADER_CHECKPOINT_ARTIFACT_REF}",
    f"summary:{LEADER_CHECKPOINT_UPDATED_AT}",
    f"summary:{EVAL_FULL_SUCCESS_RATE_MIN}",
    f"summary:{EVAL_FULL_SUCCESS_RATE_MEAN}",
    f"summary:{EVAL_FULL_EPISODE_RETURN_MEAN}",
    f"summary:{EVAL_FULL_PROGRESS_X_MAX}",
    f"summary:{EVAL_FULL_CHECKPOINT_ARTIFACT}",
]

VISIBLE_COLUMNS = PINNED_COLUMNS + [
    "tags:__ALL__",
    f"summary:{LEADER_CHECKPOINT_EVAL_SOURCE}",
]

COLUMN_WIDTHS = {
    "run:name": 320,
    "run:group": 220,
    "config:goal_slug.value": 140,
    "config:recipe_slug.value": 260,
    f"summary:{LEADER_CHECKPOINT_ARTIFACT_REF}": 460,
    f"summary:{LEADER_CHECKPOINT_UPDATED_AT}": 220,
    f"summary:{EVAL_FULL_CHECKPOINT_ARTIFACT}": 460,
}


def parse_project_path(value: str) -> tuple[str, str]:
    parts = value.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("project must be shaped as <entity>/<project>")
    return parts[0], parts[1]


def discover_goal_counts(project_path: str) -> Counter[str]:
    leaders = [
        leader
        for leader in (
            wandb_leaders.checkpoint_leader(run)
            for run in wandb_leaders.wandb_runs(
                project=project_path,
                extra_filter=wandb_leaders.checkpoint_summary_filter(),
                order=wandb_leaders.CHECKPOINT_PRIMARY_ORDER,
            )
        )
        if leader is not None and leader.goal_slug
    ]
    return Counter(leader.goal_slug for leader in leaders)


def goal_filter(goal: str):
    from wandb_workspaces import expr

    return expr.And(
        expr.Or(expr.Config("goal_slug") == goal, expr.Tags().isin([f"goal:{goal}"])),
        expr.Or(
            expr.Summary(LEADER_CHECKPOINT_ARTIFACT_REF) != "",
            expr.Summary(EVAL_FULL_CHECKPOINT_ARTIFACT) != "",
        ),
    )


def goal_rank(goal: str) -> tuple[RankCriterion, ...]:
    repo_root = Path(__file__).resolve().parents[1]
    matches = []
    for path in (repo_root / "experiments" / "goals").glob("**/_goal.yaml"):
        document = load_goal_contract(path, repo_root)
        if str(document.get("goal_id")) == goal:
            matches.append(document)
    if len(matches) != 1:
        raise ValueError(f"expected one local goal contract for {goal!r}, found {len(matches)}")
    return require_objective_rank(matches[0]["objective"]["rank"])


def leader_order_spec(criteria: Sequence[RankCriterion]) -> list[tuple[str, bool]]:
    mapped = {
        EVAL_FULL_SUCCESS_RATE_MIN: LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
        EVAL_FULL_SUCCESS_RATE_MEAN: LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
        EVAL_FULL_EPISODE_RETURN_MEAN: LEADER_CHECKPOINT_RETURN_MEAN,
        EVAL_FULL_EPISODE_RETURN_BEST: LEADER_CHECKPOINT_BEST_RETURN,
        LEADER_CHECKPOINT_STEPS_TO_GOAL: LEADER_CHECKPOINT_STEPS_TO_GOAL,
        LEADER_CHECKPOINT_STEP: LEADER_CHECKPOINT_STEP,
    }
    result: list[tuple[str, bool]] = []
    for index, criterion in enumerate(criteria):
        summary_metric = LEADER_CHECKPOINT_OBJECTIVE if index == 0 else mapped.get(criterion.metric)
        if summary_metric is None:
            raise ValueError(
                f"objective.rank criterion cannot be represented in the W&B leaderboard: "
                f"{criterion.metric}"
            )
        result.append((summary_metric, criterion.direction == "min"))
    return result


def goal_runset(*, entity: str, project: str, goal: str, criteria: Sequence[RankCriterion]):
    from wandb.apis.reports import v2 as wr

    return wr.Runset(
        entity=entity,
        project=project,
        name=f"{goal} checkpoint leaders",
        filters=goal_filter(goal),
        order=[
            wr.OrderBy(wr.SummaryMetric(metric), ascending=ascending)
            for metric, ascending in leader_order_spec(criteria)
        ],
        pinned_columns=PINNED_COLUMNS,
        visible_columns=VISIBLE_COLUMNS,
        column_order=VISIBLE_COLUMNS,
        column_widths=COLUMN_WIDTHS,
        lock_columns=True,
    )


def build_report(
    *, entity: str, project: str, goals: Sequence[str], goal_counts: Counter[str], title: str
):
    from wandb.apis.reports import v2 as wr

    blocks = [
        wr.MarkdownBlock(
            "Live checkpoint leaderboards grouped by goal. Each section is a W&B runset filtered "
            "to runs with evaluated checkpoint summaries and sorted by that goal's explicit "
            "objective.rank contract."
        )
    ]
    for goal in goals:
        count = goal_counts.get(goal, 0)
        blocks.extend(
            [
                wr.H2(goal),
                wr.MarkdownBlock(
                    f"Discovered {count} evaluated checkpoint leader row(s) for `{goal}` at report creation time."
                ),
                wr.PanelGrid(
                    runsets=[
                        goal_runset(
                            entity=entity,
                            project=project,
                            goal=goal,
                            criteria=goal_rank(goal),
                        )
                    ],
                    hide_run_sets=False,
                    panels=[],
                ),
            ]
        )
    return wr.Report(
        entity=entity,
        project=project,
        title=title,
        description="Automatically generated rlab checkpoint leaderboard report.",
        width="fluid",
        blocks=blocks,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a W&B checkpoint leaderboard report.")
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH)
    parser.add_argument("--title", default="rlab checkpoint leaderboards by goal")
    parser.add_argument(
        "--goal",
        action="append",
        default=[],
        help="Goal slug to include. May be repeated. Defaults to all goals with checkpoint leaders.",
    )
    parser.add_argument("--draft", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entity, project = parse_project_path(args.project)
    goal_counts = discover_goal_counts(args.project)
    goals = sorted(args.goal or goal_counts.keys())
    if not goals:
        raise SystemExit(f"No evaluated checkpoint leaders found in {args.project}")
    report = build_report(
        entity=entity,
        project=project,
        goals=goals,
        goal_counts=goal_counts,
        title=args.title,
    )
    report.save(draft=args.draft)
    print(report.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
