from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence

from rlab import wandb_leaders
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH


PINNED_COLUMNS = [
    "run:name",
    "run:state",
    "run:group",
    "config:goal_slug.value",
    "config:recipe_slug.value",
    "config:spec_slug.value",
    "config:seed.value",
    "summary:leader/checkpoint/completion_rate",
    "summary:leader/checkpoint/completion_rate_mean",
    "summary:leader/checkpoint/steps_to_completion_goal",
    "summary:leader/checkpoint/reward_mean",
    "summary:leader/checkpoint/max_x_max",
    "summary:leader/checkpoint/step",
    "summary:leader/checkpoint/artifact_ref",
    "summary:leader/checkpoint/updated_at",
    "summary:eval/done/level_change/from_rate/min",
    "summary:eval/done/level_change/from_rate/mean",
    "summary:eval/reward/mean",
    "summary:eval/progress/x/max",
    "summary:eval/checkpoint/artifact",
]

VISIBLE_COLUMNS = PINNED_COLUMNS + [
    "tags:__ALL__",
    "summary:leader/checkpoint/eval_source",
]

COLUMN_WIDTHS = {
    "run:name": 320,
    "run:group": 220,
    "config:goal_slug.value": 140,
    "config:recipe_slug.value": 260,
    "config:spec_slug.value": 180,
    "summary:leader/checkpoint/artifact_ref": 460,
    "summary:leader/checkpoint/updated_at": 220,
    "summary:eval/checkpoint/artifact": 460,
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
            expr.Summary("leader/checkpoint/artifact_ref") != "",
            expr.Summary("eval/checkpoint/artifact") != "",
            expr.Summary("eval/checkpoint_artifact") != "",
        ),
    )


def goal_runset(*, entity: str, project: str, goal: str):
    from wandb.apis.reports import v2 as wr

    return wr.Runset(
        entity=entity,
        project=project,
        name=f"{goal} checkpoint leaders",
        filters=goal_filter(goal),
        order=[
            wr.OrderBy(wr.SummaryMetric("leader/checkpoint/completion_rate"), ascending=False),
            wr.OrderBy(wr.SummaryMetric("leader/checkpoint/completion_rate_mean"), ascending=False),
            wr.OrderBy(
                wr.SummaryMetric("leader/checkpoint/steps_to_completion_goal"),
                ascending=True,
            ),
            wr.OrderBy(wr.SummaryMetric("leader/checkpoint/reward_mean"), ascending=False),
            wr.OrderBy(wr.SummaryMetric("eval/done/level_change/from_rate/min"), ascending=False),
            wr.OrderBy(wr.SummaryMetric("eval/done/level_change/from_rate/mean"), ascending=False),
            wr.OrderBy(wr.SummaryMetric("eval/reward/mean"), ascending=False),
        ],
        pinned_columns=PINNED_COLUMNS,
        visible_columns=VISIBLE_COLUMNS,
        column_order=VISIBLE_COLUMNS,
        column_widths=COLUMN_WIDTHS,
        lock_columns=True,
    )


def build_report(*, entity: str, project: str, goals: Sequence[str], goal_counts: Counter[str], title: str):
    from wandb.apis.reports import v2 as wr

    blocks = [
        wr.MarkdownBlock(
            "Live checkpoint leaderboards grouped by goal. Each section is a W&B runset filtered "
            "to runs with evaluated checkpoint summaries and sorted by completion minimum, "
            "completion mean, least steps to completion goal, then reward."
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
                    runsets=[goal_runset(entity=entity, project=project, goal=goal)],
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
