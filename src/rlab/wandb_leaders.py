from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from rlab.json_utils import json_safe
from rlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_BEST,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    EVAL_FULL_SUCCESS_RATE_MEAN,
    EVAL_FULL_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_BEST_RETURN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_OBJECTIVE_NAME,
    LEADER_CHECKPOINT_PROGRESS_MAX,
    LEADER_CHECKPOINT_RANK,
    LEADER_CHECKPOINT_RANK_VALUES,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_GOAL,
    LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
    LEGACY_TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
)
from rlab.ranking import parse_objective_rank, rank_score
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


RUN_OBJECTIVE_KEYS = (
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
    LEGACY_TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN,
)
RUN_PRIMARY_ORDER = "-created_at"
CHECKPOINT_SUCCESS_KEYS = (LEADER_CHECKPOINT_SUCCESS_RATE_MIN,)
CHECKPOINT_SUCCESS_MEAN_KEYS = (LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,)
CHECKPOINT_OBJECTIVE_KEYS = (LEADER_CHECKPOINT_OBJECTIVE,)
CHECKPOINT_PROGRESS_KEYS = (LEADER_CHECKPOINT_PROGRESS_MAX,)
CHECKPOINT_RETURN_KEYS = (LEADER_CHECKPOINT_RETURN_MEAN,)
CHECKPOINT_STEPS_TO_GOAL_KEYS = (LEADER_CHECKPOINT_STEPS_TO_GOAL,)
CHECKPOINT_STEP_KEYS = (LEADER_CHECKPOINT_STEP,)
# API ordering is only a retrieval hint. Goal-specific ranking happens in Python,
# because the primary objective may be either minimized or maximized.
CHECKPOINT_PRIMARY_ORDER = "-created_at"
WANDB_RUNS_PER_PAGE = 200


@dataclass(frozen=True)
class RunScore:
    goal_slug: str
    recipe_slug: str
    run_id: str
    run_name: str
    url: str
    seed: int | None
    objective: float


@dataclass(frozen=True)
class RunLeader:
    goal_slug: str
    recipe_slug: str
    seeds: int
    worst_seed: float
    mean_seed: float
    best_seed: float
    runs: tuple[RunScore, ...]


@dataclass(frozen=True)
class CheckpointLeader:
    goal_slug: str
    recipe_slug: str
    run_id: str
    run_name: str
    url: str
    objective: float
    objective_name: str
    success_rate_min: float | None
    success_rate_mean: float | None
    progress_max: float | None
    return_mean: float
    steps_to_goal: float | None
    checkpoint_step: int | None
    artifact_ref: str
    eval_source: str
    rank_score: tuple[float, ...]


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping.get(key)
    except AttributeError:
        return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_float(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _mapping_value(mapping, key)
        if value is None:
            continue
        try:
            return float(value)
        except TypeError, ValueError:
            continue
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _tag_value(tags: Iterable[Any], prefix: str) -> str:
    for tag in tags:
        text = str(tag or "")
        if text.startswith(prefix):
            return text[len(prefix) :]
    return ""


def _summary_metric_key(metric: str) -> str:
    return f"summary_metrics.{metric}"


def _exists_filter(metric: str) -> dict[str, Any]:
    return {_summary_metric_key(metric): {"$exists": True}}


def _and_filters(*filters: Mapping[str, Any] | None) -> dict[str, Any]:
    parts = [dict(item) for item in filters if item]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def goal_run_filter(goal: str | None) -> dict[str, Any]:
    if not goal:
        return {}
    return {
        "$or": [
            {"config.goal_slug": goal},
            {"tags": f"goal_id:{goal}"},
            {"tags": f"goal:{goal}"},
        ]
    }


def run_objective_filter(objective_keys: Sequence[str]) -> dict[str, Any]:
    return {"$or": [_exists_filter(key) for key in objective_keys]}


def run_query_objective_keys(args: argparse.Namespace) -> tuple[str, ...]:
    explicit_keys = tuple(args.objective_key or ())
    if explicit_keys:
        return explicit_keys
    return RUN_OBJECTIVE_KEYS


def checkpoint_summary_filter() -> dict[str, Any]:
    return {
        "$and": [
            {
                "$or": [
                    _exists_filter(LEADER_CHECKPOINT_OBJECTIVE),
                    _exists_filter(LEADER_CHECKPOINT_SUCCESS_RATE_MIN),
                ]
            },
            _exists_filter(LEADER_CHECKPOINT_RETURN_MEAN),
            _exists_filter(LEADER_CHECKPOINT_ARTIFACT_REF),
        ]
    }


def run_score(run: Any, *, objective_keys: Sequence[str]) -> RunScore | None:
    config = dict(getattr(run, "config", {}) or {})
    summary = getattr(run, "summary", {}) or {}
    objective = _first_float(summary, objective_keys)
    if objective is None:
        return None
    tags = tuple(getattr(run, "tags", ()) or ())
    goal_slug = _first_text(
        config.get("goal_slug"),
        _tag_value(tags, "goal_id:"),
        _tag_value(tags, "goal:"),
    )
    recipe_slug = _first_text(config.get("recipe_slug"), getattr(run, "group", ""))
    if not goal_slug or not recipe_slug:
        return None
    return RunScore(
        goal_slug=goal_slug,
        recipe_slug=recipe_slug,
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        seed=_optional_int(config.get("seed")),
        objective=float(objective),
    )


def rank_run_leaders(scores: Iterable[RunScore], *, min_seeds: int = 1) -> list[RunLeader]:
    grouped: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.goal_slug, score.recipe_slug)].append(score)

    leaders: list[RunLeader] = []
    for (goal_slug, recipe_slug), group_scores in grouped.items():
        if len(group_scores) < min_seeds:
            continue
        ordered_runs = tuple(sorted(group_scores, key=lambda item: item.objective, reverse=True))
        values = [item.objective for item in ordered_runs]
        leaders.append(
            RunLeader(
                goal_slug=goal_slug,
                recipe_slug=recipe_slug,
                seeds=len(ordered_runs),
                worst_seed=min(values),
                mean_seed=mean(values),
                best_seed=max(values),
                runs=ordered_runs,
            )
        )
    return sorted(
        leaders,
        key=lambda item: (item.worst_seed, item.mean_seed, item.best_seed, item.seeds),
        reverse=True,
    )


def checkpoint_leader(run: Any) -> CheckpointLeader | None:
    config = dict(getattr(run, "config", {}) or {})
    summary = getattr(run, "summary", {}) or {}
    success = _first_float(summary, CHECKPOINT_SUCCESS_KEYS)
    success_mean = _first_float(summary, CHECKPOINT_SUCCESS_MEAN_KEYS)
    objective = _first_float(summary, CHECKPOINT_OBJECTIVE_KEYS)
    objective_name = _first_text(_mapping_value(summary, LEADER_CHECKPOINT_OBJECTIVE_NAME))
    progress = _first_float(summary, CHECKPOINT_PROGRESS_KEYS)
    episode_return = _first_float(summary, CHECKPOINT_RETURN_KEYS)
    checkpoint_step = _optional_int(_first_float(summary, CHECKPOINT_STEP_KEYS))
    steps_to_goal = _first_float(summary, CHECKPOINT_STEPS_TO_GOAL_KEYS)
    artifact_ref = _first_text(
        _mapping_value(summary, LEADER_CHECKPOINT_ARTIFACT_REF),
    )
    if objective is None:
        return None
    if episode_return is None or not artifact_ref:
        return None
    tags = tuple(getattr(run, "tags", ()) or ())
    rank = parse_objective_rank(_mapping_value(summary, LEADER_CHECKPOINT_RANK))
    if not rank:
        rank = parse_objective_rank(config.get("selection_rank"))
    rank_metrics: dict[str, Any] = {
        EVAL_FULL_SUCCESS_RATE_MIN: success,
        EVAL_FULL_SUCCESS_RATE_MEAN: success_mean,
        EVAL_FULL_EPISODE_RETURN_MEAN: episode_return,
        EVAL_FULL_EPISODE_RETURN_BEST: _first_float(summary, (LEADER_CHECKPOINT_BEST_RETURN,)),
        LEADER_CHECKPOINT_STEPS_TO_GOAL: steps_to_goal,
        "checkpoint_step": checkpoint_step,
    }
    saved_rank_values = _mapping_value(summary, LEADER_CHECKPOINT_RANK_VALUES)
    if (
        rank
        and isinstance(saved_rank_values, Sequence)
        and not isinstance(saved_rank_values, str | bytes)
    ):
        rank_metrics.update(
            {
                criterion.metric: value
                for criterion, value in zip(rank, saved_rank_values, strict=False)
            }
        )
    if not rank:
        return None
    return CheckpointLeader(
        goal_slug=_first_text(
            config.get("goal_slug"),
            _tag_value(tags, "goal_id:"),
            _tag_value(tags, "goal:"),
        ),
        recipe_slug=_first_text(config.get("recipe_slug"), getattr(run, "group", "")),
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        objective=objective,
        objective_name=objective_name,
        success_rate_min=success,
        success_rate_mean=success_mean,
        progress_max=progress,
        return_mean=episode_return,
        steps_to_goal=steps_to_goal,
        checkpoint_step=checkpoint_step,
        artifact_ref=artifact_ref,
        eval_source=_first_text(_mapping_value(summary, LEADER_CHECKPOINT_EVAL_SOURCE)),
        rank_score=rank_score(rank_metrics, rank),
    )


def rank_checkpoint_leaders(leaders: Iterable[CheckpointLeader]) -> list[CheckpointLeader]:
    return sorted(
        leaders,
        key=lambda item: item.rank_score,
        reverse=True,
    )


def wandb_runs(
    *,
    project: str,
    goal: str | None = None,
    extra_filter: Mapping[str, Any] | None = None,
    order: str = "+created_at",
    lazy: bool = True,
):
    load_wandb_env()
    import wandb

    api = wandb.Api()
    filters = _and_filters(goal_run_filter(goal), extra_filter)
    return api.runs(
        project,
        filters=filters or None,
        order=order,
        per_page=WANDB_RUNS_PER_PAGE,
        lazy=lazy,
    )


def print_json(rows: Sequence[Any]) -> None:
    print(json.dumps(json_safe([asdict(row) for row in rows]), indent=2, sort_keys=True))


def print_run_leaders(rows: Sequence[RunLeader]) -> None:
    print("goal_slug\trecipe_slug\tseeds\tworst_seed\tmean_seed\tbest_seed")
    for row in rows:
        print(
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.seeds}\t"
            f"{row.worst_seed:.6g}\t{row.mean_seed:.6g}\t{row.best_seed:.6g}"
        )


def print_checkpoint_leaders(rows: Sequence[CheckpointLeader]) -> None:
    print(
        "goal_slug\trecipe_slug\tobjective\tobjective_name\tsuccess_min\t"
        "success_mean\tsteps_to_goal\treturn\tprogress\tstep\trun\tartifact_ref"
    )
    for row in rows:
        steps_to_goal = f"{row.steps_to_goal:.6g}" if row.steps_to_goal is not None else ""
        success_rate = f"{row.success_rate_min:.6g}" if row.success_rate_min is not None else ""
        success_rate_mean = (
            f"{row.success_rate_mean:.6g}" if row.success_rate_mean is not None else ""
        )
        progress = f"{row.progress_max:.6g}" if row.progress_max is not None else ""
        print(
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.objective:.6g}\t"
            f"{row.objective_name}\t{success_rate}\t"
            f"{success_rate_mean}\t{steps_to_goal}\t{row.return_mean:.6g}\t"
            f"{progress}\t"
            f"{row.checkpoint_step or ''}\t{row.run_name}\t{row.artifact_ref}"
        )


def add_common_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--project",
        default=argparse.SUPPRESS if suppress_defaults else DEFAULT_WANDB_PROJECT_PATH,
    )
    parser.add_argument(
        "--goal",
        default=default,
        help="Limit to one W&B config.goal_slug.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 20,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Print JSON instead of TSV.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query W&B leaderboards for rlab goals.")
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runs = subparsers.add_parser("runs", help="Rank run/recipe winners across seeds from W&B.")
    add_common_args(runs, suppress_defaults=True)
    runs.add_argument("--min-seeds", type=int, default=1)
    runs.add_argument(
        "--objective-key",
        action="append",
        default=[],
        help=(
            "W&B summary metric to rank; may be repeated. Defaults to the current primary "
            "goal metric."
        ),
    )
    runs.set_defaults(func=cmd_runs)

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="Rank best evaluated checkpoints from W&B run summaries.",
    )
    add_common_args(checkpoints, suppress_defaults=True)
    checkpoints.set_defaults(func=cmd_checkpoints)
    return parser


def cmd_runs(args: argparse.Namespace) -> int:
    objective_keys = tuple(args.objective_key or RUN_OBJECTIVE_KEYS)
    query_objective_keys = run_query_objective_keys(args)
    scores = [
        score
        for score in (
            run_score(run, objective_keys=objective_keys)
            for run in wandb_runs(
                project=args.project,
                goal=args.goal,
                extra_filter=run_objective_filter(query_objective_keys),
                order=RUN_PRIMARY_ORDER,
                lazy=False,
            )
        )
        if score is not None and (not args.goal or score.goal_slug == args.goal)
    ]
    leaders = rank_run_leaders(scores, min_seeds=max(1, int(args.min_seeds)))[
        : max(0, int(args.limit))
    ]
    if args.json:
        print_json(leaders)
    else:
        print_run_leaders(leaders)
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    leaders = [
        leader
        for leader in (
            checkpoint_leader(run)
            for run in wandb_runs(
                project=args.project,
                goal=args.goal,
                extra_filter=checkpoint_summary_filter(),
                order=CHECKPOINT_PRIMARY_ORDER,
            )
        )
        if leader is not None and (not args.goal or leader.goal_slug == args.goal)
    ]
    ranked = rank_checkpoint_leaders(leaders)[: max(0, int(args.limit))]
    if args.json:
        print_json(ranked)
    else:
        print_checkpoint_leaders(ranked)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
