from __future__ import annotations

import argparse
import json
import os
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
    GLOBAL_STEP,
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
    METRICS_SCHEMA_VERSION,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
)
from rlab.ranking import parse_objective_rank, parse_persisted_objective_rank, rank_score
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env
from rlab.dotenv import load_env_file
from rlab.job_queue import connect
from rlab.telemetry_evidence import (
    authoritative_checkpoint_evidence,
    authoritative_run_facts,
)


RUN_OBJECTIVE_KEYS = (
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
    LEGACY_TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
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
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
    run_id: str
    run_name: str
    url: str
    seed: int | None
    objective: float
    steps: int | None = None


@dataclass(frozen=True)
class RunLeader:
    goal_slug: str
    recipe_slug: str
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
    seeds: int
    worst_seed: float
    mean_seed: float
    best_seed: float
    runs: tuple[RunScore, ...]
    mean_steps: float | None = None


@dataclass(frozen=True)
class CheckpointLeader:
    goal_slug: str
    recipe_slug: str
    reward_shape: str
    reward_shape_sha256: str
    effective_goal_contract_sha256: str
    reward_shape_is_default: bool
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
    configured_rank = parse_persisted_objective_rank(config.get("selection_rank"))
    configured_primary = (
        configured_rank[0].metric
        if configured_rank
        and (
            configured_rank[0].metric.startswith("train/")
            or configured_rank[0].metric == GLOBAL_STEP
        )
        else ""
    )
    candidate_keys = tuple(
        dict.fromkeys((configured_primary, *objective_keys) if configured_primary else objective_keys)
    )
    objective = _first_float(summary, candidate_keys)
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
        reward_shape=_first_text(config.get("reward_shape")),
        reward_shape_sha256=_first_text(config.get("reward_shape_sha256")),
        effective_goal_contract_sha256=_first_text(
            config.get("effective_goal_contract_sha256"),
            config.get("goal_contract_sha256"),
        ),
        reward_shape_is_default=bool(config.get("reward_shape_is_default", False)),
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        seed=_optional_int(config.get("seed")),
        objective=float(objective),
        steps=_optional_int(_mapping_value(summary, GLOBAL_STEP)),
    )


def rank_run_leaders(scores: Iterable[RunScore], *, min_seeds: int = 1) -> list[RunLeader]:
    grouped: dict[tuple[str, str, str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        shape_identity = score.reward_shape_sha256 or score.reward_shape or "legacy"
        grouped[
            (
                score.goal_slug,
                score.recipe_slug,
                shape_identity,
                score.effective_goal_contract_sha256,
            )
        ].append(score)

    leaders: list[RunLeader] = []
    for (
        goal_slug,
        recipe_slug,
        _shape_identity,
        effective_goal_hash,
    ), group_scores in grouped.items():
        if len(group_scores) < min_seeds:
            continue
        ordered_runs = tuple(
            sorted(
                group_scores,
                key=lambda item: (
                    item.objective,
                    -(item.steps if item.steps is not None else float("inf")),
                ),
                reverse=True,
            )
        )
        values = [item.objective for item in ordered_runs]
        step_values = [item.steps for item in ordered_runs if item.steps is not None]
        leaders.append(
            RunLeader(
                goal_slug=goal_slug,
                recipe_slug=recipe_slug,
                reward_shape=group_scores[0].reward_shape,
                reward_shape_sha256=group_scores[0].reward_shape_sha256,
                effective_goal_contract_sha256=effective_goal_hash,
                reward_shape_is_default=group_scores[0].reward_shape_is_default,
                seeds=len(ordered_runs),
                worst_seed=min(values),
                mean_seed=mean(values),
                best_seed=max(values),
                runs=ordered_runs,
                mean_steps=mean(step_values) if step_values else None,
            )
        )
    return sorted(
        leaders,
        key=lambda item: (
            item.reward_shape_is_default,
            item.reward_shape,
            item.worst_seed,
            item.mean_seed,
            item.best_seed,
            -(item.mean_steps if item.mean_steps is not None else float("inf")),
            item.seeds,
        ),
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
    try:
        schema_version = int(config.get("metrics_schema_version", METRICS_SCHEMA_VERSION))
    except TypeError, ValueError:
        schema_version = 4
    rank = parse_objective_rank(
        _mapping_value(summary, LEADER_CHECKPOINT_RANK), schema_version=4
    )
    if not rank:
        rank = parse_objective_rank(config.get("selection_rank"), schema_version=schema_version)
    if not rank:
        rank = parse_persisted_objective_rank(config.get("selection_rank"))
    rank_metrics: dict[str, Any] = {
        EVAL_FULL_SUCCESS_RATE_MIN: success,
        EVAL_FULL_SUCCESS_RATE_MEAN: success_mean,
        EVAL_FULL_EPISODE_RETURN_MEAN: episode_return,
        EVAL_FULL_EPISODE_RETURN_BEST: _first_float(summary, (LEADER_CHECKPOINT_BEST_RETURN,)),
        LEADER_CHECKPOINT_STEPS_TO_GOAL: steps_to_goal,
        LEADER_CHECKPOINT_STEP: checkpoint_step,
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
        reward_shape=_first_text(config.get("reward_shape")),
        reward_shape_sha256=_first_text(config.get("reward_shape_sha256")),
        effective_goal_contract_sha256=_first_text(
            config.get("effective_goal_contract_sha256"),
            config.get("goal_contract_sha256"),
        ),
        reward_shape_is_default=bool(config.get("reward_shape_is_default", False)),
        run_id=str(getattr(run, "id", "") or ""),
        run_name=str(getattr(run, "name", "") or ""),
        url=str(getattr(run, "url", "") or ""),
        objective=objective,
        objective_name=objective_name or rank[0].metric,
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
    grouped: dict[tuple[str, str, str], list[CheckpointLeader]] = defaultdict(list)
    for leader in leaders:
        shape_identity = leader.reward_shape_sha256 or leader.reward_shape or "legacy"
        grouped[(leader.goal_slug, shape_identity, leader.effective_goal_contract_sha256)].append(
            leader
        )
    ordered_groups = sorted(
        grouped.values(),
        key=lambda rows: (
            not rows[0].reward_shape_is_default,
            rows[0].reward_shape or "legacy",
        ),
    )
    return [
        leader
        for rows in ordered_groups
        for leader in sorted(rows, key=lambda item: item.rank_score, reverse=True)
    ]


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
    print(
        "goal_slug\trecipe_slug\treward_shape\tseeds\tworst_seed\tmean_seed\tbest_seed"
        "\tmean_steps"
    )
    for row in rows:
        mean_steps = f"{row.mean_steps:.6g}" if row.mean_steps is not None else ""
        print(
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.reward_shape or 'legacy'}\t{row.seeds}\t"
            f"{row.worst_seed:.6g}\t{row.mean_seed:.6g}\t{row.best_seed:.6g}\t{mean_steps}"
        )


def print_checkpoint_leaders(rows: Sequence[CheckpointLeader]) -> None:
    print(
        "goal_slug\trecipe_slug\treward_shape\tobjective\tobjective_name\tsuccess_min\t"
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
            f"{row.goal_slug}\t{row.recipe_slug}\t{row.reward_shape or 'legacy'}\t"
            f"{row.objective:.6g}\t"
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
        "--reward-shape",
        default=default,
        help="Limit to one W&B config.reward_shape; required for an unambiguous winner query.",
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
    parser.add_argument(
        "--database-url",
        default=argparse.SUPPRESS if suppress_defaults else os.environ.get("DATABASE_URL"),
        help="Authoritative telemetry PostgreSQL URL (defaults to DATABASE_URL).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab leaders",
        description="Query exact authoritative telemetry evidence for rlab goals.",
    )
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runs = subparsers.add_parser(
        "runs", help="Rank complete exact run cohorts from run_final_exact facts."
    )
    add_common_args(runs, suppress_defaults=True)
    runs.add_argument("--min-seeds", type=int, default=1)
    runs.add_argument(
        "--objective-key",
        action="append",
        default=[],
        help=(
            "Authoritative final metric to rank; may be repeated. It must match the "
            "run_final_exact rank contract."
        ),
    )
    runs.set_defaults(func=cmd_runs)

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="Rank exact checkpoint evaluation evidence.",
    )
    add_common_args(checkpoints, suppress_defaults=True)
    checkpoints.set_defaults(func=cmd_checkpoints)
    return parser


def cmd_runs(args: argparse.Namespace) -> int:
    if not args.database_url:
        raise RuntimeError("leaders require DATABASE_URL authoritative evidence access")
    conn = connect(str(args.database_url))
    try:
        facts = authoritative_run_facts(
            conn,
            goal_slug=args.goal,
            reward_shape=args.reward_shape,
        )
    finally:
        conn.close()
    requested = tuple(args.objective_key or ())
    grouped_facts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        dimensions = dict(fact["dimensions"])
        rank_metric = str(dimensions["rank_metric"])
        if requested and rank_metric not in requested:
            continue
        metrics = dict(fact["metrics"])
        if rank_metric not in metrics:
            raise RuntimeError(f"run_final_exact lacks its rank metric: {rank_metric}")
        grouped_facts[
            (str(fact["comparability_sha256"]), str(fact["cohort_manifest_sha256"]))
        ].append(fact)
    ranked_rows: list[dict[str, Any]] = []
    for cohort in grouped_facts.values():
        dimensions = dict(cohort[0]["dimensions"])
        metric = str(dimensions["rank_metric"])
        direction = str(dimensions["rank_direction"])
        values = [float(dict(fact["metrics"])[metric]) for fact in cohort]
        if len(values) < max(1, int(args.min_seeds)):
            continue
        steps = [
            int(dict(fact["metrics"])["global_step"])
            for fact in cohort
            if dict(fact["metrics"]).get("global_step") is not None
        ]
        ranked_rows.append(
            {
                "goal_slug": dimensions["goal_slug"],
                "recipe_slug": dimensions["recipe_slug"],
                "reward_shape": dimensions["reward_program_name"],
                "rank_metric": metric,
                "rank_direction": direction,
                "seeds": len(values),
                "worst_seed": min(values) if direction == "max" else max(values),
                "mean_seed": mean(values),
                "best_seed": max(values) if direction == "max" else min(values),
                "mean_steps": mean(steps) if steps else None,
                "scope_sha256s": [fact["scope_sha256"] for fact in cohort],
            }
        )
    ranked_rows.sort(
        key=lambda row: (
            (
                float(row["worst_seed"])
                if row["rank_direction"] == "max"
                else -float(row["worst_seed"])
            ),
            (
                float(row["mean_seed"])
                if row["rank_direction"] == "max"
                else -float(row["mean_seed"])
            ),
            (
                float(row["best_seed"])
                if row["rank_direction"] == "max"
                else -float(row["best_seed"])
            ),
            -(float(row["mean_steps"]) if row["mean_steps"] is not None else float("inf")),
        ),
        reverse=True,
    )
    ranked_rows = ranked_rows[: max(0, int(args.limit))]
    if args.json:
        print(json.dumps(json_safe(ranked_rows), indent=2, sort_keys=True))
    else:
        print(
            "goal_slug\trecipe_slug\treward_shape\trank_metric\trank_direction\t"
            "seeds\tworst_seed\tmean_seed\tbest_seed\tmean_steps"
        )
        for row in ranked_rows:
            mean_steps = (
                "" if row["mean_steps"] is None else f"{float(row['mean_steps']):.6g}"
            )
            print(
                f"{row['goal_slug']}\t{row['recipe_slug']}\t{row['reward_shape']}\t"
                f"{row['rank_metric']}\t{row['rank_direction']}\t{row['seeds']}\t"
                f"{float(row['worst_seed']):.6g}\t{float(row['mean_seed']):.6g}\t"
                f"{float(row['best_seed']):.6g}\t{mean_steps}"
            )
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    if not args.database_url:
        raise RuntimeError("leaders require DATABASE_URL authoritative evidence access")
    conn = connect(str(args.database_url))
    try:
        ranked = authoritative_checkpoint_evidence(conn, goal_slug=args.goal)[
            : max(0, int(args.limit))
        ]
    finally:
        conn.close()
    if args.json:
        print(json.dumps(json_safe(ranked), indent=2, sort_keys=True))
    else:
        print("scope_sha256\texecution_key\tcheckpoint_sha256\taccepted")
        for row in ranked:
            checkpoint = dict(row.get("checkpoint") or {})
            results = list(row.get("results") or [])
            accepted = bool(results) and all(bool(item.get("passed")) for item in results)
            print(
                f"{row.get('scope_sha256', '')}\t{row.get('execution_key', '')}\t"
                f"{checkpoint.get('sha256', '')}\t{str(accepted).lower()}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env_file(key_filter=lambda key: key == "DATABASE_URL")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
