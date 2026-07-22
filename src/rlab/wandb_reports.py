from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlab.metric_names import (
    EVAL_FULL_BY_START,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    EVAL_FULL_SUCCESS_RATE_MEAN,
    EVAL_FULL_SUCCESS_RATE_MIN,
    GLOBAL_STEP,
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_OBJECTIVE,
    LEADER_CHECKPOINT_PROGRESS_MAX,
    LEADER_CHECKPOINT_RETURN_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
    LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
    LEADER_CHECKPOINT_UPDATED_AT,
    TRAIN_A2C_EXPLAINED_VARIANCE,
    TRAIN_A2C_LEARNING_RATE,
    TRAIN_A2C_POLICY_ENTROPY,
    TRAIN_A2C_VALUE_LOSS,
    TRAIN_ARTIFACT_SAVE_SECONDS,
    TRAIN_ARTIFACT_UPLOAD_SECONDS,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
    TRAIN_PPO_APPROX_KL,
    TRAIN_PPO_CLIP_FRACTION,
    TRAIN_PPO_EXPLAINED_VARIANCE,
    TRAIN_PPO_LEARNING_RATE,
    TRAIN_PPO_POLICY_ENTROPY,
    TRAIN_PPO_VALUE_LOSS,
    TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
    TRAIN_THROUGHPUT_ENV_STEP_FPS,
    TRAIN_THROUGHPUT_LOOP_FPS,
    TRAIN_THROUGHPUT_ROLLOUT_FPS,
    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
)
from rlab.ranking import RankCriterion, objective_rank_strings, require_objective_rank
from rlab.recipe_documents import goal_contract_sha256, repo_git_commit
from rlab.reward_programs import select_goal_reward_shape
from rlab.wandb_utils import (
    load_wandb_env,
    resolve_wandb_project,
    wandb_entity_from_env,
)


REPORT_SCHEMA_VERSION = 1
MARIO_FAMILY = "SuperMarioBros-Nes-v0"
REPORT_ID_MARKER = "rlab-report-id"
REPORT_SOURCE_MARKER = "rlab-source-sha"
LEGACY_PORTFOLIO_TITLE = "rlab checkpoint leaderboards by goal"

PORTFOLIO_SECTIONS = frozenset({"goal_navigation", "checkpoint_leaders", "active_runs"})
GOAL_SECTIONS = frozenset(
    {
        "objective_summary",
        "checkpoint_leaderboard",
        "evaluation_progress",
        "evaluation_by_start",
        "training_progress",
        "active_runs",
        "failure_reasons",
        "algorithm_health",
        "throughput",
        "historical_contracts",
    }
)

LEADER_COLUMNS = [
    "run:name",
    "run:state",
    "run:group",
    "config:goal_slug.value",
    "config:goal_contract_sha256.value",
    "config:recipe_slug.value",
    "config:seed.value",
    f"summary:{LEADER_CHECKPOINT_OBJECTIVE}",
    f"summary:{LEADER_CHECKPOINT_RETURN_MEAN}",
    f"summary:{LEADER_CHECKPOINT_PROGRESS_MAX}",
    f"summary:{LEADER_CHECKPOINT_STEP}",
    f"summary:{LEADER_CHECKPOINT_ARTIFACT_REF}",
    f"summary:{LEADER_CHECKPOINT_UPDATED_AT}",
]
ACTIVE_COLUMNS = [
    "run:name",
    "run:state",
    "run:group",
    "config:goal_slug.value",
    "config:recipe_slug.value",
    "config:campaign_id.value",
    "config:batch_id.value",
    "config:seed.value",
    "config:algorithm_id.value",
    f"summary:{EVAL_ACCEPTANCE_PASS}",
    f"summary:{EVAL_ACCEPTANCE_EPISODES_COMPLETED}",
    f"summary:{EVAL_ACCEPTANCE_EPISODES_PLANNED}",
]
COLUMN_WIDTHS = {
    "run:name": 320,
    "run:group": 220,
    "config:goal_contract_sha256.value": 300,
    "config:recipe_slug.value": 260,
    f"summary:{LEADER_CHECKPOINT_ARTIFACT_REF}": 460,
    f"summary:{LEADER_CHECKPOINT_UPDATED_AT}": 220,
}


@dataclass(frozen=True)
class GoalReportSpec:
    identity: str
    title: str
    project: str
    goal_id: str
    goal_title: str
    goal_path: Path
    goal_contract_sha256: str
    effective_goal_contract_sha256: str
    reward_shape: str
    reward_shape_sha256: str
    starts: tuple[str, ...]
    rank: tuple[RankCriterion, ...]
    sections: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "goal"

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kind": self.kind,
            "title": self.title,
            "project": self.project,
            "goal_id": self.goal_id,
            "goal_path": str(self.goal_path),
            "goal_contract_sha256": self.goal_contract_sha256,
            "effective_goal_contract_sha256": self.effective_goal_contract_sha256,
            "reward_shape": self.reward_shape,
            "reward_shape_sha256": self.reward_shape_sha256,
            "starts": list(self.starts),
            "rank": list(objective_rank_strings(self.rank)),
            "sections": list(self.sections),
        }


@dataclass(frozen=True)
class PortfolioReportSpec:
    identity: str
    title: str
    project: str
    family: str
    sections: tuple[str, ...]
    goals: tuple[GoalReportSpec, ...]

    @property
    def kind(self) -> str:
        return "portfolio"

    def to_json(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kind": self.kind,
            "title": self.title,
            "project": self.project,
            "family": self.family,
            "sections": list(self.sections),
            "goals": [goal.identity for goal in self.goals],
        }


ReportSpec = GoalReportSpec | PortfolioReportSpec


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(document: Mapping[str, Any], allowed: set[str] | frozenset[str], label: str):
    unknown = sorted(str(key) for key in document if key not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _non_empty_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _sections(value: Any, *, allowed: frozenset[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list of semantic section ids")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{label} must contain at least one non-empty section id")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate section ids")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown section id(s): {', '.join(unknown)}")
    return result


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is not valid YAML: {exc}") from exc
    return _mapping(value, label=label)


def _goal_starts(document: Mapping[str, Any]) -> tuple[str, ...]:
    objective = _mapping(document.get("objective"), label="goal objective")
    if "states" in objective:
        return tuple(str(value) for value in objective["states"])
    train = _mapping(document.get("train"), label="goal train")
    environment = _mapping(train.get("environment"), label="goal train.environment")
    env_config = _mapping(
        environment.get("env_config", environment), label="goal train.environment.env_config"
    )
    if "states" in env_config:
        return tuple(str(value) for value in env_config["states"])
    if "state" in env_config:
        return (str(env_config["state"]),)
    return ()


def _goal_project(document: Mapping[str, Any]) -> str:
    train = _mapping(document.get("train"), label="goal train")
    environment = _mapping(train.get("environment"), label="goal train.environment")
    env_config = _mapping(
        environment.get("env_config", environment), label="goal train.environment.env_config"
    )
    return resolve_wandb_project(
        None,
        str(env_config.get("game") or ""),
        env_provider=environment.get("env_provider") or env_config.get("env_provider"),
    )


def _format_goal_title(template: str, *, goal_id: str, goal_title: str) -> str:
    try:
        return template.format(goal_id=goal_id, goal_title=goal_title)
    except KeyError as exc:
        raise ValueError(
            f"goal report title uses unknown template variable: {exc.args[0]}"
        ) from exc


def compile_report_specs(
    repo_root: Path | str = Path("."), *, goal: str | None = None
) -> tuple[ReportSpec, ...]:
    from rlab.config_validation import load_goal_contract

    repo_root = Path(repo_root).resolve()
    family_root = repo_root / "experiments" / "goals" / MARIO_FAMILY
    manifest_path = family_root / "_reports.yaml"
    manifest = _load_yaml(manifest_path, label=f"report manifest {manifest_path}")
    _reject_unknown(manifest, {"schema_version", "portfolio", "goal"}, str(manifest_path))
    if manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}.schema_version must equal {REPORT_SCHEMA_VERSION}")
    portfolio_config = _mapping(manifest.get("portfolio"), label=f"{manifest_path}.portfolio")
    goal_config = _mapping(manifest.get("goal"), label=f"{manifest_path}.goal")
    _reject_unknown(portfolio_config, {"title", "sections"}, f"{manifest_path}.portfolio")
    _reject_unknown(goal_config, {"enabled", "title", "sections"}, f"{manifest_path}.goal")
    portfolio_title = _non_empty_text(
        portfolio_config.get("title"), label=f"{manifest_path}.portfolio.title"
    )
    portfolio_sections = _sections(
        portfolio_config.get("sections"),
        allowed=PORTFOLIO_SECTIONS,
        label=f"{manifest_path}.portfolio.sections",
    )
    if not isinstance(goal_config.get("enabled"), bool):
        raise ValueError(f"{manifest_path}.goal.enabled must be a boolean")
    goal_title_template = _non_empty_text(
        goal_config.get("title"), label=f"{manifest_path}.goal.title"
    )
    goal_sections = _sections(
        goal_config.get("sections"),
        allowed=GOAL_SECTIONS,
        label=f"{manifest_path}.goal.sections",
    )

    goal_paths = sorted(family_root.glob("*/_goal.yaml"))
    goal_dirs = {path.parent.resolve() for path in goal_paths}
    for override_path in sorted(family_root.rglob("_report.yaml")):
        if override_path.parent.resolve() not in goal_dirs:
            raise ValueError(f"report override {override_path} has no sibling _goal.yaml")

    compiled_goals: list[GoalReportSpec] = []
    for goal_path in goal_paths:
        document = load_goal_contract(goal_path, repo_root)
        goal_id = _non_empty_text(document.get("goal_id"), label=f"{goal_path}.goal_id")
        if goal is not None and goal_id != goal:
            continue
        enabled = bool(goal_config["enabled"])
        title_template = goal_title_template
        sections = goal_sections
        override_path = goal_path.parent / "_report.yaml"
        if override_path.is_file():
            override = _load_yaml(override_path, label=f"report override {override_path}")
            _reject_unknown(override, {"enabled", "title", "sections"}, str(override_path))
            if "enabled" in override:
                if not isinstance(override["enabled"], bool):
                    raise ValueError(f"{override_path}.enabled must be a boolean")
                enabled = override["enabled"]
            if "title" in override:
                title_template = _non_empty_text(override["title"], label=f"{override_path}.title")
            if "sections" in override:
                sections = _sections(
                    override["sections"],
                    allowed=GOAL_SECTIONS,
                    label=f"{override_path}.sections",
                )
        if not enabled:
            continue
        goal_title = _non_empty_text(document.get("title"), label=f"{goal_path}.title")
        selected_reward = select_goal_reward_shape(
            document,
            None,
            label=f"goal report {goal_path}",
        )
        effective_goal = selected_reward.goal if selected_reward is not None else document
        compiled_goals.append(
            GoalReportSpec(
                identity=f"{MARIO_FAMILY}/goal/{goal_id}",
                title=_format_goal_title(title_template, goal_id=goal_id, goal_title=goal_title),
                project=_goal_project(document),
                goal_id=goal_id,
                goal_title=goal_title,
                goal_path=goal_path.relative_to(repo_root),
                goal_contract_sha256=goal_contract_sha256(document),
                effective_goal_contract_sha256=goal_contract_sha256(effective_goal),
                reward_shape=selected_reward.key if selected_reward is not None else "",
                reward_shape_sha256=(
                    selected_reward.semantic_sha256 if selected_reward is not None else ""
                ),
                starts=_goal_starts(document),
                rank=require_objective_rank(document["objective"]["rank"]),
                sections=sections,
            )
        )
    if goal is not None and not compiled_goals:
        raise ValueError(f"no enabled Mario goal report found for {goal!r}")
    projects = {compiled.project for compiled in compiled_goals}
    if len(projects) != 1:
        raise ValueError(
            f"Mario goal reports must resolve to one W&B project, got {sorted(projects)}"
        )
    identities = [compiled.identity for compiled in compiled_goals]
    if len(set(identities)) != len(identities):
        raise ValueError("compiled report identities must be unique")
    portfolio = PortfolioReportSpec(
        identity=f"{MARIO_FAMILY}/portfolio",
        title=portfolio_title,
        project=next(iter(projects)),
        family=MARIO_FAMILY,
        sections=portfolio_sections,
        goals=tuple(compiled_goals),
    )
    return (*compiled_goals, portfolio)


def validate_report_declarations(repo_root: Path | str = Path(".")) -> int:
    specs = compile_report_specs(repo_root)
    return len(specs)


def _current_filter_text(goal: GoalReportSpec) -> str:
    text = (
        f"Config('goal_slug') = '{goal.goal_id}' and "
        f"Config('goal_contract_sha256') = '{goal.goal_contract_sha256}'"
    )
    if goal.reward_shape:
        text += (
            f" and Config('effective_goal_contract_sha256') = "
            f"'{goal.effective_goal_contract_sha256}'"
            f" and Config('reward_shape_sha256') = '{goal.reward_shape_sha256}'"
        )
    return text


def _active_filter_text(goal: GoalReportSpec) -> str:
    return _current_filter_text(goal) + " and Metric('state') in ['running']"


def _historical_filter_text(goal: GoalReportSpec) -> str:
    return (
        f"Config('goal_slug') = '{goal.goal_id}' and "
        f"Config('goal_contract_sha256') != '{goal.goal_contract_sha256}'"
    )


def _leader_filter_text(goal: GoalReportSpec) -> str:
    return (
        _current_filter_text(goal) + f" and SummaryMetric('{LEADER_CHECKPOINT_ARTIFACT_REF}') != ''"
    )


def _leader_order_spec(criteria: Sequence[RankCriterion]) -> list[tuple[str, bool]]:
    mapped = {
        EVAL_FULL_SUCCESS_RATE_MIN: LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
        EVAL_FULL_SUCCESS_RATE_MEAN: LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
        EVAL_FULL_EPISODE_RETURN_MEAN: LEADER_CHECKPOINT_RETURN_MEAN,
        LEADER_CHECKPOINT_STEP: LEADER_CHECKPOINT_STEP,
    }
    result: list[tuple[str, bool]] = []
    for index, criterion in enumerate(criteria):
        metric = LEADER_CHECKPOINT_OBJECTIVE if index == 0 else mapped.get(criterion.metric)
        if metric is None:
            raise ValueError(
                f"objective.rank criterion cannot be represented in W&B: {criterion.metric}"
            )
        result.append((metric, criterion.direction == "min"))
    return result


def _description(identity: str, source_sha: str) -> str:
    return (
        "Generated from checked-in rlab report declarations. Manual edits are replaced on sync.\n\n"
        f"<!-- {REPORT_ID_MARKER}:{identity} -->\n"
        f"<!-- {REPORT_SOURCE_MARKER}:{source_sha or 'unknown'} -->"
    )


def extract_report_identity(description: object) -> str | None:
    match = re.search(
        rf"<!--\s*{re.escape(REPORT_ID_MARKER)}:([^>]+?)\s*-->", str(description or "")
    )
    return match.group(1).strip() if match else None


def _canonical_report_url(url: str) -> str:
    """Match the canonical URL form returned by the public W&B API."""
    prefix = "/reports/"
    if prefix not in url:
        return url
    base, slug = url.split(prefix, 1)
    return f"{base}{prefix}{slug.replace(':', '').rstrip('=')}"


def _leader_runset(wr, goal: GoalReportSpec, *, entity: str):
    return wr.Runset(
        entity=entity,
        project=goal.project,
        name=f"{goal.goal_id} current-contract checkpoint leaders",
        filters=_leader_filter_text(goal),
        order=[
            wr.OrderBy(wr.SummaryMetric(metric), ascending=ascending)
            for metric, ascending in _leader_order_spec(goal.rank)
        ],
        pinned_columns=LEADER_COLUMNS,
        visible_columns=[
            *LEADER_COLUMNS,
            "tags:__ALL__",
            f"summary:{LEADER_CHECKPOINT_EVAL_SOURCE}",
        ],
        column_order=LEADER_COLUMNS,
        column_widths=COLUMN_WIDTHS,
        lock_columns=True,
    )


def _run_table(wr, goal: GoalReportSpec, *, entity: str, active: bool):
    return wr.Runset(
        entity=entity,
        project=goal.project,
        name=f"{goal.goal_id} {'active' if active else 'historical'} runs",
        filters=_active_filter_text(goal) if active else _historical_filter_text(goal),
        pinned_columns=ACTIVE_COLUMNS if active else LEADER_COLUMNS,
        visible_columns=ACTIVE_COLUMNS if active else [*LEADER_COLUMNS, "tags:__ALL__"],
        column_order=ACTIVE_COLUMNS if active else LEADER_COLUMNS,
        column_widths=COLUMN_WIDTHS,
        lock_columns=True,
    )


def _current_runset(wr, goal: GoalReportSpec, *, entity: str):
    return wr.Runset(
        entity=entity,
        project=goal.project,
        name=f"{goal.goal_id} current contract",
        filters=_current_filter_text(goal),
    )


def _line(wr, *, title: str, x: str, y: Sequence[str], w: int = 12, h: int = 8):
    return wr.LinePlot(
        title=title,
        x=x,
        y=list(y),
        layout=wr.Layout(w=w, h=h),
        smoothing_type="none",
    )


def _goal_section_blocks(wr, section: str, goal: GoalReportSpec, *, entity: str):
    runset = _current_runset(wr, goal, entity=entity)
    if section == "objective_summary":
        rank = " → ".join(objective_rank_strings(goal.rank))
        starts = ", ".join(goal.starts) or "provider-defined"
        return [
            wr.H2("Goal contract and objective"),
            wr.MarkdownBlock(
                f"**Goal:** `{goal.goal_id}`  \n"
                f"**Contract:** `{goal.goal_contract_sha256}`  \n"
                f"**Reward shape:** `{goal.reward_shape or 'inline/legacy'}`  \n"
                f"**Starts:** {starts}  \n"
                f"**Checkpoint ranking:** {rank}"
            ),
        ]
    if section == "checkpoint_leaderboard":
        return [
            wr.H2("Evaluated checkpoint leaderboard"),
            wr.PanelGrid(
                runsets=[_leader_runset(wr, goal, entity=entity)],
                hide_run_sets=False,
                panels=[],
            ),
        ]
    if section == "evaluation_progress":
        return [
            wr.H2("Full-evaluation progress"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _line(
                        wr,
                        title="Acceptance result",
                        x=GLOBAL_STEP,
                        y=[EVAL_ACCEPTANCE_PASS],
                    ),
                    _line(
                        wr,
                        title="Acceptance episode progress",
                        x=GLOBAL_STEP,
                        y=[
                            EVAL_ACCEPTANCE_EPISODES_COMPLETED,
                            EVAL_ACCEPTANCE_EPISODES_PLANNED,
                        ],
                    ),
                    _line(
                        wr,
                        title=EVAL_FULL_EPISODE_RETURN_MEAN,
                        x=GLOBAL_STEP,
                        y=[EVAL_FULL_EPISODE_RETURN_MEAN],
                    ),
                ],
            ),
        ]
    if section == "evaluation_by_start":
        panels = [
            wr.WeavePanelSummaryTable(
                table_name=EVAL_FULL_BY_START,
                layout=wr.Layout(w=24, h=12),
            )
        ]
        return [
            wr.H2("Full evaluation by start"),
            wr.PanelGrid(runsets=[runset], panels=panels),
        ]
    if section == "training_progress":
        panels = [
            _line(
                wr,
                title="Cumulative training success",
                x=GLOBAL_STEP,
                y=[
                    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN,
                    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
                ],
            ),
            _line(
                wr,
                title="Window-100 training success",
                x=GLOBAL_STEP,
                y=[
                    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
                    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN,
                ],
            ),
            _line(
                wr,
                title=TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
                x=GLOBAL_STEP,
                y=[TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE],
            ),
        ]
        if len(goal.starts) <= 4:
            for start in goal.starts:
                metric = f"train/outcome/success/from/{start}/rate/window_100"
                panels.append(_line(wr, title=metric, x=GLOBAL_STEP, y=[metric], w=12, h=7))
        return [
            wr.H2("Training progress (diagnostic only)"),
            wr.MarkdownBlock(
                "Training curves help diagnose learning. Only full evaluation establishes ranking or acceptance."
            ),
            wr.PanelGrid(runsets=[runset], panels=panels),
        ]
    if section == "active_runs":
        return [
            wr.H2("Active experiments"),
            wr.PanelGrid(
                runsets=[_run_table(wr, goal, entity=entity, active=True)],
                hide_run_sets=False,
                panels=[],
            ),
        ]
    if section == "failure_reasons":
        return [
            wr.H2("Failure reasons"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    wr.LinePlot(
                        title="Training failure-reason rates",
                        x=GLOBAL_STEP,
                        y=[],
                        metric_regex=r"train/outcome/reason/.*/rate/window_100",
                        layout=wr.Layout(w=12, h=8),
                    ),
                ],
            ),
        ]
    if section == "algorithm_health":
        return [
            wr.H2("Actor-critic health"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _line(
                        wr,
                        title="Explained variance",
                        x=GLOBAL_STEP,
                        y=[TRAIN_PPO_EXPLAINED_VARIANCE, TRAIN_A2C_EXPLAINED_VARIANCE],
                    ),
                    _line(
                        wr,
                        title="Value loss",
                        x=GLOBAL_STEP,
                        y=[TRAIN_PPO_VALUE_LOSS, TRAIN_A2C_VALUE_LOSS],
                    ),
                    _line(
                        wr,
                        title="Learning rate",
                        x=GLOBAL_STEP,
                        y=[TRAIN_PPO_LEARNING_RATE, TRAIN_A2C_LEARNING_RATE],
                    ),
                    _line(
                        wr,
                        title="Policy entropy",
                        x=GLOBAL_STEP,
                        y=[TRAIN_PPO_POLICY_ENTROPY, TRAIN_A2C_POLICY_ENTROPY],
                    ),
                    _line(
                        wr,
                        title="PPO update constraints",
                        x=GLOBAL_STEP,
                        y=[TRAIN_PPO_APPROX_KL, TRAIN_PPO_CLIP_FRACTION],
                    ),
                ],
            ),
        ]
    if section == "throughput":
        return [
            wr.H2("Throughput and artifact timing"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _line(
                        wr,
                        title="Training throughput",
                        x=GLOBAL_STEP,
                        y=[
                            TRAIN_THROUGHPUT_LOOP_FPS,
                            TRAIN_THROUGHPUT_ROLLOUT_FPS,
                            TRAIN_THROUGHPUT_ENV_STEP_FPS,
                        ],
                    ),
                    _line(
                        wr,
                        title="Between-rollout and overhead time",
                        x=GLOBAL_STEP,
                        y=[
                            TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
                            TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
                        ],
                    ),
                    _line(
                        wr,
                        title="Artifact timing",
                        x=GLOBAL_STEP,
                        y=[TRAIN_ARTIFACT_SAVE_SECONDS, TRAIN_ARTIFACT_UPLOAD_SECONDS],
                    ),
                ],
            ),
        ]
    if section == "historical_contracts":
        return [
            wr.H2("Historical and incompatible goal contracts"),
            wr.MarkdownBlock(
                "These runs share the goal id but not the current composed goal contract. They are excluded from current leaderboards."
            ),
            wr.PanelGrid(
                runsets=[_run_table(wr, goal, entity=entity, active=False)],
                hide_run_sets=False,
                panels=[],
            ),
        ]
    raise AssertionError(f"unhandled goal report section: {section}")


def _portfolio_section_blocks(
    wr,
    section: str,
    portfolio: PortfolioReportSpec,
    *,
    entity: str,
    goal_urls: Mapping[str, str],
):
    if section == "goal_navigation":
        links = [
            f"- [{goal.goal_id}]({goal_urls[goal.identity]}) — {goal.goal_title}"
            for goal in portfolio.goals
        ]
        return [
            wr.H2("Goal reports"),
            wr.MarkdownBlock("\n".join(links)),
        ]
    if section == "checkpoint_leaders":
        blocks: list[Any] = [
            wr.H2("Current-contract checkpoint leaders by goal"),
        ]
        for goal in portfolio.goals:
            blocks.extend(
                [
                    wr.H3(goal.goal_id),
                    wr.PanelGrid(
                        runsets=[_leader_runset(wr, goal, entity=entity)],
                        hide_run_sets=False,
                        panels=[],
                    ),
                ]
            )
        return blocks
    if section == "active_runs":
        filters = " or ".join(f"({_active_filter_text(goal)})" for goal in portfolio.goals)
        runset = wr.Runset(
            entity=entity,
            project=portfolio.project,
            name="Active current-contract Mario runs grouped by goal",
            filters=filters,
            groupby=["config.goal_slug"],
            pinned_columns=ACTIVE_COLUMNS,
            visible_columns=ACTIVE_COLUMNS,
            column_order=ACTIVE_COLUMNS,
            column_widths=COLUMN_WIDTHS,
            lock_columns=True,
        )
        return [
            wr.H2("Active runs by goal"),
            wr.PanelGrid(runsets=[runset], hide_run_sets=False, panels=[]),
        ]
    raise AssertionError(f"unhandled portfolio report section: {section}")


def build_wandb_report(
    spec: ReportSpec,
    *,
    entity: str,
    source_sha: str,
    goal_urls: Mapping[str, str] | None = None,
):
    import wandb_workspaces.reports.v2 as wr

    blocks: list[Any] = [wr.TableOfContents()]
    if isinstance(spec, GoalReportSpec):
        for section in spec.sections:
            blocks.extend(_goal_section_blocks(wr, section, spec, entity=entity))
    else:
        blocks.extend(
            [
                wr.CalloutBlock(
                    "Specialist success across individual levels does not prove that one generalist policy solves every level. The all-level policy must be evaluated under its own multi-start goal contract."
                )
            ]
        )
        urls = dict(goal_urls or {})
        missing = sorted(goal.identity for goal in spec.goals if goal.identity not in urls)
        if missing:
            raise ValueError(f"portfolio is missing goal report URLs: {missing}")
        for section in spec.sections:
            blocks.extend(
                _portfolio_section_blocks(
                    wr,
                    section,
                    spec,
                    entity=entity,
                    goal_urls=urls,
                )
            )
    return wr.Report(
        entity=entity,
        project=spec.project,
        title=spec.title,
        description=_description(spec.identity, source_sha),
        width="fluid",
        blocks=blocks,
    )


def _existing_reports(api, *, entity: str, project: str) -> list[Any]:
    return list(api.reports(f"{entity}/{project}", per_page=100))


def _report_index(reports: Sequence[Any]) -> dict[str, list[Any]]:
    index: dict[str, list[Any]] = {}
    for report in reports:
        identity = extract_report_identity(getattr(report, "description", ""))
        if identity:
            index.setdefault(identity, []).append(report)
    return index


def _preflight_existing(
    specs: Sequence[ReportSpec], reports: Sequence[Any]
) -> dict[str, Any | None]:
    index = _report_index(reports)
    duplicates = {identity: rows for identity, rows in index.items() if len(rows) > 1}
    if duplicates:
        raise ValueError(
            "duplicate generated W&B report identities: " + ", ".join(sorted(duplicates))
        )
    existing: dict[str, Any | None] = {
        spec.identity: (index.get(spec.identity) or [None])[0] for spec in specs
    }
    portfolio = next(spec for spec in specs if isinstance(spec, PortfolioReportSpec))
    if existing[portfolio.identity] is None:
        legacy = [
            report
            for report in reports
            if getattr(report, "display_name", None) == LEGACY_PORTFOLIO_TITLE
            and extract_report_identity(getattr(report, "description", "")) is None
        ]
        if len(legacy) > 1:
            raise ValueError(
                "multiple legacy checkpoint leaderboard reports require manual cleanup"
            )
        if legacy:
            existing[portfolio.identity] = legacy[0]
    return existing


def _replace_and_save(desired, existing):
    if existing is None:
        desired.save()
        return desired
    from wandb_workspaces.reports.v2 import Report

    current = Report.from_url(existing.url)
    current.entity = desired.entity
    current.project = desired.project
    current.title = desired.title
    current.description = desired.description
    current.width = desired.width
    current.blocks = desired.blocks
    current.save()
    return current


def sync_reports(
    specs: Sequence[ReportSpec],
    *,
    api=None,
    entity: str | None = None,
    source_sha: str | None = None,
) -> list[dict[str, str]]:
    import wandb

    load_wandb_env()
    entity = entity or wandb_entity_from_env()
    source_sha = source_sha or repo_git_commit() or "unknown"
    api = api or wandb.Api(timeout=30)
    projects = {spec.project for spec in specs}
    if len(projects) != 1:
        raise ValueError("one report sync may target only one W&B project")
    goal_specs = [spec for spec in specs if isinstance(spec, GoalReportSpec)]
    placeholder_urls = {
        goal.identity: f"https://wandb.ai/{entity}/{goal.project}/reports/pending-{goal.goal_id}"
        for goal in goal_specs
    }
    for spec in specs:
        report = build_wandb_report(
            spec,
            entity=entity,
            source_sha=source_sha,
            goal_urls=placeholder_urls if isinstance(spec, PortfolioReportSpec) else None,
        )
        report._to_model()
    reports = _existing_reports(api, entity=entity, project=next(iter(projects)))
    existing = _preflight_existing(specs, reports)
    goals = goal_specs
    portfolio = next(spec for spec in specs if isinstance(spec, PortfolioReportSpec))
    results: list[dict[str, str]] = []
    goal_urls: dict[str, str] = {}
    for goal in goals:
        desired = build_wandb_report(goal, entity=entity, source_sha=source_sha)
        saved = _replace_and_save(desired, existing[goal.identity])
        saved_url = _canonical_report_url(saved.url)
        goal_urls[goal.identity] = saved_url
        results.append({"identity": goal.identity, "url": saved_url})
    desired_portfolio = build_wandb_report(
        portfolio,
        entity=entity,
        source_sha=source_sha,
        goal_urls=goal_urls,
    )
    saved_portfolio = _replace_and_save(desired_portfolio, existing[portfolio.identity])
    results.append(
        {"identity": portfolio.identity, "url": _canonical_report_url(saved_portfolio.url)}
    )
    return results


def _normalized_report(value: Any) -> Any:
    # Compare the canonical workspace model rather than the public dataclasses.
    # W&B parses filters and metric names into structured expressions on readback,
    # assigns fresh internal ids, and may reflow layouts. Those representation-only
    # changes are not source drift; titles, text, runsets, filters, metrics, and
    # section order remain part of the canonical payload below.
    if hasattr(value, "_to_model"):
        # Runset serialization normally resolves a W&B-internal project id over
        # GraphQL. The report already owns the entity/project routing, so omit the
        # redundant runset lookup while producing a secret-free semantic digest.
        value = copy.deepcopy(value)
        for block in getattr(value, "blocks", ()):
            for runset in getattr(block, "runsets", ()):
                object.__setattr__(runset, "entity", None)
                object.__setattr__(runset, "project", None)
        value = value._to_model().model_dump()
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        ignored = {"created_at", "id", "layout", "updated_at", "width"}
        normalized = {
            str(key): _normalized_report(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if not str(key).startswith("_") and key not in ignored
        }
        for grouping in normalized.get("grouping", ()):
            if grouping.get("section") == "config" and grouping.get("name", "").endswith(".value"):
                grouping["section"] = grouping["name"].removesuffix(".value")
                grouping["name"] = "value"
        return normalized
    if isinstance(value, tuple | list):
        return [_normalized_report(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"https://wandb\.ai/[^\s)]+/reports/[^\s)]+",
            lambda match: _canonical_report_url(match.group(0)),
            value,
        )
    return value


def _structure_sha256(report: Any) -> str:
    payload = json.dumps(
        _normalized_report(report), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_reports(
    specs: Sequence[ReportSpec],
    *,
    api=None,
    entity: str | None = None,
    source_sha: str | None = None,
    report_loader=None,
    include_orphans: bool = True,
) -> dict[str, Any]:
    import wandb
    from wandb_workspaces.reports.v2 import Report

    load_wandb_env()
    entity = entity or wandb_entity_from_env()
    source_sha = source_sha or repo_git_commit() or "unknown"
    api = api or wandb.Api(timeout=30)
    report_loader = report_loader or Report.from_url
    project = next(iter({spec.project for spec in specs}))
    reports = _existing_reports(api, entity=entity, project=project)
    index = _report_index(reports)
    desired_ids = {spec.identity for spec in specs}
    issues: list[dict[str, str]] = []
    for identity, rows in sorted(index.items()):
        if len(rows) > 1:
            issues.append({"identity": identity, "issue": "duplicate"})
        elif (
            include_orphans
            and identity.startswith(f"{MARIO_FAMILY}/")
            and identity not in desired_ids
        ):
            issues.append({"identity": identity, "issue": "orphan"})
    existing_by_id = {identity: rows[0] for identity, rows in index.items() if len(rows) == 1}
    goals = [spec for spec in specs if isinstance(spec, GoalReportSpec)]
    goal_urls = {
        goal.identity: existing_by_id[goal.identity].url
        for goal in goals
        if goal.identity in existing_by_id
    }
    checked: list[dict[str, str]] = []
    for spec in specs:
        existing = existing_by_id.get(spec.identity)
        if existing is None:
            issues.append({"identity": spec.identity, "issue": "missing"})
            continue
        try:
            desired = build_wandb_report(
                spec,
                entity=entity,
                source_sha=source_sha,
                goal_urls=goal_urls if isinstance(spec, PortfolioReportSpec) else None,
            )
        except ValueError as exc:
            issues.append({"identity": spec.identity, "issue": str(exc)})
            continue
        actual = report_loader(existing.url)
        desired_sha = _structure_sha256(desired)
        actual_sha = _structure_sha256(actual)
        checked.append(
            {"identity": spec.identity, "url": existing.url, "structure_sha256": actual_sha}
        )
        if actual_sha != desired_sha:
            issues.append({"identity": spec.identity, "issue": "content_drift"})
    return {"ok": not issues, "checked": checked, "issues": issues}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab reports",
        description="Plan, synchronize, and verify declarative W&B reports.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "sync", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--goal", help="Limit work to one Mario goal id.")
        command.add_argument("--repo-root", type=Path, default=Path("."))
        if name in {"plan", "verify"}:
            command.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = compile_report_specs(args.repo_root, goal=args.goal)
    if args.command == "plan":
        payload = {"reports": [spec.to_json() for spec in specs]}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            for spec in specs:
                print(f"{spec.kind}\t{spec.identity}\t{spec.project}\t{','.join(spec.sections)}")
        return 0
    if args.command == "sync":
        for result in sync_reports(specs):
            print(f"{result['identity']}\t{result['url']}")
        return 0
    verification = verify_reports(specs, include_orphans=args.goal is None)
    if args.json:
        print(json.dumps(verification, sort_keys=True))
    else:
        for row in verification["checked"]:
            print(f"ok\t{row['identity']}\t{row['url']}")
        for row in verification["issues"]:
            print(f"error\t{row['identity']}\t{row['issue']}")
    return 0 if verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
