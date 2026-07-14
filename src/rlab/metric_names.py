from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


METRICS_SCHEMA_VERSION = 2
GLOBAL_STEP = "global_step"

TRAIN_EPISODE_RETURN_SHAPED_MEAN = "train/episode/return/shaped/mean"
TRAIN_EPISODE_LENGTH_MEAN = "train/episode/length/mean"
TRAIN_EPISODE_COUNT = "train/episode/count"

TRAIN_OUTCOME_TERMINAL_COUNT = "train/outcome/terminal/count"
TRAIN_OUTCOME_SUCCESS_ROOT = "train/outcome/success"
TRAIN_OUTCOME_SUCCESS_RATE_CURRENT_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/rate/current/min"
TRAIN_OUTCOME_SUCCESS_RATE_CURRENT_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/rate/current/mean"
TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/rate/window_100/min"
TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/rate/window_100/mean"
TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/start_coverage/rate"
)

TRAIN_REWARD_ROOT = "train/reward"

TRAIN_ALGORITHM_PPO_ROOT = "train/algorithm/ppo"
TRAIN_PPO_APPROX_KL = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/approx_kl"
TRAIN_PPO_CLIP_FRACTION = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/clip_fraction"
TRAIN_PPO_EXPLAINED_VARIANCE = f"{TRAIN_ALGORITHM_PPO_ROOT}/value/explained_variance"
TRAIN_PPO_POLICY_GRADIENT_LOSS = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/policy_gradient_loss"
TRAIN_PPO_VALUE_LOSS = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/value_loss"
TRAIN_PPO_LEARNING_RATE = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/learning_rate"
TRAIN_PPO_POLICY_ENTROPY = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/entropy"
TRAIN_PPO_POLICY_DISTRIBUTION_STD = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/distribution_std"
TRAIN_PPO_POLICY_DOMINANT_ACTION_RATE = (
    f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/dominant_action_rate"
)
TRAIN_PPO_POLICY_ACTION_HIST = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/action_hist"
TRAIN_PPO_VALUE_PREDICTION_ROOT = f"{TRAIN_ALGORITHM_PPO_ROOT}/rollout/value_prediction"
TRAIN_PPO_VALUE_PREDICTION_HIST = f"{TRAIN_PPO_VALUE_PREDICTION_ROOT}/hist"
TRAIN_PPO_ADVANTAGE_ROOT = f"{TRAIN_ALGORITHM_PPO_ROOT}/rollout/advantage"
TRAIN_PPO_ADVANTAGE_HIST = f"{TRAIN_PPO_ADVANTAGE_ROOT}/hist"
TRAIN_PPO_ENTROPY_COEFFICIENT = (
    f"{TRAIN_ALGORITHM_PPO_ROOT}/hyperparameter/entropy_coefficient"
)
TRAIN_PPO_ADVANTAGE_NORMALIZATION_MODE = (
    f"{TRAIN_ALGORITHM_PPO_ROOT}/advantage/normalization_mode"
)

TRAIN_THROUGHPUT_ROOT = "train/throughput"
TRAIN_THROUGHPUT_LOOP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/loop_fps"
TRAIN_THROUGHPUT_ROLLOUT_FPS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_fps"
TRAIN_THROUGHPUT_ENV_STEP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_fps"
TRAIN_THROUGHPUT_LOOP_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/loop_seconds"
TRAIN_THROUGHPUT_ROLLOUT_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_seconds"
TRAIN_THROUGHPUT_ENV_STEP_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_seconds"
TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS = (
    f"{TRAIN_THROUGHPUT_ROOT}/rollout_overhead_seconds"
)
TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS = (
    f"{TRAIN_THROUGHPUT_ROOT}/between_rollouts_seconds"
)

TRAIN_ARTIFACT_SAVE_SECONDS = "train/artifact/save/seconds"
TRAIN_ARTIFACT_UPLOAD_SECONDS = "train/artifact/upload/seconds"

EVAL_ROOT = "eval"
EVAL_PROTOCOLS = ("screen", "confirm", "full")
EVAL_FULL_ROOT = f"{EVAL_ROOT}/full"
EVAL_FULL_EPISODE_RETURN_MEAN = f"{EVAL_FULL_ROOT}/episode/return/mean"
EVAL_FULL_EPISODE_RETURN_STD = f"{EVAL_FULL_ROOT}/episode/return/std"
EVAL_FULL_EPISODE_RETURN_MEDIAN = f"{EVAL_FULL_ROOT}/episode/return/median"
EVAL_FULL_EPISODE_RETURN_BEST = f"{EVAL_FULL_ROOT}/episode/return/best"
EVAL_FULL_EPISODE_LENGTH_MEAN = f"{EVAL_FULL_ROOT}/episode/length/mean"
EVAL_FULL_EPISODE_COUNT = f"{EVAL_FULL_ROOT}/episode/count"
EVAL_FULL_PROGRESS_X_MAX = f"{EVAL_FULL_ROOT}/progress/x/max"
EVAL_FULL_SUCCESS_RATE_MIN = f"{EVAL_FULL_ROOT}/outcome/success/rate/min"
EVAL_FULL_SUCCESS_RATE_MEAN = f"{EVAL_FULL_ROOT}/outcome/success/rate/mean"
EVAL_FULL_BY_START = f"{EVAL_FULL_ROOT}/by_start"
EVAL_FULL_CHECKPOINT_STEP = f"{EVAL_FULL_ROOT}/checkpoint/step"
EVAL_FULL_CHECKPOINT_ARTIFACT = f"{EVAL_FULL_ROOT}/checkpoint/artifact"
EVAL_FULL_DURATION_SECONDS = f"{EVAL_FULL_ROOT}/duration/seconds"
EVAL_FULL_SOURCE = f"{EVAL_FULL_ROOT}/source"

CHECKPOINT_EVAL_CANDIDATE_PASS = "eval/confirm/candidate/pass"
CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX = "eval/confirm/candidate/stage_index"
CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP = "eval/confirm/candidate/checkpoint_step"
CHECKPOINT_EVAL_CANDIDATE_EPISODES = "eval/confirm/candidate/episodes"

LEADER_CHECKPOINT_SUCCESS_RATE_MIN = "leader/checkpoint/success_rate_min"
LEADER_CHECKPOINT_SUCCESS_RATE_MEAN = "leader/checkpoint/success_rate_mean"
LEADER_CHECKPOINT_OBJECTIVE = "leader/checkpoint/objective"
LEADER_CHECKPOINT_OBJECTIVE_NAME = "leader/checkpoint/objective_name"
LEADER_CHECKPOINT_RETURN_MEAN = "leader/checkpoint/return_mean"
LEADER_CHECKPOINT_BEST_RETURN = "leader/checkpoint/best_return"
LEADER_CHECKPOINT_RANK = "leader/checkpoint/rank"
LEADER_CHECKPOINT_RANK_VALUES = "leader/checkpoint/rank_values"
LEADER_CHECKPOINT_PROGRESS_MAX = "leader/checkpoint/progress_max"
LEADER_CHECKPOINT_STEP = "leader/checkpoint/step"
LEADER_CHECKPOINT_STEPS_TO_GOAL = "leader/checkpoint/steps_to_goal"
LEADER_CHECKPOINT_ARTIFACT_REF = "leader/checkpoint/artifact_ref"
LEADER_CHECKPOINT_LOCAL_PATH = "leader/checkpoint/local_path"
LEADER_CHECKPOINT_EVAL_SOURCE = "leader/checkpoint/eval_source"
LEADER_CHECKPOINT_UPDATED_AT = "leader/checkpoint/updated_at"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    unit: str
    cadence: str
    storage: str


def _definition(
    name: str,
    description: str,
    unit: str = "scalar",
    cadence: str = "rollout",
    storage: str = "history",
) -> MetricDefinition:
    return MetricDefinition(name, description, unit, cadence, storage)


METRIC_DEFINITIONS = (
    _definition(GLOBAL_STEP, "Policy environment transitions consumed.", "steps", "frame"),
    _definition(TRAIN_EPISODE_RETURN_SHAPED_MEAN, "Mean shaped episode return."),
    _definition(TRAIN_EPISODE_LENGTH_MEAN, "Mean episode length.", "steps"),
    _definition(TRAIN_EPISODE_COUNT, "Cumulative completed training episodes.", "episodes"),
    _definition(TRAIN_OUTCOME_TERMINAL_COUNT, "Cumulative terminal episode records.", "episodes"),
    _definition("train/outcome/reason/{reason}/count", "Cumulative episodes containing an outcome reason.", "episodes"),
    _definition("train/outcome/reason/{reason}/rate/window_100", "Reason incidence over the latest 100 terminal episodes.", "fraction"),
    _definition("train/outcome/success/from/{start}/count", "Cumulative successful episodes from a start.", "episodes"),
    _definition("train/outcome/success/from/{start}/attempts", "Cumulative episode attempts from a start.", "episodes"),
    _definition("train/outcome/success/from/{start}/rate/current", "Cumulative success rate from a start.", "fraction"),
    _definition("train/outcome/success/from/{start}/rate/window_100", "Success rate over the latest 100 attempts from a start.", "fraction"),
    _definition(TRAIN_OUTCOME_SUCCESS_RATE_CURRENT_MIN, "Minimum cumulative success rate across observed starts.", "fraction"),
    _definition(TRAIN_OUTCOME_SUCCESS_RATE_CURRENT_MEAN, "Mean cumulative success rate across observed starts.", "fraction"),
    _definition(TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN, "Minimum window-100 success rate after every start has 100 attempts.", "fraction"),
    _definition(TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MEAN, "Mean window-100 success rate after every start has 100 attempts.", "fraction"),
    _definition(TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE, "Configured starts with an attempt divided by configured starts.", "fraction"),
    _definition("train/reward/shaped/{stat}", "Distribution of shaped per-step reward."),
    _definition("train/reward/raw/{stat}", "Distribution of raw per-step reward when distinct from shaped reward."),
    _definition("train/reward/component/{component}/{stat}", "Active reward-component attribution."),
    _definition("train/reward/signal/{signal}/{stat}", "Configured reward-source signal."),
    _definition("train/algorithm/{algorithm}/update/{metric}", "Algorithm update-health diagnostic."),
    _definition("train/algorithm/{algorithm}/policy/{metric}", "Policy behavior or distribution diagnostic."),
    _definition("train/algorithm/{algorithm}/value/{metric}", "Value-function diagnostic."),
    _definition("train/algorithm/{algorithm}/rollout/{distribution}/{stat}", "Rollout-buffer distribution diagnostic."),
    _definition("train/algorithm/{algorithm}/hyperparameter/{metric}", "Scheduled algorithm hyperparameter."),
    _definition("train/algorithm/{algorithm}/advantage/{metric}", "Advantage normalization diagnostic."),
    _definition("train/algorithm/{algorithm}/advantage/task/{task}/{stat}", "Per-task advantage normalization diagnostic."),
    _definition("train/throughput/{metric}", "Training-loop rate or phase duration."),
    _definition(TRAIN_ARTIFACT_SAVE_SECONDS, "Local model save duration.", "seconds", "artifact"),
    _definition(TRAIN_ARTIFACT_UPLOAD_SECONDS, "External storage and W&B artifact publication duration.", "seconds", "artifact"),
    _definition("eval/{protocol}/episode/return/{stat}", "Evaluation episode-return distribution.", "return", "evaluation"),
    _definition("eval/{protocol}/episode/length/mean", "Mean evaluation episode length.", "steps", "evaluation"),
    _definition("eval/{protocol}/episode/count", "Evaluation episodes represented.", "episodes", "evaluation"),
    _definition("eval/{protocol}/outcome/success/from/{start}/rate", "Evaluation success rate from a start.", "fraction", "evaluation"),
    _definition("eval/{protocol}/outcome/success/rate/{stat}", "Aggregate per-start evaluation success rate.", "fraction", "evaluation"),
    _definition("eval/{protocol}/outcome/reason/{reason}/count", "Evaluation episodes containing a reason.", "episodes", "evaluation"),
    _definition("eval/{protocol}/outcome/reason/{reason}/rate", "Evaluation reason incidence.", "fraction", "evaluation"),
    _definition("eval/{protocol}/progress/{progress}/{stat}", "Goal-configured evaluation progress summary.", "value", "evaluation"),
    _definition("eval/{protocol}/checkpoint/{field}", "Evaluated checkpoint identity.", "metadata", "evaluation"),
    _definition("eval/{protocol}/duration/seconds", "Evaluation wall duration.", "seconds", "evaluation"),
    _definition("eval/{protocol}/source", "Evaluation execution source.", "text", "evaluation"),
    _definition("eval/{protocol}/candidate/{field}", "Staged checkpoint decision signal.", "scalar", "evaluation"),
    _definition(EVAL_FULL_BY_START, "Structured full-evaluation evidence by start and reason.", "table", "evaluation"),
    _definition("leader/checkpoint/{field}", "Selected checkpoint summary field.", "summary", "selection", "summary"),
)


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PLACEHOLDER_PATTERNS = {
    "protocol": "(?:screen|confirm|full)",
    "stat": "[A-Za-z0-9_.-]+",
    "algorithm": "[A-Za-z0-9_.-]+",
    "reason": "[A-Za-z0-9_.-]+",
    "start": "[A-Za-z0-9_.-]+",
    "component": "[A-Za-z0-9_.-]+",
    "signal": "[A-Za-z0-9_.-]+",
    "metric": "[A-Za-z0-9_.-]+",
    "distribution": "[A-Za-z0-9_.-]+",
    "task": "[A-Za-z0-9_.-]+",
    "progress": "[A-Za-z0-9_.-]+",
    "field": "[A-Za-z0-9_.-]+",
}


def _definition_pattern(template: str) -> re.Pattern[str]:
    cursor = 0
    parts: list[str] = []
    for match in re.finditer(r"\{([a-z_]+)\}", template):
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(_PLACEHOLDER_PATTERNS[match.group(1)])
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$")


_DEFINITION_PATTERNS = tuple(
    (definition, _definition_pattern(definition.name)) for definition in METRIC_DEFINITIONS
)


def metric_definition(name: str) -> MetricDefinition | None:
    for definition, pattern in _DEFINITION_PATTERNS:
        if pattern.fullmatch(name):
            return definition
    return None


def validate_metric_name(name: str) -> str:
    if metric_definition(name) is None:
        raise ValueError(f"unknown metric name: {name}")
    return name


def validate_metric_payload(payload: Mapping[str, Any]) -> None:
    for name in payload:
        validate_metric_name(str(name))


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    if not segment or _SAFE_SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(
            f"metric dimension must match {_SAFE_SEGMENT_RE.pattern}: {value!r}"
        )
    return segment


def metric_value_segment(value: object) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("metric dimension sequence must not be empty")
        return "-".join(metric_path_segment(item) for item in value)
    return metric_path_segment(value)


def stat_metric(prefix: str, stat: str) -> str:
    return f"{prefix}/{metric_path_segment(stat)}"


def eval_metric(protocol: str, suffix: str) -> str:
    protocol = metric_path_segment(protocol)
    if protocol not in EVAL_PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    return f"{EVAL_ROOT}/{protocol}/{suffix.strip('/')}"


def checkpoint_eval_stage_metric(stage_name: str, name: str) -> str:
    return eval_metric(stage_name, name)


def staged_metric_name(stage_name: str, metric_name: str) -> str:
    if metric_name == GLOBAL_STEP:
        return metric_name
    suffix = metric_name.removeprefix(f"{EVAL_FULL_ROOT}/")
    if suffix == metric_name:
        raise ValueError(f"staged evaluation requires an eval/full metric: {metric_name}")
    return eval_metric(stage_name, suffix)


def train_outcome_reason_count_metric(reason: object) -> str:
    return f"train/outcome/reason/{metric_path_segment(reason)}/count"


def train_outcome_reason_window_rate_metric(reason: object) -> str:
    return f"train/outcome/reason/{metric_path_segment(reason)}/rate/window_100"


def train_success_from_metric(start: object, suffix: str) -> str:
    return f"{TRAIN_OUTCOME_SUCCESS_ROOT}/from/{metric_value_segment(start)}/{suffix}"


def train_success_count_metric(start: object) -> str:
    return train_success_from_metric(start, "count")


def train_success_attempts_metric(start: object) -> str:
    return train_success_from_metric(start, "attempts")


def train_success_current_rate_metric(start: object) -> str:
    return train_success_from_metric(start, "rate/current")


def train_success_window_rate_metric(start: object) -> str:
    return train_success_from_metric(start, "rate/window_100")


def train_reward_component_metric(component: object, stat: str) -> str:
    return f"{TRAIN_REWARD_ROOT}/component/{metric_path_segment(component)}/{metric_path_segment(stat)}"


def train_reward_signal_metric(signal: object, stat: str) -> str:
    return f"{TRAIN_REWARD_ROOT}/signal/{metric_path_segment(signal)}/{metric_path_segment(stat)}"


def train_adv_task_metric(task_id: object, stat: str) -> str:
    return (
        f"{TRAIN_ALGORITHM_PPO_ROOT}/advantage/task/{metric_path_segment(task_id)}/"
        f"{metric_path_segment(stat)}"
    )


def eval_success_from_rate_metric(protocol: str, start: object) -> str:
    return eval_metric(protocol, f"outcome/success/from/{metric_value_segment(start)}/rate")


def eval_success_rate_metric(protocol: str, stat: str) -> str:
    return eval_metric(protocol, f"outcome/success/rate/{metric_path_segment(stat)}")


def eval_reason_count_metric(protocol: str, reason: object) -> str:
    return eval_metric(protocol, f"outcome/reason/{metric_path_segment(reason)}/count")


def eval_reason_rate_metric(protocol: str, reason: object) -> str:
    return eval_metric(protocol, f"outcome/reason/{metric_path_segment(reason)}/rate")


def eval_progress_metric(protocol: str, progress: object, stat: str) -> str:
    return eval_metric(
        protocol,
        f"progress/{metric_path_segment(progress)}/{metric_path_segment(stat)}",
    )


SB3_SCALAR_MAP = {
    "rollout/ep_rew_mean": (TRAIN_EPISODE_RETURN_SHAPED_MEAN, 1.0),
    "rollout/ep_len_mean": (TRAIN_EPISODE_LENGTH_MEAN, 1.0),
    "train/approx_kl": (TRAIN_PPO_APPROX_KL, 1.0),
    "train/clip_fraction": (TRAIN_PPO_CLIP_FRACTION, 1.0),
    "train/entropy_loss": (TRAIN_PPO_POLICY_ENTROPY, -1.0),
    "train/explained_variance": (TRAIN_PPO_EXPLAINED_VARIANCE, 1.0),
    "train/policy_gradient_loss": (TRAIN_PPO_POLICY_GRADIENT_LOSS, 1.0),
    "train/value_loss": (TRAIN_PPO_VALUE_LOSS, 1.0),
    "train/learning_rate": (TRAIN_PPO_LEARNING_RATE, 1.0),
    "train/std": (TRAIN_PPO_POLICY_DISTRIBUTION_STD, 1.0),
}


def canonical_training_scalars(key_values: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in key_values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        mapped = SB3_SCALAR_MAP.get(str(key))
        if mapped is not None:
            name, multiplier = mapped
            payload[name] = numeric * multiplier
        elif metric_definition(str(key)) is not None and str(key) != GLOBAL_STEP:
            payload[str(key)] = numeric
    validate_metric_payload(payload)
    return payload


def render_metric_registry_markdown() -> str:
    lines = [
        "| Metric or template | Meaning | Unit | Cadence | Surface |",
        "|---|---|---|---|---|",
    ]
    for definition in METRIC_DEFINITIONS:
        lines.append(
            f"| `{definition.name}` | {definition.description} | {definition.unit} | "
            f"{definition.cadence} | {definition.storage} |"
        )
    return "\n".join(lines)
