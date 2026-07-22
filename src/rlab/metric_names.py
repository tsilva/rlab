from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


METRICS_SCHEMA_VERSION = 5
GLOBAL_STEP = "global_step"

TRAIN_EPISODE_RETURN_SHAPED_MEAN = "train/episode/return/shaped/mean"
TRAIN_EPISODE_LENGTH_MEAN = "train/episode/length/mean"
TRAIN_EPISODE_COUNT = "train/episode/count"

TRAIN_OUTCOME_TERMINAL_COUNT = "train/outcome/terminal/count"
TRAIN_OUTCOME_SUCCESS_ROOT = "train/outcome/success"
TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/current/rate/min"
TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/current/rate/mean"
TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/window_100/rate/min"
TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/window_100/rate/mean"
LEGACY_TRAIN_OUTCOME_SUCCESS_RATE_WINDOW_100_MIN = (
    f"{TRAIN_OUTCOME_SUCCESS_ROOT}/rate/window_100/min"
)
TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE = f"{TRAIN_OUTCOME_SUCCESS_ROOT}/start_coverage/rate"

TRAIN_REWARD_ROOT = "train/reward"

TRAIN_ALGORITHM_ROOT = "train/algorithm"
TRAIN_ACTOR_CRITIC_ALGORITHMS = ("ppo", "a2c")
TRAIN_ALGORITHM_JERK_ROOT = f"{TRAIN_ALGORITHM_ROOT}/jerk"
TRAIN_ALGORITHM_JERK_RETAINED_COUNT = f"{TRAIN_ALGORITHM_JERK_ROOT}/retained/count"
TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/return_mean"
TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH = f"{TRAIN_ALGORITHM_JERK_ROOT}/best/sequence_length"
TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN = (
    f"{TRAIN_ALGORITHM_JERK_ROOT}/archive/selected_prefix_return_mean"
)
TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY = f"{TRAIN_ALGORITHM_JERK_ROOT}/exploit/probability"


def train_algorithm_root(algorithm_id: str) -> str:
    if algorithm_id not in TRAIN_ACTOR_CRITIC_ALGORITHMS:
        raise ValueError(f"unsupported actor-critic algorithm id: {algorithm_id}")
    return f"{TRAIN_ALGORITHM_ROOT}/{algorithm_id}"


def train_algorithm_metric(algorithm_id: str, suffix: str) -> str:
    return f"{train_algorithm_root(algorithm_id)}/{suffix}"


TRAIN_ALGORITHM_PPO_ROOT = train_algorithm_root("ppo")
TRAIN_ALGORITHM_A2C_ROOT = train_algorithm_root("a2c")
TRAIN_PPO_APPROX_KL = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/approx_kl"
TRAIN_PPO_CLIP_FRACTION = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/clip_fraction"
TRAIN_PPO_EXPLAINED_VARIANCE = f"{TRAIN_ALGORITHM_PPO_ROOT}/value/explained_variance"
TRAIN_PPO_POLICY_GRADIENT_LOSS = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/policy_gradient_loss"
TRAIN_PPO_VALUE_LOSS = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/value_loss"
TRAIN_PPO_LEARNING_RATE = f"{TRAIN_ALGORITHM_PPO_ROOT}/update/learning_rate"
TRAIN_PPO_POLICY_ENTROPY = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/entropy"
TRAIN_PPO_POLICY_DISTRIBUTION_STD = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/distribution_std"
TRAIN_PPO_POLICY_DOMINANT_ACTION_RATE = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/dominant_action_rate"
TRAIN_PPO_POLICY_ACTION_HIST = f"{TRAIN_ALGORITHM_PPO_ROOT}/policy/action_hist"
TRAIN_PPO_VALUE_PREDICTION_ROOT = f"{TRAIN_ALGORITHM_PPO_ROOT}/rollout/value_prediction"
TRAIN_PPO_VALUE_PREDICTION_HIST = f"{TRAIN_PPO_VALUE_PREDICTION_ROOT}/hist"
TRAIN_PPO_ADVANTAGE_ROOT = f"{TRAIN_ALGORITHM_PPO_ROOT}/rollout/advantage"
TRAIN_PPO_ADVANTAGE_HIST = f"{TRAIN_PPO_ADVANTAGE_ROOT}/hist"
TRAIN_PPO_ENTROPY_COEFFICIENT = f"{TRAIN_ALGORITHM_PPO_ROOT}/hyperparameter/entropy_coefficient"

TRAIN_A2C_EXPLAINED_VARIANCE = f"{TRAIN_ALGORITHM_A2C_ROOT}/value/explained_variance"
TRAIN_A2C_POLICY_GRADIENT_LOSS = f"{TRAIN_ALGORITHM_A2C_ROOT}/update/policy_gradient_loss"
TRAIN_A2C_VALUE_LOSS = f"{TRAIN_ALGORITHM_A2C_ROOT}/update/value_loss"
TRAIN_A2C_LEARNING_RATE = f"{TRAIN_ALGORITHM_A2C_ROOT}/update/learning_rate"
TRAIN_A2C_POLICY_ENTROPY = f"{TRAIN_ALGORITHM_A2C_ROOT}/policy/entropy"
TRAIN_A2C_POLICY_DISTRIBUTION_STD = f"{TRAIN_ALGORITHM_A2C_ROOT}/policy/distribution_std"
TRAIN_A2C_POLICY_DOMINANT_ACTION_RATE = f"{TRAIN_ALGORITHM_A2C_ROOT}/policy/dominant_action_rate"
TRAIN_A2C_POLICY_ACTION_HIST = f"{TRAIN_ALGORITHM_A2C_ROOT}/policy/action_hist"
TRAIN_A2C_VALUE_PREDICTION_ROOT = f"{TRAIN_ALGORITHM_A2C_ROOT}/rollout/value_prediction"
TRAIN_A2C_VALUE_PREDICTION_HIST = f"{TRAIN_A2C_VALUE_PREDICTION_ROOT}/hist"
TRAIN_A2C_ADVANTAGE_ROOT = f"{TRAIN_ALGORITHM_A2C_ROOT}/rollout/advantage"
TRAIN_A2C_ADVANTAGE_HIST = f"{TRAIN_A2C_ADVANTAGE_ROOT}/hist"
TRAIN_A2C_ENTROPY_COEFFICIENT = f"{TRAIN_ALGORITHM_A2C_ROOT}/hyperparameter/entropy_coefficient"

TRAIN_THROUGHPUT_ROOT = "train/throughput"
TRAIN_THROUGHPUT_LOOP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/loop_fps"
TRAIN_THROUGHPUT_ROLLOUT_FPS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_fps"
TRAIN_THROUGHPUT_ENV_STEP_FPS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_fps"
TRAIN_THROUGHPUT_LOOP_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/loop_seconds"
TRAIN_THROUGHPUT_ROLLOUT_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_seconds"
TRAIN_THROUGHPUT_ENV_STEP_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/env_step_seconds"
TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/rollout_overhead_seconds"
TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS = f"{TRAIN_THROUGHPUT_ROOT}/between_rollouts_seconds"

TRAIN_ARTIFACT_SAVE_SECONDS = "train/artifact/save/seconds"
TRAIN_ARTIFACT_UPLOAD_SECONDS = "train/artifact/upload/seconds"

EVAL_ROOT = "eval"
EVAL_PROTOCOLS = ("screen", "confirm", "full")
ACTIVE_EVAL_PROTOCOLS = ("full",)
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
EVAL_SCREEN_PREVIEW = "eval/screen/preview"
EVAL_ACCEPTANCE_PASS = "eval/acceptance/pass"
EVAL_ACCEPTANCE_EPISODES_PLANNED = "eval/acceptance/episodes/planned"
EVAL_ACCEPTANCE_EPISODES_COMPLETED = "eval/acceptance/episodes/completed"
EVAL_ACCEPTANCE_FAILURE_COUNT = "eval/acceptance/failure/count"
EVAL_ACCEPTANCE_DURATION_SECONDS = "eval/acceptance/duration/seconds"

CHECKPOINT_EVAL_CANDIDATE_PASS = "eval/confirm/candidate/pass"
CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX = "eval/confirm/candidate/stage_index"
CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP = "eval/confirm/candidate/checkpoint_step"
CHECKPOINT_EVAL_CANDIDATE_EPISODES = "eval/confirm/candidate/episodes"

LEADER_CHECKPOINT_SUCCESS_RATE_MIN = "leader/checkpoint/success_rate_min"
LEADER_CHECKPOINT_SUCCESS_RATE_MEAN = "leader/checkpoint/success_rate_mean"
LEADER_CHECKPOINT_ACCEPTANCE_PASS = "leader/checkpoint/acceptance_pass"
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
    _definition(
        TRAIN_EPISODE_RETURN_SHAPED_MEAN,
        "Rolling mean shaped return over the latest 100 completed training episodes.",
    ),
    _definition(
        TRAIN_EPISODE_LENGTH_MEAN,
        "Rolling mean length over the latest 100 completed training episodes.",
        "steps",
    ),
    _definition(TRAIN_OUTCOME_TERMINAL_COUNT, "Cumulative terminal episode records.", "episodes"),
    _definition(
        "train/outcome/reason/{reason}/count",
        "Cumulative failed episodes containing a reason.",
        "episodes",
    ),
    _definition(
        "train/outcome/reason/{reason}/rate/window_100",
        "Failure-reason incidence over the latest 100 terminal episodes.",
        "fraction",
    ),
    _definition(
        "train/outcome/success/from/{start}/count",
        "Cumulative successful episodes from a start.",
        "episodes",
    ),
    _definition(
        "train/outcome/success/from/{start}/attempts",
        "Cumulative episode attempts from a start.",
        "episodes",
    ),
    _definition(
        "train/outcome/success/from/{start}/rate/window_100",
        "Success rate over the latest 100 attempts from a start.",
        "fraction",
    ),
    _definition(
        TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN,
        "Minimum cumulative success rate across observed starts.",
        "fraction",
    ),
    _definition(
        TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
        "Mean cumulative success rate across observed starts.",
        "fraction",
    ),
    _definition(
        TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
        "Minimum window-100 success rate after every start has 100 attempts.",
        "fraction",
    ),
    _definition(
        TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN,
        "Mean window-100 success rate after every start has 100 attempts.",
        "fraction",
    ),
    _definition(
        TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
        "Configured starts with an attempt divided by configured starts.",
        "fraction",
    ),
    *(
        _definition(f"{TRAIN_REWARD_ROOT}/shaped/{stat}", "Distribution of shaped per-step reward.")
        for stat in ("mean", "std", "min", "max", "nonzero_rate")
    ),
    *(
        _definition(
            f"{TRAIN_REWARD_ROOT}/raw/{stat}",
            "Distribution of raw per-step reward when distinct from shaped reward.",
        )
        for stat in ("mean", "std")
    ),
    *(
        _definition(
            f"{TRAIN_REWARD_ROOT}/component/{{component}}/{stat}",
            "Active reward-component attribution.",
        )
        for stat in ("mean", "nonzero_rate", "share")
    ),
    *(
        _definition(
            f"{TRAIN_REWARD_ROOT}/signal/{{signal}}/{stat}", "Configured reward-source signal."
        )
        for stat in ("mean", "max", "nonzero_rate")
    ),
    _definition(TRAIN_PPO_APPROX_KL, "Approximate KL divergence for the PPO update."),
    _definition(TRAIN_PPO_CLIP_FRACTION, "Fraction of policy ratios clipped by PPO."),
    _definition(
        TRAIN_ALGORITHM_JERK_RETAINED_COUNT,
        "Distinct action sequences retained by JERK search.",
        "sequences",
    ),
    _definition(
        TRAIN_ALGORITHM_JERK_BEST_RETURN_MEAN,
        "Mean observed return of JERK's highest-ranked retained sequence.",
        "return",
    ),
    _definition(
        TRAIN_ALGORITHM_JERK_BEST_SEQUENCE_LENGTH,
        "Action length of JERK's highest-ranked retained sequence.",
        "steps",
    ),
    _definition(
        TRAIN_ALGORITHM_JERK_ARCHIVE_SELECTED_PREFIX_RETURN_MEAN,
        "Cumulative mean retained-prefix return selected for JERK archive replay.",
        "return",
    ),
    _definition(
        TRAIN_ALGORITHM_JERK_EXPLOIT_PROBABILITY,
        "Probability that JERK starts an episode by sampling a retained archive sequence.",
        "fraction",
    ),
    _definition(
        "train/algorithm/{algorithm}/value/explained_variance",
        "Actor-critic value-function explained variance.",
    ),
    _definition(
        "train/algorithm/{algorithm}/update/policy_gradient_loss",
        "Actor-critic policy-gradient loss.",
    ),
    _definition(
        "train/algorithm/{algorithm}/update/value_loss",
        "Actor-critic value loss.",
    ),
    _definition(
        "train/algorithm/{algorithm}/update/learning_rate",
        "Current actor-critic learning rate.",
    ),
    _definition(
        "train/algorithm/{algorithm}/policy/entropy",
        "Positive actor-critic policy entropy.",
    ),
    _definition(
        "train/algorithm/{algorithm}/policy/distribution_std",
        "Continuous-action distribution standard deviation.",
    ),
    _definition(
        "train/algorithm/{algorithm}/policy/dominant_action_rate",
        "Fraction assigned to the most frequent sampled discrete action.",
    ),
    _definition(
        "train/algorithm/{algorithm}/policy/action_hist",
        "Sampled discrete-action histogram.",
        "histogram",
        "every 64 rollouts",
    ),
    *(
        _definition(
            f"train/algorithm/{{algorithm}}/rollout/value_prediction/{stat}",
            "Rollout value-prediction distribution diagnostic.",
        )
        for stat in ("mean", "std", "min", "max")
    ),
    _definition(
        "train/algorithm/{algorithm}/rollout/value_prediction/hist",
        "Rollout value-prediction histogram.",
        "histogram",
        "every 64 rollouts",
    ),
    *(
        _definition(
            f"train/algorithm/{{algorithm}}/rollout/advantage/{stat}",
            "Rollout advantage distribution diagnostic.",
        )
        for stat in ("mean", "std", "min", "max")
    ),
    _definition(
        "train/algorithm/{algorithm}/rollout/advantage/hist",
        "Rollout advantage histogram.",
        "histogram",
        "every 64 rollouts",
    ),
    _definition(
        "train/algorithm/{algorithm}/hyperparameter/entropy_coefficient",
        "Current scheduled entropy coefficient.",
    ),
    _definition(
        TRAIN_THROUGHPUT_LOOP_FPS,
        "Policy transitions divided by rollout-start-to-next-rollout-start wall time.",
        "steps/second",
    ),
    _definition(
        TRAIN_THROUGHPUT_ROLLOUT_FPS,
        "Policy transitions divided by rollout-collection wall time.",
        "steps/second",
    ),
    _definition(
        TRAIN_THROUGHPUT_ENV_STEP_FPS,
        "Policy transitions divided by native-provider step wall time accumulated during the rollout.",
        "steps/second",
    ),
    _definition(
        TRAIN_THROUGHPUT_ROLLOUT_SECONDS,
        "Wall time spent collecting one rollout.",
        "seconds",
    ),
    _definition(
        TRAIN_THROUGHPUT_ENV_STEP_SECONDS,
        "Native-provider step wall time accumulated while collecting one rollout.",
        "seconds",
    ),
    _definition(
        TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
        "Rollout wall time outside native-provider step calls, including policy inference and wrapper, buffer, reset, task, and callback work.",
        "seconds",
    ),
    _definition(
        TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
        "Wall time after rollout collection and before the next rollout, including optimizer updates, callbacks, and logging.",
        "seconds",
    ),
    _definition(TRAIN_ARTIFACT_SAVE_SECONDS, "Local model save duration.", "seconds", "artifact"),
    _definition(
        TRAIN_ARTIFACT_UPLOAD_SECONDS,
        "External storage and W&B artifact publication duration.",
        "seconds",
        "artifact",
    ),
    *(
        _definition(
            f"eval/{{protocol}}/episode/return/{stat}",
            "Evaluation episode-return distribution.",
            "return",
            "evaluation",
        )
        for stat in ("mean", "std", "median")
    ),
    _definition(
        EVAL_FULL_EPISODE_RETURN_BEST,
        "Best full-evaluation episode return.",
        "return",
        "evaluation",
    ),
    _definition(
        "eval/{protocol}/episode/length/mean",
        "Mean evaluation episode length.",
        "steps",
        "evaluation",
    ),
    _definition(
        "eval/{protocol}/episode/count",
        "Evaluation episodes represented.",
        "episodes",
        "evaluation",
    ),
    _definition(
        "eval/{protocol}/outcome/success/from/{start}/rate",
        "Evaluation success rate from a start.",
        "fraction",
        "evaluation",
    ),
    *(
        _definition(
            f"eval/{{protocol}}/outcome/success/rate/{stat}",
            "Aggregate per-start evaluation success rate.",
            "fraction",
            "evaluation",
        )
        for stat in ("min", "mean")
    ),
    _definition(
        "eval/{protocol}/outcome/reason/{reason}/rate",
        "Evaluation failure-reason incidence.",
        "fraction",
        "evaluation",
    ),
    *(
        _definition(
            f"eval/full/progress/{{progress}}/{stat}",
            "Goal-configured full-evaluation progress summary.",
            "value",
            "evaluation",
        )
        for stat in ("mean", "max")
    ),
    _definition(
        "eval/{protocol}/checkpoint/artifact",
        "Evaluated checkpoint artifact reference.",
        "metadata",
        "evaluation",
    ),
    _definition(
        "eval/{protocol}/duration/seconds", "Evaluation wall duration.", "seconds", "evaluation"
    ),
    _definition("eval/{protocol}/source", "Evaluation execution source.", "text", "evaluation"),
    _definition(
        EVAL_ACCEPTANCE_PASS,
        "Per-checkpoint acceptance result; W&B summarizes its history with max, not as the verdict.",
        "boolean",
        "acceptance evaluation",
    ),
    _definition(
        EVAL_ACCEPTANCE_EPISODES_PLANNED,
        "Exact episode identities required by the acceptance manifest.",
        "episodes",
        "acceptance evaluation",
    ),
    _definition(
        EVAL_ACCEPTANCE_EPISODES_COMPLETED,
        "Valid planned episode rows completed before acceptance or fail-fast rejection.",
        "episodes",
        "acceptance evaluation",
    ),
    _definition(
        EVAL_ACCEPTANCE_DURATION_SECONDS,
        "Acceptance-worker evaluation wall duration.",
        "seconds",
        "acceptance evaluation",
    ),
    _definition(
        EVAL_FULL_BY_START,
        "Structured full-evaluation evidence by start and reason.",
        "table",
        "evaluation",
    ),
    _definition(
        LEADER_CHECKPOINT_ACCEPTANCE_PASS,
        "Canonical promoted-checkpoint acceptance verdict restamped from database promotion state.",
        "boolean",
        "selection",
        "summary",
    ),
    *(
        _definition(name, "Selected checkpoint summary field.", "summary", "selection", "summary")
        for name in (
            LEADER_CHECKPOINT_SUCCESS_RATE_MIN,
            LEADER_CHECKPOINT_SUCCESS_RATE_MEAN,
            LEADER_CHECKPOINT_OBJECTIVE,
            LEADER_CHECKPOINT_RETURN_MEAN,
            LEADER_CHECKPOINT_BEST_RETURN,
            LEADER_CHECKPOINT_RANK_VALUES,
            LEADER_CHECKPOINT_PROGRESS_MAX,
            LEADER_CHECKPOINT_STEP,
            LEADER_CHECKPOINT_ARTIFACT_REF,
            LEADER_CHECKPOINT_EVAL_SOURCE,
            LEADER_CHECKPOINT_UPDATED_AT,
        )
    ),
)


V4_ONLY_METRIC_DEFINITIONS = (
    _definition(TRAIN_EPISODE_COUNT, "Cumulative completed training episodes.", "episodes"),
    _definition(
        "train/outcome/success/from/{start}/rate/current",
        "Cumulative success rate from a start.",
        "fraction",
    ),
    _definition(
        TRAIN_THROUGHPUT_LOOP_SECONDS,
        "Wall time from one rollout start to the next rollout start.",
        "seconds",
    ),
    _definition(
        "eval/{protocol}/outcome/reason/{reason}/count",
        "Failed evaluation episodes containing a reason.",
        "episodes",
        "evaluation",
    ),
    _definition(
        "eval/{protocol}/checkpoint/step", "Evaluated checkpoint step.", "steps", "evaluation"
    ),
    _definition(
        EVAL_ACCEPTANCE_FAILURE_COUNT,
        "Failed planned episodes; zero for acceptance and one for fail-fast rejection.",
        "episodes",
        "acceptance evaluation",
    ),
    _definition(
        EVAL_SCREEN_PREVIEW,
        "Historical checkpoint-screen preview; new acceptance jobs do not capture or publish previews.",
        "html",
        "historical evaluation",
        "media",
    ),
    *(
        _definition(
            f"eval/{protocol}/candidate/pass",
            "Historical staged checkpoint pass signal.",
            "boolean",
            "historical evaluation",
        )
        for protocol in ("screen", "confirm")
    ),
    *(
        _definition(
            f"eval/{protocol}/candidate/stage_index",
            "Historical staged checkpoint protocol index.",
            "index",
            "historical evaluation",
        )
        for protocol in ("screen", "confirm")
    ),
    _definition(
        CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP,
        "Historical confirmed candidate checkpoint step.",
        "steps",
        "historical evaluation",
    ),
    _definition(
        CHECKPOINT_EVAL_CANDIDATE_EPISODES,
        "Historical confirmed candidate evaluation episodes.",
        "episodes",
        "historical evaluation",
    ),
    *(
        _definition(name, "Selected checkpoint summary field.", "summary", "selection", "summary")
        for name in (
            LEADER_CHECKPOINT_OBJECTIVE_NAME,
            LEADER_CHECKPOINT_RANK,
            LEADER_CHECKPOINT_STEPS_TO_GOAL,
            LEADER_CHECKPOINT_LOCAL_PATH,
        )
    ),
)


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PLACEHOLDER_PATTERNS = {
    "algorithm": "(?:ppo|a2c)",
    "protocol": "(?:full)",
    "reason": "[A-Za-z0-9_.-]+",
    "start": "[A-Za-z0-9_.-]+",
    "component": "[A-Za-z0-9_.-]+",
    "signal": "[A-Za-z0-9_.-]+",
    "progress": "[A-Za-z0-9_.-]+",
}
_V4_PLACEHOLDER_PATTERNS = {
    **_PLACEHOLDER_PATTERNS,
    "protocol": "(?:screen|confirm|full)",
}


def _definition_pattern(
    template: str, *, placeholders: Mapping[str, str] = _PLACEHOLDER_PATTERNS
) -> re.Pattern[str]:
    cursor = 0
    parts: list[str] = []
    for match in re.finditer(r"\{([a-z_]+)\}", template):
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(placeholders[match.group(1)])
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$")


_DEFINITION_PATTERNS = tuple(
    (definition, _definition_pattern(definition.name)) for definition in METRIC_DEFINITIONS
)
_V4_DEFINITION_PATTERNS = tuple(
    (
        definition,
        _definition_pattern(definition.name, placeholders=_V4_PLACEHOLDER_PATTERNS),
    )
    for definition in (*METRIC_DEFINITIONS, *V4_ONLY_METRIC_DEFINITIONS)
)


def _supported_schema_version(schema_version: int) -> int:
    try:
        version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported metrics schema version: {schema_version!r}") from exc
    if version not in {4, METRICS_SCHEMA_VERSION}:
        raise ValueError(f"unsupported metrics schema version: {version}")
    return version


def metric_definition(
    name: str, *, schema_version: int = METRICS_SCHEMA_VERSION
) -> MetricDefinition | None:
    version = _supported_schema_version(schema_version)
    patterns = _V4_DEFINITION_PATTERNS if version == 4 else _DEFINITION_PATTERNS
    for definition, pattern in patterns:
        if pattern.fullmatch(name):
            return definition
    return None


def validate_metric_name(name: str, *, schema_version: int = METRICS_SCHEMA_VERSION) -> str:
    if metric_definition(name, schema_version=schema_version) is None:
        raise ValueError(f"unknown metric name: {name}")
    return name


def validate_metric_payload(
    payload: Mapping[str, Any], *, schema_version: int = METRICS_SCHEMA_VERSION
) -> None:
    for name in payload:
        validate_metric_name(str(name), schema_version=schema_version)


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    if not segment or _SAFE_SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(f"metric dimension must match {_SAFE_SEGMENT_RE.pattern}: {value!r}")
    return segment


def metric_value_segment(value: object) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("metric dimension sequence must not be empty")
        return "-".join(metric_path_segment(item) for item in value)
    return metric_path_segment(value)


def stat_metric(prefix: str, stat: str) -> str:
    return validate_metric_name(f"{prefix}/{metric_path_segment(stat)}")


def eval_metric(
    protocol: str, suffix: str, *, schema_version: int = METRICS_SCHEMA_VERSION
) -> str:
    protocol = metric_path_segment(protocol)
    version = _supported_schema_version(schema_version)
    protocols = EVAL_PROTOCOLS if version == 4 else ACTIVE_EVAL_PROTOCOLS
    if protocol not in protocols:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    return validate_metric_name(
        f"{EVAL_ROOT}/{protocol}/{suffix.strip('/')}", schema_version=schema_version
    )


def checkpoint_eval_stage_metric(stage_name: str, name: str) -> str:
    return eval_metric(stage_name, name, schema_version=4)


def staged_metric_name(stage_name: str, metric_name: str) -> str:
    if metric_name == GLOBAL_STEP:
        return metric_name
    suffix = metric_name.removeprefix(f"{EVAL_FULL_ROOT}/")
    if suffix == metric_name:
        raise ValueError(f"staged evaluation requires an eval/full metric: {metric_name}")
    return eval_metric(stage_name, suffix, schema_version=4)


def train_outcome_reason_count_metric(reason: object) -> str:
    return validate_metric_name(f"train/outcome/reason/{metric_path_segment(reason)}/count")


def train_outcome_reason_window_rate_metric(reason: object) -> str:
    return validate_metric_name(
        f"train/outcome/reason/{metric_path_segment(reason)}/rate/window_100"
    )


def train_success_from_metric(start: object, suffix: str) -> str:
    return validate_metric_name(
        f"{TRAIN_OUTCOME_SUCCESS_ROOT}/from/{metric_value_segment(start)}/{suffix}"
    )


def train_success_count_metric(start: object) -> str:
    return train_success_from_metric(start, "count")


def train_success_attempts_metric(start: object) -> str:
    return train_success_from_metric(start, "attempts")


def train_success_current_rate_metric(start: object) -> str:
    return validate_metric_name(
        f"{TRAIN_OUTCOME_SUCCESS_ROOT}/from/{metric_value_segment(start)}/rate/current",
        schema_version=4,
    )


def train_success_window_rate_metric(start: object) -> str:
    return train_success_from_metric(start, "rate/window_100")


def train_reward_component_metric(component: object, stat: str) -> str:
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/component/{metric_path_segment(component)}/{metric_path_segment(stat)}"
    )


def train_reward_signal_metric(signal: object, stat: str) -> str:
    return validate_metric_name(
        f"{TRAIN_REWARD_ROOT}/signal/{metric_path_segment(signal)}/{metric_path_segment(stat)}"
    )


def eval_success_from_rate_metric(
    protocol: str, start: object, *, schema_version: int = METRICS_SCHEMA_VERSION
) -> str:
    return eval_metric(
        protocol,
        f"outcome/success/from/{metric_value_segment(start)}/rate",
        schema_version=schema_version,
    )


def eval_success_rate_metric(
    protocol: str, stat: str, *, schema_version: int = METRICS_SCHEMA_VERSION
) -> str:
    return eval_metric(
        protocol,
        f"outcome/success/rate/{metric_path_segment(stat)}",
        schema_version=schema_version,
    )


def eval_reason_count_metric(protocol: str, reason: object) -> str:
    return eval_metric(
        protocol,
        f"outcome/reason/{metric_path_segment(reason)}/count",
        schema_version=4,
    )


def eval_reason_rate_metric(
    protocol: str, reason: object, *, schema_version: int = METRICS_SCHEMA_VERSION
) -> str:
    return eval_metric(
        protocol,
        f"outcome/reason/{metric_path_segment(reason)}/rate",
        schema_version=schema_version,
    )


def eval_progress_metric(
    protocol: str,
    progress: object,
    stat: str,
    *,
    schema_version: int = METRICS_SCHEMA_VERSION,
) -> str:
    return eval_metric(
        protocol,
        f"progress/{metric_path_segment(progress)}/{metric_path_segment(stat)}",
        schema_version=schema_version,
    )


SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP = {
    "rollout/ep_rew_mean": (TRAIN_EPISODE_RETURN_SHAPED_MEAN, 1.0),
    "rollout/ep_len_mean": (TRAIN_EPISODE_LENGTH_MEAN, 1.0),
    "train/entropy_loss": ("policy/entropy", -1.0),
    "train/explained_variance": ("value/explained_variance", 1.0),
    "train/policy_gradient_loss": ("update/policy_gradient_loss", 1.0),
    "train/policy_loss": ("update/policy_gradient_loss", 1.0),
    "train/value_loss": ("update/value_loss", 1.0),
    "train/learning_rate": ("update/learning_rate", 1.0),
    "train/std": ("policy/distribution_std", 1.0),
}

SB3_PPO_SCALAR_MAP = {
    "train/approx_kl": (TRAIN_PPO_APPROX_KL, 1.0),
    "train/clip_fraction": (TRAIN_PPO_CLIP_FRACTION, 1.0),
}

SB3_IGNORED_SCALARS = {
    "rollout/ep_rew_mean",  # mapped above; listed here only for documentation symmetry
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
    "train/clip_range",
    "train/clip_range_vf",
    "train/loss",
    "train/n_updates",
}

_RLAB_OWNED_PREFIXES = (
    "train/episode/",
    "train/outcome/",
    "train/reward/",
    "train/algorithm/",
    "train/throughput/",
    "train/artifact/",
    "eval/",
    "leader/",
)


def canonical_training_scalars(
    key_values: Mapping[str, Any],
    *,
    algorithm_id: str = "ppo",
) -> dict[str, float]:
    train_algorithm_root(algorithm_id)
    payload: dict[str, float] = {}
    for key, value in key_values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        numeric = float(value)
        raw_name = str(key)
        mapped = SB3_SHARED_ACTOR_CRITIC_SCALAR_MAP.get(raw_name)
        if mapped is not None:
            name, multiplier = mapped
            if name.startswith(("train/", "eval/", "leader/")):
                payload[name] = numeric * multiplier
            else:
                payload[train_algorithm_metric(algorithm_id, name)] = numeric * multiplier
        elif algorithm_id == "ppo" and (mapped := SB3_PPO_SCALAR_MAP.get(raw_name)) is not None:
            name, multiplier = mapped
            payload[name] = numeric * multiplier
        elif metric_definition(raw_name) is not None and raw_name != GLOBAL_STEP:
            payload[raw_name] = numeric
        elif raw_name in SB3_IGNORED_SCALARS:
            continue
        elif raw_name.startswith(_RLAB_OWNED_PREFIXES):
            raise ValueError(f"unknown rlab metric at logger boundary: {key}")
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
