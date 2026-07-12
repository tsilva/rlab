from __future__ import annotations

import re

GLOBAL_STEP = "global_step"

TIME_TIME_ELAPSED = "time/time_elapsed"

THROUGHPUT_ROLLOUT_FPS = "throughput/rollout_fps"
THROUGHPUT_LOOP_FPS = "throughput/loop_fps"
THROUGHPUT_NATIVE_ENV_STEP_FPS = "throughput/native_env_step_fps"
THROUGHPUT_NATIVE_ENV_STEP_BATCH_FPS = "throughput/native_env_step_batch_fps"
THROUGHPUT_NATIVE_ENV_STEP_SECONDS = "throughput/native_env_step_seconds"
THROUGHPUT_NATIVE_ENV_STEP_FRACTION = "throughput/native_env_step_fraction"

TRAIN_ARTIFACT_STALL_SECONDS = "train/artifact/stall_seconds"
TRAIN_ARTIFACT_LOCAL_SAVE_SECONDS = "train/artifact/local_save_seconds"
TRAIN_ARTIFACT_LOG_SECONDS = "train/artifact/log_seconds"
TRAIN_ARTIFACT_METADATA_SECONDS = "train/artifact/metadata_seconds"
TRAIN_ARTIFACT_STORAGE_UPLOAD_SECONDS = "train/artifact/storage_upload_seconds"
TRAIN_ARTIFACT_WANDB_LOG_SECONDS = "train/artifact/wandb_log_seconds"

ROLLOUT_VALUE_PRED = "rollout/value_pred"
ROLLOUT_VALUE_PRED_HIST = "rollout/value_pred/hist"
ROLLOUT_ADVANTAGE = "rollout/advantage"
ROLLOUT_ADVANTAGE_HIST = "rollout/advantage/hist"

TRAIN_REWARD_COMPONENT_ROOT = "train/reward"
TRAIN_REWARD_SHARE_ROOT = "train/reward_share"
TRAIN_ENT_COEF = "train/ent_coef"
TRAIN_ADV_NORM_MODE = "train/adv_norm/mode"
TRAIN_ADV_ROOT = "train/adv"

TRAIN_DONE_ROOT = "train/done/"
TRAIN_DONE_ALL = "train/done/all"
TRAIN_DONE_MAX_STEPS = "train/done/max_steps"
TRAIN_DONE_UNCLASSIFIED = "train/done/unclassified"

TRAIN_INFO_LEVEL_COMPLETE_ROOT = "train/info/level_complete"
TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_CURRENT = "train/info/level_complete/rate/min/current"
TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_CURRENT = "train/info/level_complete/rate/mean/current"
TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST = "train/info/level_complete/rate/min/last"
TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST = "train/info/level_complete/rate/mean/last"

EVAL_DONE_ALL = "eval/done/all"
EVAL_DONE_ROOT = "eval/done/"
EVAL_DONE_LEVEL_CHANGE = "eval/done/level_change"
EVAL_DONE_LEVEL_CHANGE_RATE = "eval/done/level_change/rate"
EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN = "eval/done/level_change/from_rate/min"
EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN = "eval/done/level_change/from_rate/mean"
EVAL_DONE_MAX_STEPS = "eval/done/max_steps"
EVAL_DONE_MAX_STEPS_RATE = "eval/done/max_steps/rate"
EVAL_DONE_TERMINATED = "eval/done/terminated"
EVAL_DONE_TERMINATED_RATE = "eval/done/terminated/rate"
EVAL_DONE_UNCLASSIFIED = "eval/done/unclassified"
EVAL_DONE_UNCLASSIFIED_RATE = "eval/done/unclassified/rate"
EVAL_REWARD_MEAN = "eval/reward/mean"
EVAL_REWARD_STD = "eval/reward/std"
EVAL_REWARD_MAX = "eval/reward/max"
EVAL_PROGRESS_X_MEAN = "eval/progress/x/mean"
EVAL_PROGRESS_X_MAX = "eval/progress/x/max"
EVAL_PROGRESS_LEVEL_X_MEAN = "eval/progress/level_x/mean"
EVAL_PROGRESS_LEVEL_X_MAX = "eval/progress/level_x/max"
EVAL_DEATH_COUNT = "eval/death/count"
EVAL_DEATH_RATE = "eval/death/rate"
EVAL_BEST_REWARD = "eval/best/reward"
EVAL_BEST_X = "eval/best/x"
EVAL_CHECKPOINT_STEP = "eval/checkpoint/step"
EVAL_CHECKPOINT_ARTIFACT = "eval/checkpoint/artifact"
EVAL_CONFIG_HUD_CROP_TOP = "eval/config/hud_crop_top"
EVAL_DURATION_SECONDS = "eval/duration/seconds"
EVAL_SOURCE = "eval/source"
EVAL_EPISODES = "eval/episodes"
EVAL_INFO_ROOT = "eval/info/"

CHECKPOINT_EVAL_CANDIDATE_PASS = "checkpoint_eval/candidate/pass"
CHECKPOINT_EVAL_CANDIDATE_STAGE_INDEX = "checkpoint_eval/candidate/stage_index"
CHECKPOINT_EVAL_CANDIDATE_CHECKPOINT_STEP = "checkpoint_eval/candidate/checkpoint_step"
CHECKPOINT_EVAL_CANDIDATE_EPISODES = "checkpoint_eval/candidate/episodes"

LEADER_CHECKPOINT_COMPLETION_RATE = "leader/checkpoint/completion_rate"
LEADER_CHECKPOINT_COMPLETION_RATE_MEAN = "leader/checkpoint/completion_rate_mean"
LEADER_CHECKPOINT_OBJECTIVE = "leader/checkpoint/objective"
LEADER_CHECKPOINT_OBJECTIVE_NAME = "leader/checkpoint/objective_name"
LEADER_CHECKPOINT_REWARD_MEAN = "leader/checkpoint/reward_mean"
LEADER_CHECKPOINT_BEST_REWARD = "leader/checkpoint/best_reward"
LEADER_CHECKPOINT_RANK = "leader/checkpoint/rank"
LEADER_CHECKPOINT_RANK_VALUES = "leader/checkpoint/rank_values"
LEADER_CHECKPOINT_MAX_X_MAX = "leader/checkpoint/max_x_max"
LEADER_CHECKPOINT_STEP = "leader/checkpoint/step"
LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL = "leader/checkpoint/steps_to_completion_goal"
LEADER_CHECKPOINT_ARTIFACT_REF = "leader/checkpoint/artifact_ref"
LEADER_CHECKPOINT_LOCAL_PATH = "leader/checkpoint/local_path"
LEADER_CHECKPOINT_EVAL_SOURCE = "leader/checkpoint/eval_source"
LEADER_CHECKPOINT_UPDATED_AT = "leader/checkpoint/updated_at"


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment)
    return segment.strip("_") or "unknown"


def stat_metric(prefix: str, stat: str) -> str:
    return f"{prefix}/{stat}"


def checkpoint_eval_stage_metric(stage_name: str, name: str) -> str:
    return f"checkpoint_eval/{stage_name}/{name}"


def staged_metric_name(stage_name: str, metric_name: str) -> str:
    if metric_name == GLOBAL_STEP:
        return metric_name
    return checkpoint_eval_stage_metric(stage_name, metric_name.removeprefix("eval/"))


def train_done_reason_metric(reason: object) -> str:
    return f"{TRAIN_DONE_ROOT}{metric_path_segment(reason)}"


def train_adv_task_metric(task_id: object, stat: str) -> str:
    return f"{TRAIN_ADV_ROOT}/task{metric_path_segment(task_id)}/{metric_path_segment(stat)}"


def metric_value_segment(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(metric_path_segment(item) for item in value) or "unknown"
    return metric_path_segment(value)


def train_done_value_metric(reason: object, direction: str, value: object) -> str:
    return f"{train_done_reason_metric(reason)}/{metric_path_segment(direction)}/{metric_value_segment(value)}"


def train_done_from_rate_metric(reason: object, stat: str) -> str:
    return f"{train_done_reason_metric(reason)}/from_rate/{metric_path_segment(stat)}"


def train_info_level_complete_from_metric(value: object) -> str:
    return f"{TRAIN_INFO_LEVEL_COMPLETE_ROOT}/from/{metric_value_segment(value)}"


def train_info_level_complete_count_metric(value: object) -> str:
    return f"{train_info_level_complete_from_metric(value)}/count"


def train_info_level_complete_attempts_metric(value: object) -> str:
    return f"{train_info_level_complete_from_metric(value)}/attempts"


def train_info_level_complete_current_rate_metric(value: object) -> str:
    return f"{train_info_level_complete_from_metric(value)}/rate/current"


def train_info_level_complete_rate_metric(value: object) -> str:
    return f"{train_info_level_complete_from_metric(value)}/rate"


def eval_done_reason_metric(reason: object) -> str:
    return f"eval/done/{metric_path_segment(reason)}"


def eval_done_value_metric(reason: object, direction: str, value: object) -> str:
    return f"{eval_done_reason_metric(reason)}/{metric_path_segment(direction)}/{metric_value_segment(value)}"
