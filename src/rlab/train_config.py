from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rlab.env import EnvConfig
from rlab.seeds import DEFAULT_TRAIN_SEED, EVAL_SEED_START


ADVANTAGE_NORMALIZATION_CHOICES = ("auto", "none", "global", "per-task")
REWARD_MODE_CHOICES = ("auto", "baseline", "bounded", "additive", "score", "native")
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
WANDB_MODE_CHOICES = ("online", "offline", "disabled")
EARLY_STOP_OPERATOR_CHOICES = (">", ">=", "<", "<=")

FieldKind = Literal["value", "store_true", "bool_optional"]
TypeName = Literal["str", "int", "float", "json", "obs_crop"]
SerializeMode = Literal["str", "json", "csv", "rows", "skip_nonpositive_float"]
SequenceItemKind = Literal["str", "number", "rows"]


@dataclass(frozen=True)
class TrainConfigField:
    dest: str
    flags: tuple[str, ...]
    false_flag: str | None = None
    kind: FieldKind = "value"
    type_name: TypeName = "str"
    default: Any = None
    env_default: str | None = None
    choices: tuple[str, ...] = ()
    help: str | None = None
    suppress_help: bool = False
    serialize: SerializeMode = "str"
    env_config_key: str | None = None
    queue_required: bool = False
    non_empty: bool = False
    validation_min: float | None = None
    validation_max: float | None = None
    sequence_items: SequenceItemKind | None = None
    allow_empty_sequence: bool = False
    mapping_value: bool = False

    @property
    def command_flag(self) -> str:
        return self.flags[0]

    @property
    def command_false_flag(self) -> str:
        return self.false_flag or f"--no-{self.command_flag.removeprefix('--')}"

    @property
    def is_env_config_field(self) -> bool:
        return self.env_config_key is not None


def _env_default(env_defaults: EnvConfig, field: TrainConfigField) -> Any:
    if field.env_default is None:
        return field.default
    return getattr(env_defaults, field.env_default)


def _type_callable(
    field: TrainConfigField,
    *,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> Callable[[Any], Any] | type | None:
    if field.kind != "value":
        return None
    if field.type_name == "int":
        return int
    if field.type_name == "float":
        return float
    if field.type_name == "json":
        return parse_json_value
    if field.type_name == "obs_crop":
        return parse_obs_crop
    return None


def add_train_config_args(
    parser: argparse.ArgumentParser,
    *,
    env_defaults: EnvConfig | None = None,
    preset_choices: Sequence[str] = (),
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> None:
    env_defaults = env_defaults or EnvConfig()
    for field in TRAIN_CONFIG_FIELDS:
        kwargs: dict[str, Any] = {"dest": field.dest}
        if field.suppress_help:
            kwargs["help"] = argparse.SUPPRESS
        elif field.help is not None:
            kwargs["help"] = field.help
        if field.dest == "preset" and preset_choices:
            kwargs["choices"] = sorted(preset_choices)
        elif field.choices:
            kwargs["choices"] = field.choices
        if field.kind == "store_true":
            kwargs["action"] = "store_true"
        elif field.kind == "bool_optional":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            type_callable = _type_callable(
                field,
                parse_json_value=parse_json_value,
                parse_obs_crop=parse_obs_crop,
            )
            if type_callable is not None:
                kwargs["type"] = type_callable
        kwargs["default"] = _env_default(env_defaults, field)
        option_flags = field.flags[:1] if field.kind == "bool_optional" else field.flags
        parser.add_argument(*option_flags, **kwargs)


def _serialize_value(field: TrainConfigField, value: Any) -> str | None:
    if value is None or value == "":
        return None
    if field.serialize == "skip_nonpositive_float" and float(value) <= 0:
        return None
    if field.serialize == "json" and isinstance(value, Mapping | list | tuple):
        return json.dumps(value, separators=(",", ":"))
    if field.serialize == "rows" and isinstance(value, list | tuple):
        return ";".join(
            ",".join(str(item) for item in row) if isinstance(row, list | tuple) else str(row)
            for row in value
        )
    if field.serialize == "csv" and isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def build_train_command_from_fields(options: Mapping[str, Any]) -> list[str]:
    cmd = ["rlab", "train", "local"]
    for field in TRAIN_CONFIG_FIELDS:
        if field.dest not in options:
            continue
        value = options[field.dest]
        if field.kind == "store_true":
            if value:
                cmd.append(field.command_flag)
            continue
        if field.kind == "bool_optional":
            if value is True:
                cmd.append(field.flags[0])
            elif value is False:
                cmd.append(field.command_false_flag)
            continue
        serialized = _serialize_value(field, value)
        if serialized is None:
            continue
        cmd.extend([field.command_flag, serialized])
    return cmd


def train_config_field_names() -> frozenset[str]:
    return frozenset(field.dest for field in TRAIN_CONFIG_FIELDS)


def train_config_field_for_key(key: str, *, include_env_aliases: bool = True) -> TrainConfigField | None:
    for field in TRAIN_CONFIG_FIELDS:
        if field.dest == key or (include_env_aliases and field.env_config_key == key):
            return field
    return None


def queue_required_train_config_fields() -> tuple[str, ...]:
    return tuple(field.dest for field in TRAIN_CONFIG_FIELDS if field.queue_required)


def env_config_arg_fields() -> tuple[TrainConfigField, ...]:
    return tuple(field for field in TRAIN_CONFIG_FIELDS if field.is_env_config_field)


def env_config_allowed_keys() -> frozenset[str]:
    keys: set[str] = set()
    for field in env_config_arg_fields():
        keys.add(field.dest)
        if field.env_config_key:
            keys.add(field.env_config_key)
    return frozenset(keys)


def _label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_number_bounds(
    *,
    key: str,
    label: str,
    number: float,
    minimum: float | None,
    maximum: float | None,
) -> None:
    if minimum is not None and number < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{_label_path(label, key)} must be <= {maximum:g}")


def _validate_string_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-empty string")


def _validate_number_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a number")


def _validate_row_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or not row:
            raise ValueError(f"{_label_path(label, key)}[{row_index}] must be a non-empty list")
        for column_index, item in enumerate(row):
            if not isinstance(item, int | str) or isinstance(item, bool):
                raise ValueError(
                    f"{_label_path(label, key)}[{row_index}][{column_index}] "
                    "must be an integer or string"
                )


def _validate_obs_crop_value(*, key: str, label: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{_label_path(label, key)} must be [top, right, bottom, left]")
    for index, item in enumerate(value):
        if not _is_int(item) or item < 0:
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-negative integer")


def validate_train_config_value(
    key: str,
    value: Any,
    *,
    label: str = "train_config",
) -> None:
    field = train_config_field_for_key(key)
    if field is None:
        raise ValueError(f"{_label_path(label, key)} is not a known train config field")
    if field.sequence_items == "str":
        _validate_string_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.sequence_items == "number":
        _validate_number_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.sequence_items == "rows":
        _validate_row_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.mapping_value:
        if not isinstance(value, Mapping):
            raise ValueError(f"{_label_path(label, key)} must be an object")
        return
    if field.kind in {"store_true", "bool_optional"}:
        if not isinstance(value, bool):
            raise ValueError(f"{_label_path(label, key)} must be a boolean")
        return
    if field.type_name == "int":
        if not _is_int(value):
            raise ValueError(f"{_label_path(label, key)} must be an integer")
        _validate_number_bounds(
            key=key,
            label=label,
            number=float(value),
            minimum=field.validation_min,
            maximum=field.validation_max,
        )
        return
    if field.type_name == "float":
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{_label_path(label, key)} must be a number")
        _validate_number_bounds(
            key=key,
            label=label,
            number=float(value),
            minimum=field.validation_min,
            maximum=field.validation_max,
        )
        return
    if field.type_name == "obs_crop":
        _validate_obs_crop_value(key=key, label=label, value=value)
        return
    if field.type_name == "json":
        return
    if not isinstance(value, str):
        raise ValueError(f"{_label_path(label, key)} must be a string")
    if field.non_empty and not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    if field.choices and value not in field.choices:
        choices = ", ".join(field.choices)
        raise ValueError(f"{_label_path(label, key)} must be one of {choices}")


def validate_train_config_fields(
    train_config: Mapping[str, Any],
    *,
    label: str = "train_config",
    keys: Sequence[str] | None = None,
    required_keys: Sequence[str] = (),
) -> None:
    missing = [key for key in required_keys if key not in train_config]
    if missing:
        raise ValueError(f"{label} missing required field(s): {', '.join(missing)}")
    selected_keys = tuple(keys) if keys is not None else tuple(train_config)
    for key in selected_keys:
        if key in train_config:
            validate_train_config_value(key, train_config[key], label=label)


TRAIN_CONFIG_FIELDS: tuple[TrainConfigField, ...] = (
    TrainConfigField("preset", ("--preset",), default=None),
    TrainConfigField(
        "timesteps",
        ("--timesteps",),
        type_name="int",
        default=1_000_000,
        queue_required=True,
        validation_min=1,
    ),
    TrainConfigField("n_envs", ("--n-envs",), type_name="int", default=8, validation_min=1),
    TrainConfigField(
        "env_threads",
        ("--env-threads",),
        type_name="int",
        default=0,
        env_config_key="env_threads",
        validation_min=0,
        help="Native stable-retro env threads; <=0 keeps min(n_envs, 16).",
    ),
    TrainConfigField(
        "torch_num_threads",
        ("--torch-num-threads",),
        type_name="int",
        default=0,
        help="PyTorch CPU intra-op threads; <=0 leaves the torch default.",
    ),
    TrainConfigField(
        "seed",
        ("--seed",),
        type_name="int",
        default=DEFAULT_TRAIN_SEED,
        help=(
            "Training base seed. The base seed plus vector env slots must stay below "
            f"{EVAL_SEED_START}; seeds >= {EVAL_SEED_START} are reserved for eval."
        ),
    ),
    TrainConfigField("run_name", ("--run-name",), default="ppo_retro"),
    TrainConfigField(
        "run_description",
        ("--run-description",),
        default="",
        help="Human-readable description of the experiment or ablation being run.",
    ),
    TrainConfigField("runs_dir", ("--runs-dir",), default="runs"),
    TrainConfigField(
        "env_provider",
        ("--env-provider",),
        env_default="env_provider",
        env_config_key="env_provider",
        non_empty=True,
        help="Environment provider id. Supported: stable-retro-turbo, supermariobrosnes-turbo, ale-py.",
    ),
    TrainConfigField(
        "game",
        ("--game",),
        env_default="game",
        env_config_key="game",
        queue_required=True,
        non_empty=True,
        help="Provider game id. Defaults to RETRO_GAME when set.",
    ),
    TrainConfigField(
        "state",
        ("--state",),
        env_default="state",
        env_config_key="state",
        non_empty=True,
        help="Provider state. If omitted, registered targets may provide a default.",
    ),
    TrainConfigField(
        "states",
        ("--states",),
        default="",
        serialize="csv",
        env_config_key="states",
        sequence_items="str",
        help="Comma-separated provider states. Without --state-probs, provide exactly one state per env slot in order.",
    ),
    TrainConfigField(
        "state_probs",
        ("--state-probs",),
        default="",
        serialize="csv",
        env_config_key="state_probs",
        sequence_items="number",
        help="Comma-separated non-negative sampling weights for --states. The native vector env normalizes weights and samples independently on each episode reset.",
    ),
    TrainConfigField(
        "info_events_json",
        ("--info-events-json",),
        default="",
        serialize="json",
        env_config_key="info_events",
        mapping_value=True,
        help="JSON object mapping event names to [key_or_keys, op]. Events are observed without ending episodes unless also listed in --done-on-events.",
    ),
    TrainConfigField(
        "done_on_events",
        ("--done-on-events",),
        default="",
        serialize="csv",
        env_config_key="done_on_events",
        sequence_items="str",
        allow_empty_sequence=True,
        help="Comma-separated info event names that should terminate the current episode.",
    ),
    TrainConfigField(
        "task_conditioning",
        ("--task-conditioning",),
        kind="store_true",
        default=False,
        env_config_key="task_conditioning",
        help="Use SB3 MultiInputPolicy with a one-hot task vector derived from the native active state for each env lane.",
    ),
    TrainConfigField(
        "task_conditioning_info_vars",
        ("--task-conditioning-info-vars",),
        default="",
        serialize="csv",
        env_config_key="task_conditioning_info_vars",
        sequence_items="str",
        allow_empty_sequence=True,
        help="Comma-separated info keys used to map task-conditioned one-hot vectors.",
    ),
    TrainConfigField(
        "task_conditioning_info_values",
        ("--task-conditioning-info-values",),
        default="",
        serialize="rows",
        env_config_key="task_conditioning_info_values",
        sequence_items="rows",
        allow_empty_sequence=True,
        help="Semicolon-separated info-value rows for task conditioning, for example '0,0;0,1'. Omit when values can be derived from states.",
    ),
    TrainConfigField(
        "frame_skip",
        ("--frame-skip",),
        type_name="int",
        default=4,
        env_config_key="frame_skip",
        validation_min=1,
    ),
    TrainConfigField(
        "sticky_action_prob",
        ("--sticky-action-prob",),
        type_name="float",
        env_default="sticky_action_prob",
        env_config_key="sticky_action_prob",
        help="Probability of replaying the previous high-level action; 0 disables sticky actions.",
    ),
    TrainConfigField(
        "max_pool_frames",
        ("--max-pool-frames",),
        false_flag="--no-max-pool-frames",
        kind="bool_optional",
        default=True,
        env_config_key="max_pool_frames",
        help="Max-pool over the last two raw frames inside each frame-skip step.",
    ),
    TrainConfigField(
        "max_episode_steps",
        ("--max-episode-steps",),
        type_name="int",
        default=4500,
        env_config_key="max_episode_steps",
        validation_min=0,
    ),
    TrainConfigField(
        "observation_size",
        ("--observation-size",),
        type_name="int",
        env_default="observation_size",
        env_config_key="observation_size",
        validation_min=0,
    ),
    TrainConfigField(
        "hud_crop_top",
        ("--hud-crop-top",),
        type_name="int",
        env_default="hud_crop_top",
        env_config_key="hud_crop_top",
        validation_min=0,
        help="Crop this many pixels from the top of raw frames before grayscale resize; -1 uses the target default.",
    ),
    TrainConfigField(
        "obs_crop",
        ("--obs-crop",),
        type_name="obs_crop",
        env_default="obs_crop",
        serialize="csv",
        env_config_key="obs_crop",
        help="Four-sided raw-frame crop as top,right,bottom,left before grayscale resize.",
    ),
    TrainConfigField(
        "obs_crop_mode",
        ("--obs-crop-mode",),
        env_default="obs_crop_mode",
        choices=("remove", "mask"),
        env_config_key="obs_crop_mode",
        non_empty=True,
        help="Whether obs_crop removes pixels or masks them before resize.",
    ),
    TrainConfigField(
        "obs_crop_fill",
        ("--obs-crop-fill",),
        type_name="int",
        env_default="obs_crop_fill",
        env_config_key="obs_crop_fill",
        validation_min=0,
        validation_max=255,
        help="Pixel fill value for obs_crop_mode=mask.",
    ),
    TrainConfigField(
        "obs_resize_algorithm",
        ("--obs-resize-algorithm",),
        env_default="obs_resize_algorithm",
        env_config_key="obs_resize_algorithm",
        non_empty=True,
        help="Resize algorithm for native frame preprocessing.",
    ),
    TrainConfigField(
        "checkpoint_freq", ("--checkpoint-freq",), type_name="int", default=500_000, validation_min=0
    ),
    TrainConfigField(
        "post_train_eval",
        ("--post-train-eval",),
        false_flag="--no-post-train-eval",
        kind="bool_optional",
        default=True,
        help="Evaluate checkpoint artifacts in this training process after learning finishes.",
    ),
    TrainConfigField(
        "post_train_eval_episodes",
        ("--post-train-eval-episodes",),
        type_name="int",
        default=100,
        help="Episodes per checkpoint for post-training checkpoint eval.",
    ),
    TrainConfigField(
        "post_train_eval_n_envs",
        ("--post-train-eval-n-envs",),
        type_name="int",
        default=20,
        help="Vector eval env count for post-training checkpoint eval.",
    ),
    TrainConfigField(
        "post_train_eval_max_steps",
        ("--post-train-eval-max-steps",),
        type_name="int",
        default=0,
        help="Max steps per post-training eval episode; <=0 uses --max-episode-steps.",
    ),
    TrainConfigField(
        "post_train_eval_stochastic",
        ("--post-train-eval-stochastic",),
        false_flag="--no-post-train-eval-stochastic",
        kind="bool_optional",
        default=True,
        help="Use stochastic policy sampling for post-training checkpoint eval.",
    ),
    TrainConfigField(
        "early_stop",
        ("--early-stop",),
        type_name="json",
        default=None,
        serialize="json",
        help="JSON early-stop list of AND-combined metric threshold rules.",
    ),
    TrainConfigField(
        "early_stop_metric",
        ("--early-stop-metric",),
        default="",
        help="Training metric key that can stop training once it crosses --early-stop-threshold.",
    ),
    TrainConfigField(
        "early_stop_threshold",
        ("--early-stop-threshold",),
        type_name="float",
        default=None,
        help="Numeric threshold for --early-stop-metric. Provide both or neither.",
    ),
    TrainConfigField(
        "early_stop_operator",
        ("--early-stop-operator",),
        default=">=",
        choices=EARLY_STOP_OPERATOR_CHOICES,
        help="Comparison for the early-stop metric threshold.",
    ),
    TrainConfigField("learning_rate", ("--learning-rate",), type_name="float", default=1e-4),
    TrainConfigField(
        "learning_rate_final",
        ("--learning-rate-final",),
        type_name="float",
        default=None,
        help="If set, linearly decay learning rate from --learning-rate to this value over training.",
    ),
    TrainConfigField(
        "learning_rate_schedule_timesteps",
        ("--learning-rate-schedule-timesteps",),
        type_name="int",
        default=0,
        help="Timesteps over which to decay learning rate; <=0 decays over --timesteps.",
    ),
    TrainConfigField("n_steps", ("--n-steps",), type_name="int", default=512),
    TrainConfigField("batch_size", ("--batch-size",), type_name="int", default=256),
    TrainConfigField("n_epochs", ("--n-epochs",), type_name="int", default=10),
    TrainConfigField("device", ("--device",), default="auto", choices=DEVICE_CHOICES, non_empty=True),
    TrainConfigField("gamma", ("--gamma",), type_name="float", default=0.9),
    TrainConfigField("gae_lambda", ("--gae-lambda",), type_name="float", default=1.0),
    TrainConfigField("ent_coef", ("--ent-coef",), type_name="float", default=0.01),
    TrainConfigField(
        "ent_coef_final",
        ("--ent-coef-final",),
        type_name="float",
        default=None,
        help="If set, linearly decay entropy coefficient from --ent-coef to this value.",
    ),
    TrainConfigField(
        "ent_coef_schedule_timesteps",
        ("--ent-coef-schedule-timesteps",),
        type_name="int",
        default=0,
        help="Timesteps over which to decay entropy coefficient; <=0 decays over --timesteps.",
    ),
    TrainConfigField("vf_coef", ("--vf-coef",), type_name="float", default=1.0),
    TrainConfigField("clip_range", ("--clip-range",), type_name="float", default=0.2),
    TrainConfigField(
        "clip_range_vf",
        ("--clip-range-vf",),
        type_name="float",
        default=None,
        help="Optional PPO value-function clipping range; omitted keeps SB3 default.",
    ),
    TrainConfigField(
        "policy_net_arch",
        ("--policy-net-arch",),
        default="",
        help="Comma-separated policy MLP hidden sizes after the CNN/combined extractor.",
    ),
    TrainConfigField(
        "value_net_arch",
        ("--value-net-arch",),
        default="",
        help="Comma-separated value MLP hidden sizes after the CNN/combined extractor.",
    ),
    TrainConfigField(
        "normalize_advantage",
        ("--normalize-advantage",),
        false_flag="--no-normalize-advantage",
        kind="bool_optional",
        default=False,
        help="Normalize PPO advantages before policy updates.",
    ),
    TrainConfigField(
        "advantage_normalization",
        ("--advantage-normalization",),
        default="auto",
        choices=ADVANTAGE_NORMALIZATION_CHOICES,
        non_empty=True,
        help="PPO advantage normalization mode. auto preserves --normalize-advantage; per-task normalizes each task-conditioned rollout slice once before PPO epochs.",
    ),
    TrainConfigField("adam_eps", ("--adam-eps",), type_name="float", default=1e-8),
    TrainConfigField(
        "target_kl",
        ("--target-kl",),
        type_name="float",
        default=None,
        serialize="skip_nonpositive_float",
    ),
    TrainConfigField(
        "use_retro_reward",
        ("--use-retro-reward",),
        kind="store_true",
        default=False,
        env_config_key="use_retro_reward",
    ),
    TrainConfigField(
        "clip_rewards",
        ("--clip-rewards",),
        kind="store_true",
        default=False,
        env_config_key="clip_rewards",
    ),
    TrainConfigField(
        "reward_mode",
        ("--reward-mode",),
        env_default="reward_mode",
        choices=REWARD_MODE_CHOICES,
        env_config_key="reward_mode",
        non_empty=True,
        help="Target reward mode. Use native for unknown games without a custom target tracker.",
    ),
    TrainConfigField(
        "progress_reward_cap",
        ("--progress-reward-cap",),
        type_name="float",
        default=30.0,
        env_config_key="progress_reward_cap",
    ),
    TrainConfigField(
        "progress_reward_scale",
        ("--progress-reward-scale",),
        type_name="float",
        default=1.0,
        env_config_key="progress_reward_scale",
    ),
    TrainConfigField(
        "terminal_reward",
        ("--terminal-reward",),
        type_name="float",
        default=50.0,
        env_config_key="terminal_reward",
    ),
    TrainConfigField(
        "reward_scale",
        ("--reward-scale",),
        type_name="float",
        default=10.0,
        env_config_key="reward_scale",
    ),
    TrainConfigField(
        "time_penalty",
        ("--time-penalty",),
        type_name="float",
        default=0.0,
        env_config_key="time_penalty",
    ),
    TrainConfigField(
        "death_penalty",
        ("--death-penalty",),
        type_name="float",
        default=25.0,
        env_config_key="death_penalty",
    ),
    TrainConfigField(
        "completion_reward",
        ("--completion-reward",),
        type_name="float",
        default=0.0,
        env_config_key="completion_reward",
    ),
    TrainConfigField(
        "score_progress_clipped",
        ("--score-progress-clipped",),
        kind="store_true",
        default=False,
        env_config_key="score_progress_clipped",
        help="In score reward mode, use clipped progress_reward instead of raw progress_delta.",
    ),
    TrainConfigField(
        "env_wrappers",
        ("--env-wrappers",),
        type_name="json",
        env_default="env_wrappers",
        serialize="json",
        env_config_key="env_wrappers",
        suppress_help=True,
    ),
    TrainConfigField(
        "vec_wrappers",
        ("--vec-wrappers",),
        type_name="json",
        env_default="vec_wrappers",
        serialize="json",
        env_config_key="vec_wrappers",
        suppress_help=True,
    ),
    TrainConfigField(
        "no_progress_timeout_steps",
        ("--no-progress-timeout-steps",),
        type_name="int",
        default=0,
        env_config_key="no_progress_timeout_steps",
        validation_min=0,
        help="Truncate an episode after this many env steps without new x progress; <=0 disables.",
    ),
    TrainConfigField(
        "no_progress_min_delta",
        ("--no-progress-min-delta",),
        type_name="int",
        default=0,
        env_config_key="no_progress_min_delta",
        validation_min=0,
        help="Minimum progress_delta that resets the no-progress timeout.",
    ),
    TrainConfigField(
        "action_set",
        ("--action-set",),
        env_default="action_set",
        env_config_key="action_set",
        non_empty=True,
        help="Target-specific action set name, native, or auto for the target default.",
    ),
    TrainConfigField(
        "resume", ("--resume",), default=None, help="Path to an existing PPO .zip checkpoint"
    ),
    TrainConfigField(
        "wandb",
        ("--wandb",),
        kind="store_true",
        default=False,
        queue_required=True,
        help="Log training to Weights & Biases",
    ),
    TrainConfigField("wandb_project", ("--wandb-project",), default=None),
    TrainConfigField("wandb_entity", ("--wandb-entity",), default=None),
    TrainConfigField("wandb_group", ("--wandb-group",), default=None),
    TrainConfigField("wandb_tags", ("--wandb-tags",), default="", help="Comma-separated W&B tags"),
    TrainConfigField(
        "wandb_mode",
        ("--wandb-mode",),
        default="online",
        choices=WANDB_MODE_CHOICES,
        queue_required=True,
        non_empty=True,
    ),
    TrainConfigField(
        "runtime_image_ref",
        ("--runtime-image-ref",),
        default="",
        help="Immutable runtime image ref recorded as run metadata; does not affect training.",
    ),
    TrainConfigField(
        "run_target",
        ("--run-target",),
        default="",
        help="Canonical compute target recorded as run metadata; does not affect training.",
    ),
    TrainConfigField(
        "goal_slug", ("--goal-slug",), default="", help="Research goal slug recorded in W&B config."
    ),
    TrainConfigField(
        "recipe_slug",
        ("--recipe-slug",),
        default="",
        help="Experiment recipe slug recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_path",
        ("--recipe-path",),
        default="",
        help="Experiment recipe path recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_overrides",
        ("--recipe-overrides",),
        type_name="json",
        default=(),
        serialize="json",
        help="Hydra/OmegaConf dotlist overrides applied to the checked-in recipe.",
    ),
    TrainConfigField(
        "queue_train_job_id",
        ("--queue-train-job-id",),
        type_name="int",
        default=0,
        help="Queue train job id recorded in W&B config; 0 means local/unqueued.",
    ),
    TrainConfigField(
        "no_wandb_artifacts",
        ("--no-wandb-artifacts",),
        kind="store_true",
        default=False,
        help="Disable W&B model uploads",
    ),
    TrainConfigField(
        "wandb_artifact_storage_uri",
        ("--wandb-artifact-storage-uri",),
        default="",
        queue_required=True,
        help="Optional s3://bucket/prefix base URI for model artifacts. Model zips are stored under <game-id>/... below that URI, and W&B logs reference artifacts instead of storing file bytes.",
    ),
)
