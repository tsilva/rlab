from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rlab.env import EnvConfig
from rlab.modal_eval_protocol import SEED_PROTOCOL
from rlab.provider_config import provider_num_envs
from rlab.seeds import DEFAULT_TRAIN_SEED, EVAL_SEED_START, validate_training_seed
from rlab.validation import normalize_obs_crop


WANDB_MODE_CHOICES = ("online", "offline", "disabled")

FieldKind = Literal["value", "store_true", "bool_optional"]
TypeName = Literal["str", "int", "float", "json", "obs_crop"]
SerializeMode = Literal["str", "json", "csv", "rows", "skip_nonpositive_float"]
SequenceItemKind = Literal["str", "number", "rows"]
FieldOwner = Literal["runtime", "goal_environment", "goal_objective"]
SourceSection = Literal["runtime", "train", "goal_train"]


@dataclass(frozen=True)
class TrainConfigField:
    dest: str
    flag: str
    kind: FieldKind = "value"
    type_name: TypeName = "str"
    default: Any = None
    env_default: str | None = None
    choices: tuple[str, ...] = ()
    help: str | None = None
    serialize: SerializeMode = "str"
    environment: bool = False
    queue_required: bool = False
    non_empty: bool = False
    validation_min: float | None = None
    validation_max: float | None = None
    sequence_items: SequenceItemKind | None = None
    allow_empty_sequence: bool = False
    mapping_value: bool = False
    owner: FieldOwner = "runtime"
    source_section: SourceSection = "runtime"
    cli_exposed: bool = True

    @property
    def command_flag(self) -> str:
        return self.flag

    @property
    def is_env_config_field(self) -> bool:
        return self.environment


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


def _add_config_field_argument(
    parser: argparse.ArgumentParser,
    field: TrainConfigField,
    *,
    default: Any,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
    dest: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {"dest": dest or field.dest, "default": default}
    if field.help is not None:
        kwargs["help"] = field.help
    if field.choices:
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
    parser.add_argument(field.flag, **kwargs)


def add_train_config_args(
    parser: argparse.ArgumentParser,
    *,
    env_defaults: EnvConfig | None = None,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> None:
    env_defaults = env_defaults or EnvConfig()
    for field in TRAIN_CONFIG_FIELDS:
        if not field.cli_exposed:
            parser.set_defaults(**{field.dest: _env_default(env_defaults, field)})
            continue
        _add_config_field_argument(
            parser,
            field,
            default=_env_default(env_defaults, field),
            parse_json_value=parse_json_value,
            parse_obs_crop=parse_obs_crop,
        )


def add_env_config_args(
    parser: argparse.ArgumentParser,
    *,
    max_steps_default: int,
    defaults: EnvConfig | None = None,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> None:
    defaults = defaults or EnvConfig()
    for field in env_config_arg_fields():
        dest = field.dest
        default = _env_default(defaults, field)
        _add_config_field_argument(
            parser,
            field,
            default=default,
            parse_json_value=parse_json_value,
            parse_obs_crop=parse_obs_crop,
            dest=dest,
        )
    parser.add_argument("--max-steps", type=int, default=max_steps_default)


def load_materialized_train_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"train config file must contain a JSON object: {path}")
    return validate_and_normalize_train_config(
        payload,
        label=f"train config file {path}",
        required_keys=("training_backend",),
    )


def materialized_train_args(path: Path) -> argparse.Namespace:
    """Load queue/runtime JSON without routing an internal payload through CLI parsing."""

    payload = load_materialized_train_config(path)
    defaults = {field.dest: _env_default(EnvConfig(), field) for field in TRAIN_CONFIG_FIELDS}
    defaults.update(payload)
    args = argparse.Namespace(**defaults)
    apply_training_backend_arg_view(args, payload)
    if isinstance(defaults.get("wandb_tags"), list | tuple):
        args.wandb_tags = ",".join(str(tag) for tag in defaults["wandb_tags"])
    args.train_config_json = path
    args._train_config_json_fields = set(payload)
    args._explicit_train_arg_dests = set()
    args._materialized_train_config = payload
    validate_training_seed(
        args.seed,
        label="train_config.seed",
        seed_span=provider_num_envs(args, explicit_n_envs=payload.get("n_envs")),
    )
    return args


def apply_training_backend_arg_view(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
) -> None:
    from rlab.training_backend import (
        training_backend_config,
        training_backend_config_hash,
        training_backend_id,
    )

    backend_config = training_backend_config(payload)
    collisions = sorted(set(vars(args)) & set(backend_config))
    if collisions:
        raise ValueError(
            "training_backend.config collides with common train fields: " + ", ".join(collisions)
        )
    for key, value in backend_config.items():
        setattr(args, key, value)
    args.training_backend_id = training_backend_id(payload)
    args.training_backend_config = backend_config
    args.training_backend_config_hash = training_backend_config_hash(payload)


def train_config_field_for_key(key: str) -> TrainConfigField | None:
    for field in TRAIN_CONFIG_FIELDS:
        if field.dest == key:
            return field
    return None


def queue_required_train_config_fields() -> tuple[str, ...]:
    return tuple(field.dest for field in TRAIN_CONFIG_FIELDS if field.queue_required)


def env_config_arg_fields() -> tuple[TrainConfigField, ...]:
    return tuple(field for field in TRAIN_CONFIG_FIELDS if field.is_env_config_field)


def env_config_allowed_keys() -> frozenset[str]:
    return frozenset(field.dest for field in env_config_arg_fields())


def playback_env_arg_keys() -> dict[str, tuple[str, ...]]:
    return {field.dest: (field.dest,) for field in env_config_arg_fields()}


def train_config_keys_owned_by(owner: FieldOwner) -> frozenset[str]:
    keys: set[str] = set()
    for field in TRAIN_CONFIG_FIELDS:
        is_owned = field.owner == owner or (
            owner == "goal_environment" and field.is_env_config_field
        )
        if not is_owned:
            continue
        keys.add(field.dest)
    return frozenset(keys)


def train_config_keys_in_source_section(section: SourceSection) -> frozenset[str]:
    return frozenset(field.dest for field in TRAIN_CONFIG_FIELDS if field.source_section == section)


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
    normalize_obs_crop(value, label=_label_path(label, key))


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


def validate_and_normalize_train_config(
    train_config: Mapping[str, Any],
    *,
    label: str = "train_config",
    required_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate one flat train config and normalize its structured rule fields."""

    from rlab.checkpoint_eval_config import normalize_checkpoint_eval_stages
    from rlab.early_stop import normalize_early_stop_config

    normalized = dict(train_config)
    validate_train_config_fields(normalized, label=label, required_keys=required_keys)
    if normalized.get("post_train_eval_stochastic") is False:
        raise ValueError(
            f"{label}.post_train_eval_stochastic must be true; "
            "all policy evaluation uses stochastic sampling"
        )
    if normalized.get("early_stop") is not None:
        normalized["early_stop"] = normalize_early_stop_config(
            normalized["early_stop"], label=f"{label}.early_stop"
        )
    if normalized.get("checkpoint_eval_stages") is not None:
        if not (
            normalized.get("checkpoint_eval_backend") == "none"
            and normalized.get("checkpoint_eval_stages") == []
        ):
            normalized["checkpoint_eval_stages"] = normalize_checkpoint_eval_stages(
                normalized["checkpoint_eval_stages"],
                label=f"{label}.checkpoint_eval_stages",
            )
    if normalized.get("checkpoint_eval_backend") == "none":
        if normalized.get("early_stop") is not None:
            raise ValueError(f"{label}.early_stop must be null when checkpoint eval is disabled")
        if normalized.get("checkpoint_eval_stages"):
            raise ValueError(
                f"{label}.checkpoint_eval_stages must be empty when checkpoint eval is disabled"
            )
    if "training_backend" in normalized:
        from rlab.training_backend import normalize_training_backend

        common_config = {
            key: value for key, value in normalized.items() if key != "training_backend"
        }
        normalized["training_backend"] = normalize_training_backend(
            normalized["training_backend"],
            common_config=common_config,
            label=f"{label}.training_backend",
        )
    return normalized


TRAIN_CONFIG_FIELDS: tuple[TrainConfigField, ...] = (
    TrainConfigField(
        "timesteps",
        "--timesteps",
        type_name="int",
        default=1_000_000,
        queue_required=True,
        validation_min=1,
        source_section="train",
    ),
    TrainConfigField(
        "training_backend",
        "--training-backend",
        type_name="json",
        default=None,
        serialize="json",
        mapping_value=True,
        queue_required=True,
        help="Selected training backend id and backend-local configuration.",
    ),
    TrainConfigField(
        "n_envs",
        "--n-envs",
        type_name="int",
        default=8,
        validation_min=1,
        owner="goal_environment",
    ),
    TrainConfigField(
        "seed",
        "--seed",
        type_name="int",
        default=DEFAULT_TRAIN_SEED,
        help=(
            "Training base seed. The base seed plus vector env slots must stay below "
            f"{EVAL_SEED_START}; seeds >= {EVAL_SEED_START} are reserved for eval."
        ),
    ),
    TrainConfigField("run_name", "--run-name", default="ppo_retro"),
    TrainConfigField(
        "run_description",
        "--run-description",
        default="",
        help="Human-readable description of the experiment or ablation being run.",
    ),
    TrainConfigField("runs_dir", "--runs-dir", default="runs"),
    TrainConfigField(
        "env_provider",
        "--env-provider",
        env_default="env_provider",
        environment=True,
        non_empty=True,
        help=(
            "Environment provider id. Supported: rlab, stable-retro-turbo, "
            "supermariobrosnes-turbo, ale-py, gymnasium."
        ),
    ),
    TrainConfigField(
        "game",
        "--game",
        env_default="game",
        environment=True,
        queue_required=True,
        non_empty=True,
        help="Provider game id. Defaults to RETRO_GAME when set.",
    ),
    TrainConfigField(
        "env_args",
        "--env-args",
        type_name="json",
        default={},
        env_default="env_args",
        serialize="json",
        environment=True,
        mapping_value=True,
        help="Provider-native environment constructor arguments, serialized as a JSON object.",
    ),
    TrainConfigField(
        "task",
        "--task-json",
        type_name="json",
        default={},
        env_default="task",
        serialize="json",
        environment=True,
        mapping_value=True,
        help="Canonical bound-task definition as a JSON object.",
    ),
    TrainConfigField(
        "state",
        "--state",
        env_default="state",
        environment=True,
        non_empty=True,
        help="Provider state. If omitted, registered targets may provide a default.",
    ),
    TrainConfigField(
        "states",
        "--states",
        default="",
        serialize="csv",
        environment=True,
        sequence_items="str",
        help="Comma-separated provider states. Without --state-probs, provide exactly one state per env slot in order.",
    ),
    TrainConfigField(
        "state_probs",
        "--state-probs",
        default="",
        serialize="csv",
        environment=True,
        sequence_items="number",
        help="Comma-separated non-negative sampling weights for --states. The native vector env normalizes weights and samples independently on each episode reset.",
    ),
    TrainConfigField(
        "frame_skip",
        "--frame-skip",
        type_name="int",
        default=4,
        environment=True,
        validation_min=1,
    ),
    TrainConfigField(
        "sticky_action_prob",
        "--sticky-action-prob",
        type_name="float",
        env_default="sticky_action_prob",
        environment=True,
        help="Probability of replaying the previous high-level action; 0 disables sticky actions.",
    ),
    TrainConfigField(
        "max_pool_frames",
        "--max-pool-frames",
        kind="bool_optional",
        default=True,
        environment=True,
        help="Max-pool over the last two raw frames inside each frame-skip step.",
    ),
    TrainConfigField(
        "observation_size",
        "--observation-size",
        type_name="int",
        env_default="observation_size",
        environment=True,
        validation_min=0,
    ),
    TrainConfigField(
        "hud_crop_top",
        "--hud-crop-top",
        type_name="int",
        env_default="hud_crop_top",
        environment=True,
        validation_min=-1,
        help="Crop this many pixels from the top of raw frames before grayscale resize; -1 uses the target default.",
    ),
    TrainConfigField(
        "obs_crop",
        "--obs-crop",
        type_name="obs_crop",
        env_default="obs_crop",
        serialize="csv",
        environment=True,
        help="Four-sided raw-frame crop as top,right,bottom,left before grayscale resize.",
    ),
    TrainConfigField(
        "obs_crop_mode",
        "--obs-crop-mode",
        env_default="obs_crop_mode",
        choices=("remove", "mask"),
        environment=True,
        non_empty=True,
        help="Whether obs_crop removes pixels or masks them before resize.",
    ),
    TrainConfigField(
        "obs_crop_fill",
        "--obs-crop-fill",
        type_name="int",
        env_default="obs_crop_fill",
        environment=True,
        validation_min=0,
        validation_max=255,
        help="Pixel fill value for obs_crop_mode=mask.",
    ),
    TrainConfigField(
        "obs_resize_algorithm",
        "--obs-resize-algorithm",
        env_default="obs_resize_algorithm",
        environment=True,
        non_empty=True,
        help="Resize algorithm for native frame preprocessing.",
    ),
    TrainConfigField(
        "checkpoint_freq",
        "--checkpoint-freq",
        type_name="int",
        default=500_000,
        validation_min=0,
        source_section="goal_train",
    ),
    TrainConfigField(
        "post_train_eval_episodes",
        "--post-train-eval-episodes",
        type_name="int",
        default=100,
        owner="goal_objective",
        source_section="goal_train",
        help="Episodes per checkpoint for post-training checkpoint eval.",
    ),
    TrainConfigField(
        "checkpoint_eval_environment",
        "--checkpoint-eval-environment",
        type_name="json",
        default=None,
        serialize="json",
        mapping_value=True,
        owner="goal_objective",
        source_section="goal_train",
        help="Resolved goal-owned environment contract for checkpoint evaluation.",
    ),
    TrainConfigField(
        "checkpoint_eval_n_envs",
        "--checkpoint-eval-n-envs",
        type_name="int",
        default=20,
        validation_min=1,
        owner="goal_objective",
        source_section="goal_train",
        help="Vector env count for checkpoint eval.",
    ),
    TrainConfigField(
        "checkpoint_eval_stages",
        "--checkpoint-eval-stages",
        type_name="json",
        default=None,
        serialize="json",
        owner="goal_objective",
        source_section="goal_train",
        help="JSON list of cheap checkpoint eval stages for async candidate-stop screening.",
    ),
    TrainConfigField(
        "checkpoint_eval_backend",
        "--checkpoint-eval-backend",
        default="modal",
        choices=("local", "modal", "none"),
        non_empty=True,
        source_section="goal_train",
        help=(
            "Checkpoint evaluation backend. Queue-backed jobs default to Modal; "
            "local is an explicit fallback and none is for non-promotable smoke/debug runs."
        ),
    ),
    TrainConfigField(
        "checkpoint_eval_asset_manifest",
        "--checkpoint-eval-asset-manifest",
        type_name="json",
        default=None,
        serialize="json",
        mapping_value=True,
        cli_exposed=False,
        help="Materialized immutable private-ROM identity for remote checkpoint evaluation.",
    ),
    TrainConfigField(
        "checkpoint_eval_seed_protocol",
        "--checkpoint-eval-seed-protocol",
        default=SEED_PROTOCOL,
        choices=(SEED_PROTOCOL,),
        non_empty=True,
        cli_exposed=False,
        help="Versioned stochastic checkpoint-evaluation seed trace protocol.",
    ),
    TrainConfigField(
        "checkpoint_eval_seed",
        "--checkpoint-eval-seed",
        type_name="int",
        default=EVAL_SEED_START,
        validation_min=EVAL_SEED_START,
        cli_exposed=False,
        help="Materialized base seed for queue-backed checkpoint evaluation.",
    ),
    TrainConfigField(
        "post_train_eval_max_steps",
        "--post-train-eval-max-steps",
        type_name="int",
        default=0,
        owner="goal_objective",
        source_section="goal_train",
        help="Max steps per post-training eval episode; <=0 uses --max-episode-steps.",
    ),
    TrainConfigField(
        "post_train_eval_stochastic",
        "--post-train-eval-stochastic",
        kind="bool_optional",
        default=True,
        owner="goal_objective",
        source_section="goal_train",
        help="Fixed true: post-training checkpoint eval uses stochastic policy sampling.",
        cli_exposed=False,
    ),
    TrainConfigField(
        "early_stop",
        "--early-stop",
        type_name="json",
        default=None,
        serialize="json",
        owner="goal_objective",
        source_section="goal_train",
        help="JSON early-stop list of AND-combined metric threshold rules.",
    ),
    TrainConfigField(
        "selection_rank",
        "--selection-rank",
        type_name="json",
        default=(),
        serialize="json",
        sequence_items="str",
        owner="goal_objective",
        help="Ordered objective.rank contract carried into checkpoint selection.",
    ),
    TrainConfigField(
        "wandb",
        "--wandb",
        kind="store_true",
        default=False,
        queue_required=True,
        help="Log training to Weights & Biases",
    ),
    TrainConfigField("wandb_project", "--wandb-project", default=None),
    TrainConfigField("wandb_entity", "--wandb-entity", default=None),
    TrainConfigField("wandb_group", "--wandb-group", default=None),
    TrainConfigField("wandb_tags", "--wandb-tags", default="", help="Comma-separated W&B tags"),
    TrainConfigField(
        "wandb_mode",
        "--wandb-mode",
        default="online",
        choices=WANDB_MODE_CHOICES,
        queue_required=True,
        non_empty=True,
    ),
    TrainConfigField(
        "runtime_image_ref",
        "--runtime-image-ref",
        default="",
        help="Immutable runtime image ref recorded as run metadata; does not affect training.",
    ),
    TrainConfigField(
        "runtime_input_sha256",
        "--runtime-input-sha256",
        default="",
        cli_exposed=False,
        help="Content-addressed runtime-input identity recorded in run metadata.",
    ),
    TrainConfigField(
        "runtime_build_source_sha",
        "--runtime-build-source-sha",
        default="",
        cli_exposed=False,
        help="Source revision that originally built the reused runtime image.",
    ),
    TrainConfigField(
        "source_sha",
        "--source-sha",
        default="",
        cli_exposed=False,
        help="Exact pushed source revision defining this run and recipe composition.",
    ),
    TrainConfigField(
        "machine",
        "--machine",
        default="",
        help="Exact queue machine recorded as run metadata; does not affect training.",
    ),
    TrainConfigField(
        "batch_id",
        "--batch-id",
        default="",
        cli_exposed=False,
        help="Immutable queue submission cohort recorded in W&B config.",
    ),
    TrainConfigField(
        "campaign_id",
        "--campaign-id",
        default="",
        cli_exposed=False,
        help="Optional checked-in research campaign recorded in W&B config.",
    ),
    TrainConfigField(
        "game_family",
        "--game-family",
        default="",
        cli_exposed=False,
        help="Provider-neutral game family recorded in W&B config.",
    ),
    TrainConfigField(
        "retry_of_job_id",
        "--retry-of-job-id",
        type_name="int",
        default=0,
        cli_exposed=False,
        help="Source queue job id for an explicit retry; 0 means not a retry.",
    ),
    TrainConfigField(
        "goal_slug", "--goal-slug", default="", help="Research goal slug recorded in W&B config."
    ),
    TrainConfigField(
        "goal_path",
        "--goal-path",
        default="",
        help="Research goal path recorded in W&B config.",
    ),
    TrainConfigField(
        "goal_sha256",
        "--goal-sha256",
        default="",
        cli_exposed=False,
        help="Exact checked-in goal file hash recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_slug",
        "--recipe-slug",
        default="",
        help="Experiment recipe slug recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_path",
        "--recipe-path",
        default="",
        help="Experiment recipe path recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_sha256",
        "--recipe-sha256",
        default="",
        cli_exposed=False,
        help="Exact checked-in recipe file hash recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_composition",
        "--recipe-composition",
        type_name="json",
        default={},
        serialize="json",
        mapping_value=True,
        cli_exposed=False,
        help="Exact goal and recipe source-file composition recorded in W&B config.",
    ),
    TrainConfigField(
        "recipe_overrides",
        "--recipe-overrides",
        type_name="json",
        default=(),
        serialize="json",
        help="Hydra/OmegaConf dotlist overrides applied to the checked-in recipe.",
    ),
    TrainConfigField(
        "queue_train_job_id",
        "--queue-train-job-id",
        type_name="int",
        default=0,
        help="Queue train job id recorded in W&B config; 0 means local/unqueued.",
    ),
    TrainConfigField(
        "wandb_run_id",
        "--wandb-run-id",
        default="",
        cli_exposed=False,
        help="Stable W&B run id materialized for queue-backed publisher ownership.",
    ),
    TrainConfigField(
        "no_wandb_artifacts",
        "--no-wandb-artifacts",
        kind="store_true",
        default=False,
        help="Disable W&B model uploads",
    ),
    TrainConfigField(
        "wandb_artifact_storage_uri",
        "--wandb-artifact-storage-uri",
        default="",
        queue_required=True,
        help="Optional s3://bucket/prefix base URI for model artifacts. Model zips are stored under <game-id>/... below that URI, and W&B logs reference artifacts instead of storing file bytes.",
    ),
)
