from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlab.checkpoint_eval_config import normalize_checkpoint_eval_stages
from rlab.early_stop import normalize_early_stop_config
from rlab.env import EnvConfig
from rlab.env_config import parse_obs_crop
from rlab.provider_config import provider_num_envs
from rlab.seeds import validate_training_seed
from rlab.train_config import (
    add_train_config_args,
    build_train_command_from_fields,
    validate_and_normalize_train_config,
)


def build_train_command(options: Mapping[str, Any]) -> list[str]:
    return build_train_command_from_fields(options)


def explicit_train_arg_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    option_dests: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_dests[option] = action.dest
    return {
        option_dests[arg.split("=", 1)[0]] for arg in argv if arg.split("=", 1)[0] in option_dests
    }


def load_train_config_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"--train-config-json must contain a JSON object: {path}")
    return payload


def apply_train_config_json(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    explicit_dests: set[str],
) -> argparse.Namespace:
    path = getattr(args, "train_config_json", None)
    if path is None:
        return args

    payload = load_train_config_json(Path(path))
    payload = validate_and_normalize_train_config(
        payload,
        label=f"train config file {path}",
    )
    args._train_config_json_fields = set(payload)

    for key, value in payload.items():
        if key == "train_config_json" or key in explicit_dests:
            continue
        if key == "wandb_tags" and isinstance(value, list | tuple):
            value = ",".join(str(tag) for tag in value)
        setattr(args, key, value)
    return args


def parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {exc}") from exc


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    explicit_dests = explicit_train_arg_dests(parser, argv_list or [])
    args = parser.parse_args(argv_list)
    args._train_config_json_fields = set()
    args._explicit_train_arg_dests = set(explicit_dests)
    apply_train_config_json(args, parser, explicit_dests)
    validate_early_stop_args(args)
    validate_checkpoint_eval_stage_args(args)
    validate_training_seed(args.seed, label="--seed", seed_span=effective_n_envs(args))
    return args


def validate_early_stop_args(args: argparse.Namespace) -> None:
    early_stop = getattr(args, "early_stop", None)
    if early_stop is not None:
        args.early_stop = normalize_early_stop_config(early_stop, label="--early-stop")


def validate_checkpoint_eval_stage_args(args: argparse.Namespace) -> None:
    stages = getattr(args, "checkpoint_eval_stages", None)
    if stages is not None:
        args.checkpoint_eval_stages = normalize_checkpoint_eval_stages(
            stages,
            label="--checkpoint-eval-stages",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on a registered provider environment")
    parser.add_argument(
        "--train-config-json",
        type=Path,
        help="JSON file containing train option values. Explicit CLI flags override file values.",
    )
    add_train_config_args(
        parser,
        env_defaults=EnvConfig(),
        parse_json_value=parse_json_value,
        parse_obs_crop=parse_obs_crop,
    )
    return parser


def explicit_n_envs(args: argparse.Namespace) -> int | None:
    explicit_fields = (
        set(getattr(args, "_explicit_train_arg_dests", set()))
        | set(getattr(args, "_train_config_json_fields", set()))
    )
    return int(args.n_envs) if "n_envs" in explicit_fields else None


def effective_n_envs(args: argparse.Namespace) -> int:
    return provider_num_envs(args, explicit_n_envs=explicit_n_envs(args))
