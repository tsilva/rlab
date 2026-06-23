from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from stable_retro_ppo.artifacts import (
    PLAYBACK_ENV_ARG_KEYS,
    apply_config_defaults,
    env_config_from_metadata,
    explicit_arg_dests,
    load_model_metadata,
    write_model_metadata,
)
from stable_retro_ppo.cli_args import add_env_config_args
from stable_retro_ppo.env import resolve_env_config
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a W&B model artifact and play it locally"
    )
    parser.add_argument(
        "run_name", nargs="?", help="Training run name, e.g. modal_gpu_short_improve"
    )
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH, help="W&B entity/project")
    parser.add_argument("--artifact", help="Full artifact ref, overriding run_name/kind/project")
    parser.add_argument("--kind", choices=["final", "best", "checkpoint"], default="final")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--episodes", type=int, default=3)
    add_env_config_args(parser, max_steps_default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser


def artifact_ref(args: argparse.Namespace) -> str:
    if args.artifact:
        return args.artifact
    if not args.run_name:
        raise SystemExit("run_name is required unless --artifact is provided")
    return model_artifact_ref(
        project=args.project,
        run_name=args.run_name,
        kind=args.kind,
        version=args.version,
    )


def apply_artifact_run_config_defaults(
    args: argparse.Namespace,
    ref: str,
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> dict[str, Any]:
    load_wandb_env()

    import wandb

    try:
        run = wandb.Api().artifact(ref, type="model").logged_by()
    except Exception as exc:
        print(f"warning: could not infer playback config from {ref}: {exc}", file=sys.stderr)
        return {}
    if run is None:
        return {}

    config = getattr(run, "config", {}) or {}
    apply_config_defaults(args, config, parser_defaults, explicit_dests)
    return config if isinstance(config, dict) else {}


def append_explicit_env_args(
    cmd: list[str],
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    explicit_dests: set[str],
) -> None:
    actions = {action.dest: action for action in parser._actions}
    for dest in PLAYBACK_ENV_ARG_KEYS:
        if dest not in explicit_dests:
            continue
        action = actions.get(dest)
        if action is None:
            continue
        value = getattr(args, dest)
        if isinstance(action, argparse.BooleanOptionalAction):
            prefix = "--no-" if value is False else "--"
            option = next(
                opt for opt in action.option_strings if opt.startswith(prefix)
            )
            cmd.append(option)
        elif action.nargs == 0:
            if value:
                cmd.append(action.option_strings[0])
        else:
            cmd.extend([action.option_strings[0], str(value)])


def play_model(
    model_path: Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    explicit_dests: set[str],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "stable_retro_ppo.play",
        "--model",
        str(model_path),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--scale",
        str(args.scale),
    ]
    if args.stochastic:
        cmd.append("--stochastic")
    if args.random_seeds:
        cmd.append("--random-seeds")
    append_explicit_env_args(cmd, parser, args, explicit_dests)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    explicit_dests.update(
        {"terminate_on_life_loss", "terminate_on_level_change", "terminate_on_completion"}
    )
    args = parser.parse_args()
    ref = artifact_ref(args)
    download_root = artifact_download_dir(Path(args.root), ref)
    print(f"Downloading {ref} to {download_root}")
    model_path = download_model_artifact(ref, download_root)
    saved_config = env_config_from_metadata(load_model_metadata(model_path))
    inferred_config: dict[str, Any] = {}
    if saved_config:
        apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
    else:
        inferred_config = apply_artifact_run_config_defaults(
            args,
            ref,
            parser_defaults,
            explicit_dests,
        )
    metadata_path = None
    if not saved_config and inferred_config:
        metadata_args = parser.parse_args([])
        apply_config_defaults(metadata_args, inferred_config, parser_defaults, set())
        metadata_config = resolve_env_config(
            env_config_from_args(metadata_args, max_episode_steps_attr="max_steps")
        )
        metadata_path = write_model_metadata(model_path, args, metadata_config, kind=args.kind)
    if metadata_path is not None:
        print(f"Wrote playback metadata: {metadata_path}")
    print(f"Downloaded model: {model_path}")
    if not args.download_only:
        play_model(model_path, args, parser, explicit_dests)


if __name__ == "__main__":
    main()
