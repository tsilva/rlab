from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from stable_retro_ppo.env import EnvConfig
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


RUN_CONFIG_ARG_KEYS = {
    "game": ("game",),
    "state": ("state",),
    "frame_skip": ("frame_skip",),
    "max_pool_frames": ("max_pool_frames",),
    "max_steps": ("max_steps", "max_episode_steps"),
    "reward_mode": ("reward_mode",),
    "progress_reward_cap": ("progress_reward_cap",),
    "progress_reward_scale": ("progress_reward_scale",),
    "terminal_reward": ("terminal_reward",),
    "reward_scale": ("reward_scale",),
    "time_penalty": ("time_penalty",),
    "death_penalty": ("death_penalty",),
    "completion_reward": ("completion_reward",),
    "score_progress_clipped": ("score_progress_clipped",),
    "no_progress_timeout_steps": ("no_progress_timeout_steps",),
    "no_progress_min_delta": ("no_progress_min_delta",),
    "completion_x_threshold": ("completion_x_threshold",),
    "terminate_on_life_loss": ("terminate_on_life_loss",),
    "terminate_on_level_change": ("terminate_on_level_change",),
    "terminate_on_completion": ("terminate_on_completion",),
    "action_set": ("action_set",),
}


def build_parser() -> argparse.ArgumentParser:
    defaults = EnvConfig()
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
    parser.add_argument("--game", default=defaults.game)
    parser.add_argument("--state", default=defaults.state)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["auto", "baseline", "bounded", "additive", "score", "native"],
        default=defaults.reward_mode,
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--score-progress-clipped", action="store_true")
    parser.add_argument("--no-progress-timeout-steps", type=int, default=0)
    parser.add_argument("--no-progress-min-delta", type=int, default=0)
    parser.add_argument("--completion-x-threshold", type=int, default=defaults.completion_x_threshold)
    parser.add_argument(
        "--terminate-on-life-loss",
        action=argparse.BooleanOptionalAction,
        default=defaults.terminate_on_life_loss,
    )
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", default=defaults.action_set)
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
) -> None:
    load_wandb_env()

    import wandb

    try:
        run = wandb.Api().artifact(ref, type="model").logged_by()
    except Exception as exc:
        print(f"warning: could not infer playback config from {ref}: {exc}", file=sys.stderr)
        return
    if run is None:
        return

    config = getattr(run, "config", {}) or {}
    for arg_name, config_keys in RUN_CONFIG_ARG_KEYS.items():
        current_value = getattr(args, arg_name)
        default_value = parser_defaults[arg_name]
        if current_value != default_value and current_value not in ("", None):
            continue
        for config_key in config_keys:
            if config_key in config and config[config_key] is not None:
                setattr(args, arg_name, config[config_key])
                break


def play_model(model_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "stable_retro_ppo.play",
        "--model",
        str(model_path),
        "--episodes",
        str(args.episodes),
        "--game",
        args.game,
        "--state",
        args.state,
        "--max-steps",
        str(args.max_steps),
        "--frame-skip",
        str(args.frame_skip),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--scale",
        str(args.scale),
        "--action-set",
        args.action_set,
        "--reward-mode",
        args.reward_mode,
        "--progress-reward-cap",
        str(args.progress_reward_cap),
        "--progress-reward-scale",
        str(args.progress_reward_scale),
        "--terminal-reward",
        str(args.terminal_reward),
        "--reward-scale",
        str(args.reward_scale),
        "--time-penalty",
        str(args.time_penalty),
        "--death-penalty",
        str(args.death_penalty),
        "--completion-reward",
        str(args.completion_reward),
        "--no-progress-timeout-steps",
        str(args.no_progress_timeout_steps),
        "--no-progress-min-delta",
        str(args.no_progress_min_delta),
        "--completion-x-threshold",
        str(args.completion_x_threshold),
    ]
    if args.score_progress_clipped:
        cmd.append("--score-progress-clipped")
    if args.stochastic:
        cmd.append("--stochastic")
    if not args.max_pool_frames:
        cmd.append("--no-max-pool-frames")
    if args.random_seeds:
        cmd.append("--random-seeds")
    if args.terminate_on_life_loss is True:
        cmd.append("--terminate-on-life-loss")
    elif args.terminate_on_life_loss is False:
        cmd.append("--no-terminate-on-life-loss")
    if args.terminate_on_level_change:
        cmd.append("--terminate-on-level-change")
    if args.terminate_on_completion:
        cmd.append("--terminate-on-completion")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    args = parser.parse_args()
    ref = artifact_ref(args)
    apply_artifact_run_config_defaults(args, ref, parser_defaults)
    download_root = artifact_download_dir(Path(args.root), ref)
    print(f"Downloading {ref} to {download_root}")
    model_path = download_model_artifact(ref, download_root)
    print(f"Downloaded model: {model_path}")
    if not args.download_only:
        play_model(model_path, args)


if __name__ == "__main__":
    main()
