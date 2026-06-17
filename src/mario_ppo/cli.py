from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from mario_ppo.env import DEFAULT_HUD_CROP_TOP
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "timesteps": 512,
        "n_envs": 1,
        "batch_size": 128,
        "max_episode_steps": 600,
        "checkpoint_freq": 256,
        "run_name": "smoke",
        "run_description": "Tiny local smoke run that checks the PPO training path compiles and saves.",
    },
    "baseline": {},
    "modal-t4": {
        "n_envs": 32,
        "env_threads": 0,
        "torch_num_threads": 0,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 10,
        "wandb": True,
        "run_description": "Modal T4 baseline training shape using benchmarked default concurrency.",
    },
}

TRAIN_VALUE_OPTIONS = {
    "preset": "--preset",
    "timesteps": "--timesteps",
    "n_envs": "--n-envs",
    "env_threads": "--env-threads",
    "torch_num_threads": "--torch-num-threads",
    "seed": "--seed",
    "run_name": "--run-name",
    "run_description": "--run-description",
    "runs_dir": "--runs-dir",
    "game": "--game",
    "state": "--state",
    "states": "--states",
    "frame_skip": "--frame-skip",
    "max_episode_steps": "--max-episode-steps",
    "hud_crop_top": "--hud-crop-top",
    "eval_freq": "--eval-freq",
    "eval_episodes": "--eval-episodes",
    "completion_x_threshold": "--completion-x-threshold",
    "eval_video_fps": "--eval-video-fps",
    "eval_video_scale": "--eval-video-scale",
    "checkpoint_freq": "--checkpoint-freq",
    "stop_completion_episode_window": "--stop-completion-episode-window",
    "stop_completion_rate_threshold": "--stop-completion-rate-threshold",
    "stop_completion_rolling_window": "--stop-completion-rolling-window",
    "stop_completion_rolling_threshold": "--stop-completion-rolling-threshold",
    "learning_rate": "--learning-rate",
    "learning_rate_final": "--learning-rate-final",
    "learning_rate_schedule_timesteps": "--learning-rate-schedule-timesteps",
    "n_steps": "--n-steps",
    "batch_size": "--batch-size",
    "n_epochs": "--n-epochs",
    "device": "--device",
    "gamma": "--gamma",
    "gae_lambda": "--gae-lambda",
    "ent_coef": "--ent-coef",
    "ent_coef_final": "--ent-coef-final",
    "ent_coef_schedule_timesteps": "--ent-coef-schedule-timesteps",
    "vf_coef": "--vf-coef",
    "clip_range": "--clip-range",
    "adam_eps": "--adam-eps",
    "target_kl": "--target-kl",
    "reward_mode": "--reward-mode",
    "progress_reward_cap": "--progress-reward-cap",
    "progress_reward_scale": "--progress-reward-scale",
    "terminal_reward": "--terminal-reward",
    "reward_scale": "--reward-scale",
    "time_penalty": "--time-penalty",
    "death_penalty": "--death-penalty",
    "completion_reward": "--completion-reward",
    "no_progress_timeout_steps": "--no-progress-timeout-steps",
    "no_progress_min_delta": "--no-progress-min-delta",
    "action_set": "--action-set",
    "resume": "--resume",
    "wandb_project": "--wandb-project",
    "wandb_entity": "--wandb-entity",
    "wandb_group": "--wandb-group",
    "wandb_tags": "--wandb-tags",
    "wandb_mode": "--wandb-mode",
    "wandb_artifact_storage_uri": "--wandb-artifact-storage-uri",
}
TRAIN_TRUE_FLAGS = {
    "eval_stochastic": "--eval-stochastic",
    "no_eval_videos": "--no-eval-videos",
    "use_retro_reward": "--use-retro-reward",
    "clip_rewards": "--clip-rewards",
    "score_progress_clipped": "--score-progress-clipped",
    "no_terminate_on_life_loss": "--no-terminate-on-life-loss",
    "terminate_on_level_change": "--terminate-on-level-change",
    "terminate_on_completion": "--terminate-on-completion",
    "wandb": "--wandb",
    "no_wandb_artifacts": "--no-wandb-artifacts",
}
TRAIN_BOOLEAN_OPTIONS = {
    "max_pool_frames": ("--max-pool-frames", "--no-max-pool-frames"),
    "normalize_advantage": ("--normalize-advantage", "--no-normalize-advantage"),
}
TRAIN_COMMAND_FIELDS = (
    tuple(TRAIN_VALUE_OPTIONS) + tuple(TRAIN_TRUE_FLAGS) + tuple(TRAIN_BOOLEAN_OPTIONS)
)


def build_train_command(options: Mapping[str, Any]) -> list[str]:
    cmd = ["python", "-m", "mario_ppo.train"]
    for key, flag in TRAIN_VALUE_OPTIONS.items():
        value = options.get(key)
        if value is None or value == "":
            continue
        if key == "target_kl" and float(value) <= 0:
            continue
        cmd.extend([flag, str(value)])
    for key, flag in TRAIN_TRUE_FLAGS.items():
        if options.get(key):
            cmd.append(flag)
    for key, (true_flag, false_flag) in TRAIN_BOOLEAN_OPTIONS.items():
        if key not in options:
            continue
        value = options[key]
        if value is True:
            cmd.append(true_flag)
        elif value is False:
            cmd.append(false_flag)
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on SuperMarioBros-Nes-v0")
    parser.add_argument(
        "--preset",
        choices=sorted(TRAINING_PRESETS),
        help="Named training shape; explicit CLI flags override preset values.",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--env-threads",
        type=int,
        default=0,
        help="Native stable-retro env threads; <=0 keeps min(n_envs, 16).",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="PyTorch CPU intra-op threads; <=0 leaves the torch default.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-name", default="ppo_level1_1")
    parser.add_argument(
        "--run-description",
        default="",
        help="Human-readable description of the experiment or ablation being run.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument(
        "--states",
        default="",
        help="Comma-separated training states. If set, vector workers cycle through these states by rank.",
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--max-pool-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Max-pool over the last two raw frames inside each frame-skip step.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=4500)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize; 32 removes the Mario HUD.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Training-loop eval frequency. Keep 0 to evaluate checkpoints out of process.",
    )
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=3160,
        help="Treat an episode as level-complete if max_x_pos reaches this value; set <=0 to disable.",
    )
    parser.add_argument(
        "--no-eval-videos", action="store_true", help="Disable best-episode eval videos"
    )
    parser.add_argument("--eval-video-fps", type=float, default=30.0)
    parser.add_argument("--eval-video-scale", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--stop-completion-episode-window",
        type=int,
        default=0,
        help=(
            "Stop when completion rate over the last N completed training episodes "
            "reaches --stop-completion-rate-threshold; <=0 disables this early stop."
        ),
    )
    parser.add_argument(
        "--stop-completion-rate-threshold",
        type=float,
        default=0.0,
        help="Completion-rate threshold over completed training episodes for early stopping.",
    )
    parser.add_argument(
        "--stop-completion-rolling-window",
        type=int,
        default=0,
        help=(
            "Stop when rolling mean completion events per PPO rollout reaches the "
            "configured threshold; <=0 disables this early stop."
        ),
    )
    parser.add_argument(
        "--stop-completion-rolling-threshold",
        type=float,
        default=0.0,
        help="Rolling mean completion-events-per-rollout threshold for early stopping.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate-final",
        type=float,
        default=None,
        help="If set, linearly decay learning rate from --learning-rate to this value over training.",
    )
    parser.add_argument(
        "--learning-rate-schedule-timesteps",
        type=int,
        default=0,
        help=("Timesteps over which to decay learning rate; <=0 decays over --timesteps."),
    )
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="If set, linearly decay entropy coefficient from --ent-coef to this value.",
    )
    parser.add_argument(
        "--ent-coef-schedule-timesteps",
        type=int,
        default=0,
        help=("Timesteps over which to decay entropy coefficient; <=0 decays over --timesteps."),
    )
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize PPO advantages before policy updates.",
    )
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--use-retro-reward", action="store_true")
    parser.add_argument("--clip-rewards", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "bounded", "additive", "score", "native"],
        default="baseline",
        help="Target reward mode. Use native for unknown games without a custom target tracker.",
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument(
        "--score-progress-clipped",
        action="store_true",
        help="In score reward mode, use clipped progress_reward instead of raw progress_delta.",
    )
    parser.add_argument(
        "--no-progress-timeout-steps",
        type=int,
        default=0,
        help="Truncate an episode after this many env steps without new x progress; <=0 disables.",
    )
    parser.add_argument(
        "--no-progress-min-delta",
        type=int,
        default=0,
        help="Minimum progress_delta that resets the no-progress timeout.",
    )
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument(
        "--terminate-on-level-change",
        action="store_true",
        help="End the episode when stable-retro reports a new level via levelHi/levelLo.",
    )
    parser.add_argument(
        "--terminate-on-completion",
        action="store_true",
        help="End the episode on either real level change or the configured completion x-threshold.",
    )
    parser.add_argument("--action-set", choices=["simple", "right", "native"], default="simple")
    parser.add_argument("--resume", help="Path to an existing PPO .zip checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--no-wandb-artifacts", action="store_true", help="Disable W&B model uploads"
    )
    parser.add_argument(
        "--wandb-artifact-storage-uri",
        default="",
        help=(
            "Optional s3://bucket/prefix base URI for model artifacts. When set, model zips "
            "are uploaded there and W&B logs reference artifacts instead of storing file bytes."
        ),
    )
    return parser


def parser_defaults() -> dict[str, Any]:
    return vars(build_parser().parse_args([]))


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.preset:
        return args
    defaults = parser_defaults()
    for key, value in TRAINING_PRESETS[args.preset].items():
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
    return args
