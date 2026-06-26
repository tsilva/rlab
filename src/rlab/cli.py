from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

from rlab.env import EnvConfig
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT


ADVANTAGE_NORMALIZATION_CHOICES = ("auto", "none", "global", "per-task")


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "timesteps": 512,
        "n_envs": 1,
        "batch_size": 128,
        "max_episode_steps": 600,
        "checkpoint_freq": 256,
        "run_name": "smoke",
        "run_description": "Tiny local smoke run that checks the rlab training path compiles and saves.",
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
    "state_probs": "--state-probs",
    "done_on_info_json": "--done-on-info-json",
    "task_conditioning_info_vars": "--task-conditioning-info-vars",
    "task_conditioning_info_values": "--task-conditioning-info-values",
    "frame_skip": "--frame-skip",
    "max_episode_steps": "--max-episode-steps",
    "hud_crop_top": "--hud-crop-top",
    "eval_freq": "--eval-freq",
    "eval_episodes": "--eval-episodes",
    "completion_x_threshold": "--completion-x-threshold",
    "eval_video_fps": "--eval-video-fps",
    "eval_video_scale": "--eval-video-scale",
    "checkpoint_freq": "--checkpoint-freq",
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
    "clip_range_vf": "--clip-range-vf",
    "advantage_normalization": "--advantage-normalization",
    "policy_net_arch": "--policy-net-arch",
    "value_net_arch": "--value-net-arch",
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
    "runtime_image_ref": "--runtime-image-ref",
    "run_target": "--run-target",
}
TRAIN_TRUE_FLAGS = {
    "eval_stochastic": "--eval-stochastic",
    "no_eval_videos": "--no-eval-videos",
    "task_conditioning": "--task-conditioning",
    "use_retro_reward": "--use-retro-reward",
    "clip_rewards": "--clip-rewards",
    "score_progress_clipped": "--score-progress-clipped",
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
    cmd = ["python", "-m", "rlab.train"]
    for key, flag in TRAIN_VALUE_OPTIONS.items():
        value = options.get(key)
        if value is None or value == "":
            continue
        if key == "target_kl" and float(value) <= 0:
            continue
        if key == "done_on_info_json" and isinstance(value, Mapping):
            value = json.dumps(value, separators=(",", ":"))
        elif key == "task_conditioning_info_values" and isinstance(value, list | tuple):
            value = ";".join(
                ",".join(str(item) for item in row)
                if isinstance(row, list | tuple)
                else str(row)
                for row in value
            )
        elif isinstance(value, list | tuple):
            value = ",".join(str(item) for item in value)
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
    parser_defaults_env = EnvConfig()
    parser = argparse.ArgumentParser(description="Train PPO on an imported Stable Retro game")
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
    parser.add_argument("--run-name", default="ppo_retro")
    parser.add_argument(
        "--run-description",
        default="",
        help="Human-readable description of the experiment or ablation being run.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--game",
        default=parser_defaults_env.game,
        help="Stable Retro game id. Defaults to RETRO_GAME when set.",
    )
    parser.add_argument(
        "--state",
        default=parser_defaults_env.state,
        help="Stable Retro state. If omitted, registered targets may provide a default.",
    )
    parser.add_argument(
        "--states",
        default="",
        help=(
            "Comma-separated native-vector training states. Without --state-probs, "
            "provide exactly one state per env slot in order."
        ),
    )
    parser.add_argument(
        "--state-probs",
        default="",
        help=(
            "Comma-separated positive weights for --states. Values are normalized and "
            "sampled by the native vector env independently on each episode reset."
        ),
    )
    parser.add_argument(
        "--done-on-info-json",
        default="",
        help=(
            "JSON object mapping done-reason names to [key_or_keys, op], for example "
            '\'{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}\''
        ),
    )
    parser.add_argument(
        "--task-conditioning",
        action="store_true",
        help=(
            "Use SB3 MultiInputPolicy with a one-hot task vector derived from the "
            "native active state for each env lane."
        ),
    )
    parser.add_argument(
        "--task-conditioning-info-vars",
        default="",
        help="Comma-separated info keys used to map task-conditioned one-hot vectors.",
    )
    parser.add_argument(
        "--task-conditioning-info-values",
        default="",
        help=(
            "Semicolon-separated info-value rows for task conditioning, "
            "for example '0,0;0,1'. Omit when values can be derived from states."
        ),
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
        default=parser_defaults_env.hud_crop_top,
        help="Crop this many pixels from the top of raw frames before grayscale resize; -1 uses the target default.",
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
        default=parser_defaults_env.completion_x_threshold,
        help="Deprecated no-op; level completion is detected from stable-retro level changes.",
    )
    parser.add_argument(
        "--no-eval-videos", action="store_true", help="Disable best-episode eval videos"
    )
    parser.add_argument("--eval-video-fps", type=float, default=30.0)
    parser.add_argument("--eval-video-scale", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
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
        "--clip-range-vf",
        type=float,
        default=None,
        help="Optional PPO value-function clipping range; omitted keeps SB3 default.",
    )
    parser.add_argument(
        "--policy-net-arch",
        default="",
        help="Comma-separated policy MLP hidden sizes after the CNN/combined extractor.",
    )
    parser.add_argument(
        "--value-net-arch",
        default="",
        help="Comma-separated value MLP hidden sizes after the CNN/combined extractor.",
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize PPO advantages before policy updates.",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=ADVANTAGE_NORMALIZATION_CHOICES,
        default="auto",
        help=(
            "PPO advantage normalization mode. auto preserves --normalize-advantage; "
            "per-task normalizes each task-conditioned rollout slice once before PPO epochs."
        ),
    )
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--use-retro-reward", action="store_true")
    parser.add_argument("--clip-rewards", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["auto", "baseline", "bounded", "additive", "score", "native"],
        default=parser_defaults_env.reward_mode,
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
    parser.add_argument(
        "--action-set",
        default=parser_defaults_env.action_set,
        help="Target-specific action set name, native, or auto for the target default.",
    )
    parser.add_argument("--resume", help="Path to an existing PPO .zip checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--runtime-image-ref",
        default="",
        help="Immutable runtime image ref recorded as run metadata; does not affect training.",
    )
    parser.add_argument(
        "--run-target",
        default="",
        help="Canonical compute target recorded as run metadata; does not affect training.",
    )
    parser.add_argument(
        "--no-wandb-artifacts", action="store_true", help="Disable W&B model uploads"
    )
    parser.add_argument(
        "--wandb-artifact-storage-uri",
        default="",
        help=(
            "Optional s3://bucket/prefix base URI for model artifacts. Model zips are stored "
            "under <game-id>/... below that URI, and W&B logs reference artifacts instead "
            "of storing file bytes."
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
