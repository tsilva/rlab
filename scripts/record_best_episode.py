from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import apply_model_config_defaults, explicit_arg_dests
from stable_retro_ppo.cli_args import add_env_config_args
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import assert_rom_imported, resolve_env_config
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH


def artifact_ref(args: argparse.Namespace) -> str:
    if args.artifact:
        return args.artifact
    if not args.run_name:
        raise SystemExit("Provide --model, --artifact, or run_name")
    return model_artifact_ref(
        project=args.project,
        run_name=args.run_name,
        kind=args.kind,
        version=args.version,
    )


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model:
        return Path(args.model)
    ref = artifact_ref(args)
    download_root = artifact_download_dir(Path(args.root), ref)
    print(f"Downloading {ref} to {download_root}", flush=True)
    model_path = download_model_artifact(ref, download_root)
    print(f"Downloaded model: {model_path}", flush=True)
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run episodes and save the best-reward Stable Retro video"
    )
    parser.add_argument(
        "run_name", nargs="?", help="Training run name or W&B artifact prefix"
    )
    parser.add_argument("--model", help="Local PPO .zip model path")
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH, help="W&B entity/project")
    parser.add_argument("--artifact", help="Full artifact ref, overriding run_name/kind/project")
    parser.add_argument("--kind", choices=["final", "best", "checkpoint"], default="best")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--episodes", type=int, default=20)
    add_env_config_args(parser, max_steps_default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy actions")
    parser.add_argument("--output", default="runs/videos/best_episode.mp4")
    parser.add_argument("--summary-output", default="runs/videos/best_episode_summary.json")
    return parser


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    explicit_dests.update(
        {"terminate_on_life_loss", "terminate_on_level_change", "terminate_on_completion"}
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args)
    apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
    assert_rom_imported(args.game)
    model = PPO.load(model_path, device=resolve_sb3_device(args.device))
    config = resolve_env_config(env_config_from_args(args, max_episode_steps_attr="max_steps"))
    output = Path(args.output)
    metrics, video_path = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        completion_x_threshold=config.completion_x_threshold,
        capture_best_video=True,
        video_path=output,
        video_fps=args.fps,
        video_scale=args.scale,
        extra={"model": str(model_path)},
    )
    for episode in metrics["episode_results"]:
        print(json.dumps(episode), flush=True)
    summary = {
        "model": str(model_path),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "seed_end": args.seed + args.episodes - 1,
        "deterministic": args.deterministic,
        "action_set": args.action_set,
        "rank_order": ["level_complete", "max_x_pos", "reward"],
        "best_episode": metrics["best_episode"],
        "episode_results": metrics["episode_results"],
        "video": str(video_path or output),
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
