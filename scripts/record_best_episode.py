from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
from stable_baselines3 import PPO

from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import DEFAULT_HUD_CROP_TOP, assert_rom_imported
from mario_ppo.env_config import env_config_from_args
from mario_ppo.eval_runner import evaluate_model_episodes
from mario_ppo.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
)
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH


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
        description="Run episodes and save the best-reward Mario video"
    )
    parser.add_argument(
        "run_name", nargs="?", help="Training run name, e.g. modal_right_action_250k_lr1e4_eval50"
    )
    parser.add_argument("--model", help="Local PPO .zip model path")
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH, help="W&B entity/project")
    parser.add_argument("--artifact", help="Full artifact ref, overriding run_name/kind/project")
    parser.add_argument("--kind", choices=["final", "best", "checkpoint"], default="best")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy actions")
    parser.add_argument("--action-set", choices=["simple", "right", "native"], default="right")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "bounded", "additive", "score", "native"],
        default="baseline",
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
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=3160,
        help="Stored in the summary; ranking uses level_complete first, then max_x_pos, then reward.",
    )
    parser.add_argument("--output", default="runs/videos/best_episode.mp4")
    parser.add_argument("--summary-output", default="runs/videos/best_episode_summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert_rom_imported()
    model_path = resolve_model_path(args)
    model = PPO.load(model_path, device=resolve_sb3_device(args.device))
    config = env_config_from_args(args, max_episode_steps_attr="max_steps")
    output = Path(args.output)
    metrics, video_path = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        completion_x_threshold=args.completion_x_threshold,
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
