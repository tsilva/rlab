from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import (
    DEFAULT_HUD_CROP_TOP,
    EnvConfig,
    assert_rom_imported,
    make_mario_env,
    make_vec_envs,
)
from mario_ppo.eval_metrics import is_level_complete
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"


def artifact_ref(args: argparse.Namespace) -> str:
    if args.artifact:
        return args.artifact
    if not args.run_name:
        raise SystemExit("Provide --model, --artifact, or run_name")
    return f"{args.project}/{args.run_name}-{args.kind}:{args.version}"


def download_artifact(ref: str, root: Path) -> Path:
    load_wandb_env()

    import wandb

    root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    artifact = api.artifact(ref, type="model")
    path = Path(artifact.download(root=str(root)))
    zip_files = sorted(path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in downloaded artifact: {path}")
    if len(zip_files) > 1:
        print(f"Multiple model files found; using {zip_files[0]}", file=sys.stderr)
    return zip_files[0]


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model:
        return Path(args.model)
    ref = artifact_ref(args)
    download_root = Path(args.root) / slug(ref.replace("/", "_").replace(":", "_"))
    print(f"Downloading {ref} to {download_root}", flush=True)
    model_path = download_artifact(ref, download_root)
    print(f"Downloaded model: {model_path}", flush=True)
    return model_path


def write_video(frames: list[np.ndarray], output: Path, fps: float, scale: int) -> None:
    if not frames:
        raise ValueError("No frames to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    out_size = (width * scale, height * scale)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame in frames:
            if scale != 1:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def run_episode(
    env,
    model: PPO,
    max_steps: int,
    deterministic: bool,
    seed: int,
    completion_x_threshold: int,
    capture_actions: bool,
):
    torch.manual_seed(seed)
    env.seed(seed)
    obs = env.reset()
    actions: list[int] = []
    total_reward = 0.0
    max_x = 0
    max_level_x = 0
    final_info = {}

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_int = int(action[0])
        if capture_actions:
            actions.append(action_int)
        obs, rewards, dones, infos = env.step(action)
        info = dict(infos[0])
        total_reward += float(rewards[0])
        max_x = max(max_x, int(info.get("max_x_pos", 0)))
        max_level_x = max(max_level_x, int(info.get("level_max_x_pos", 0)))
        final_info = info
        if bool(dones[0]):
            break
    completed = is_level_complete(final_info, max_x, completion_x_threshold=completion_x_threshold)
    died = bool(final_info.get("died", False))
    death_x_pos = final_info.get("death_x_pos")
    if died and death_x_pos is None:
        death_x_pos = max_x

    return {
        "reward": total_reward,
        "max_x_pos": max_x,
        "max_level_x_pos": max_level_x,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "steps": step_idx + 1,
        "level_complete": completed,
        "died": died,
        "death_x_pos": int(death_x_pos) if death_x_pos is not None else None,
        "actions": actions,
    }


def replay_actions(env, actions: list[int], seed: int) -> list[np.ndarray]:
    env.reset(seed=seed)
    frames = [env.render()]
    for action in actions:
        _obs, _reward, terminated, truncated, _info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames


def performance_key(summary: dict) -> tuple[int, int, float]:
    return (
        int(bool(summary["level_complete"])),
        int(summary["max_x_pos"]),
        float(summary["reward"]),
    )


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
    parser.add_argument("--action-set", choices=["simple", "right"], default="right")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "bounded", "additive", "score"],
        default="baseline",
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
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
    config = EnvConfig(
        state=args.state,
        frame_skip=args.frame_skip,
        max_pool_frames=args.max_pool_frames,
        max_episode_steps=args.max_steps,
        hud_crop_top=args.hud_crop_top,
        action_set=args.action_set,
        reward_mode=args.reward_mode,
        progress_reward_cap=args.progress_reward_cap,
        progress_reward_scale=args.progress_reward_scale,
        terminal_reward=args.terminal_reward,
        reward_scale=args.reward_scale,
        time_penalty=args.time_penalty,
        death_penalty=args.death_penalty,
        completion_reward=args.completion_reward,
        completion_x_threshold=args.completion_x_threshold,
        terminate_on_level_change=args.terminate_on_level_change,
        terminate_on_completion=args.terminate_on_completion,
    )
    env = make_vec_envs(config=config, n_envs=1, seed=args.seed)

    best_episode = None
    episode_summaries = []
    try:
        for episode_idx in range(args.episodes):
            episode_seed = args.seed + episode_idx
            result = run_episode(
                env=env,
                model=model,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                seed=episode_seed,
                completion_x_threshold=args.completion_x_threshold,
                capture_actions=True,
            )
            actions = result.pop("actions")
            summary = {"episode": episode_idx + 1, "seed": episode_seed, **result}
            episode_summaries.append(summary)
            print(json.dumps(summary), flush=True)
            if best_episode is None or performance_key(summary) > performance_key(
                best_episode["summary"]
            ):
                best_episode = {"summary": summary, "actions": actions}
    finally:
        env.close()

    if best_episode is None:
        raise RuntimeError("No episode completed")

    output = Path(args.output)
    video_env = make_mario_env(config=config, seed=best_episode["summary"]["seed"])
    try:
        frames = replay_actions(
            video_env, best_episode["actions"], seed=best_episode["summary"]["seed"]
        )
    finally:
        video_env.close()
    write_video(frames, output=output, fps=args.fps, scale=args.scale)
    summary = {
        "model": str(model_path),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "seed_end": args.seed + args.episodes - 1,
        "deterministic": args.deterministic,
        "action_set": args.action_set,
        "rank_order": ["level_complete", "max_x_pos", "reward"],
        "best_episode": best_episode["summary"],
        "episode_results": episode_summaries,
        "video": str(output),
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
