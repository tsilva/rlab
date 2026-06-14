from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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
from mario_ppo.eval_metrics import (
    death_location_histogram,
    episode_rank,
    replay_actions_for_video,
    run_eval_episode,
    write_video,
)
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"


def split_project(value: str) -> tuple[str | None, str]:
    parts = value.split("/", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def artifact_aliases(artifact) -> list[str]:
    aliases = []
    for alias in getattr(artifact, "aliases", []) or []:
        aliases.append(str(getattr(alias, "alias", alias)))
    return aliases


def checkpoint_step_from_name(value: str) -> int | None:
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", value)
    return int(match.group(1)) if match else None


def checkpoint_step_from_artifact(artifact, model_path: Path | None = None) -> int | None:
    metadata = getattr(artifact, "metadata", {}) or {}
    step = metadata.get("checkpoint_step")
    if step is not None:
        return int(step)
    for alias in artifact_aliases(artifact):
        if alias.startswith("step-"):
            return int(alias.removeprefix("step-"))
    if model_path is not None:
        return checkpoint_step_from_name(model_path.name)
    return None


def artifact_ref(args: argparse.Namespace) -> str:
    if not args.run_name:
        raise SystemExit("run_name is required unless --artifact is provided")
    return f"{args.project}/{slug(args.run_name)}-checkpoint"


def find_checkpoint_artifacts(args: argparse.Namespace):
    load_wandb_env()

    import wandb

    api = wandb.Api()
    if args.artifact:
        return [api.artifact(ref, type="model") for ref in args.artifact]

    ref = artifact_ref(args)
    try:
        artifacts = list(api.artifact_versions("model", ref))
    except Exception as exc:
        raise SystemExit(f"Could not list W&B checkpoint artifacts for {ref}: {exc}") from exc

    artifacts.sort(key=lambda artifact: checkpoint_step_from_artifact(artifact) or -1)
    if args.max_checkpoints > 0:
        artifacts = artifacts[: args.max_checkpoints]
    return artifacts


def download_artifact(artifact, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = getattr(artifact, "qualified_name", None) or getattr(artifact, "name", "artifact")
    download_root = root / slug(name.replace("/", "_").replace(":", "_"))
    path = Path(artifact.download(root=str(download_root)))
    zip_files = sorted(path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in downloaded artifact: {path}")
    if len(zip_files) > 1:
        print(f"Multiple model files found; using {zip_files[0]}", file=sys.stderr)
    return zip_files[0]


def load_eval_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_eval_history(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def score(metrics: dict[str, Any]) -> tuple[float, int, float]:
    return (
        float(metrics["completion_rate"]),
        int(metrics["max_x_max"]),
        float(metrics["reward_mean"]),
    )


def best_metrics(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=score)


def evaluate_checkpoint(
    args: argparse.Namespace,
    model_path: Path,
    checkpoint_step: int,
    artifact_name: str,
) -> tuple[dict[str, Any], Path | None]:
    model = PPO.load(model_path, device=resolve_sb3_device(args.device))
    config = EnvConfig(
        state=args.state,
        frame_skip=args.frame_skip,
        max_pool_frames=args.max_pool_frames,
        max_episode_steps=args.max_steps,
        hud_crop_top=args.hud_crop_top,
        reward_mode=args.reward_mode,
        progress_reward_cap=args.progress_reward_cap,
        progress_reward_scale=args.progress_reward_scale,
        terminal_reward=args.terminal_reward,
        reward_scale=args.reward_scale,
        time_penalty=args.time_penalty,
        death_penalty=args.death_penalty,
        completion_reward=args.completion_reward,
        completion_x_threshold=args.completion_x_threshold,
        terminate_on_life_loss=not args.no_terminate_on_life_loss,
        terminate_on_level_change=args.terminate_on_level_change,
        terminate_on_completion=args.terminate_on_completion,
        action_set=args.action_set,
    )
    eval_env = make_vec_envs(config=config, n_envs=1, seed=args.seed + checkpoint_step)
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    best_episode_actions: list[int] = []
    best_episode_seed: int | None = None
    try:
        for episode_idx in range(args.episodes):
            episode_seed = args.seed + checkpoint_step + episode_idx
            torch.manual_seed(episode_seed)
            result = run_eval_episode(
                eval_env,
                model,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                seed=episode_seed,
                completion_x_threshold=args.completion_x_threshold,
                capture_actions=args.record_best_video,
            )
            actions = result.pop("actions")
            result = {"episode": episode_idx + 1, "seed": episode_seed, **result}
            episode_results.append(result)
            if best_episode_result is None or episode_rank(result) > episode_rank(
                best_episode_result
            ):
                best_episode_result = result
                best_episode_actions = actions
                best_episode_seed = episode_seed
    finally:
        eval_env.close()

    rewards = np.array([episode["reward"] for episode in episode_results], dtype=np.float64)
    x_positions = np.array([episode["max_x_pos"] for episode in episode_results], dtype=np.float64)
    level_x_positions = np.array(
        [episode["max_level_x_pos"] for episode in episode_results],
        dtype=np.float64,
    )
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode["level_complete"])
    death_count = sum(1 for episode in episode_results if episode["died"])

    metrics: dict[str, Any] = {
        "checkpoint_step": checkpoint_step,
        "checkpoint_artifact": artifact_name,
        "model": str(model_path),
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "hud_crop_top": args.hud_crop_top,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_max": float(rewards.max()),
        "max_x_mean": float(x_positions.mean()),
        "max_x_max": int(x_positions.max()),
        "max_level_x_mean": float(level_x_positions.mean()),
        "max_level_x_max": int(level_x_positions.max()),
        "completion_count": completion_count,
        "completion_rate": completion_count / args.episodes,
        "death_count": death_count,
        "death_rate": death_count / args.episodes,
        "death_x_histogram": death_location_histogram(death_x_positions),
        "best_episode": best_episode_result,
        "episode_results": episode_results,
    }

    video_path = None
    if args.record_best_video and best_episode_actions and best_episode_seed is not None:
        video_path = (
            Path(args.eval_dir)
            / args.run_name
            / "videos"
            / f"best_episode_{checkpoint_step}_steps.mp4"
        )
        video_env = make_mario_env(config=config, seed=best_episode_seed)
        try:
            frames = replay_actions_for_video(
                video_env,
                actions=best_episode_actions,
                seed=best_episode_seed,
            )
        finally:
            video_env.close()
        write_video(frames, video_path, fps=args.video_fps, scale=args.video_scale)
        metrics["best_episode_video"] = str(video_path)

    return metrics, video_path


def resolve_wandb_run_id(args: argparse.Namespace, artifacts) -> str | None:
    if args.wandb_run_id:
        return args.wandb_run_id
    if args.wandb_run_path:
        return args.wandb_run_path.rstrip("/").rsplit("/", 1)[-1]
    for artifact in artifacts:
        metadata = getattr(artifact, "metadata", {}) or {}
        if metadata.get("wandb_run_id"):
            return str(metadata["wandb_run_id"])
        if metadata.get("wandb_run_path"):
            return str(metadata["wandb_run_path"]).rstrip("/").rsplit("/", 1)[-1]
        try:
            logged_by = artifact.logged_by()
        except Exception:
            logged_by = None
        if logged_by is not None and getattr(logged_by, "id", None):
            return str(logged_by.id)
    return None


def init_wandb_run(args: argparse.Namespace, artifacts):
    if args.no_wandb_log:
        return None
    load_wandb_env()

    import wandb

    run_id = resolve_wandb_run_id(args, artifacts)
    if not run_id:
        raise SystemExit(
            "Could not infer the W&B run id. Pass --wandb-run-id or --wandb-run-path.",
        )
    entity, project = split_project(args.project)
    return wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=args.run_name,
        resume="allow",
        mode=args.wandb_mode,
    )


def log_wandb_eval(wandb_run, metrics: dict[str, Any], video_path: Path | None) -> None:
    if wandb_run is None:
        return
    import wandb

    payload: dict[str, Any] = {
        "eval/reward_mean": metrics["reward_mean"],
        "eval/reward_std": metrics["reward_std"],
        "eval/reward_max": metrics["reward_max"],
        "eval/max_x_mean": metrics["max_x_mean"],
        "eval/max_x_max": metrics["max_x_max"],
        "eval/max_level_x_mean": metrics["max_level_x_mean"],
        "eval/max_level_x_max": metrics["max_level_x_max"],
        "eval/completion_count": metrics["completion_count"],
        "eval/completion_rate": metrics["completion_rate"],
        "eval/death_count": metrics["death_count"],
        "eval/death_rate": metrics["death_rate"],
        "eval/best_episode_reward": metrics["best_episode"]["reward"],
        "eval/best_episode_max_x": metrics["best_episode"]["max_x_pos"],
        "eval/checkpoint_step": metrics["checkpoint_step"],
        "eval/checkpoint_artifact": metrics["checkpoint_artifact"],
        "eval/hud_crop_top": metrics["hud_crop_top"],
    }
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in metrics["episode_results"]
        if episode.get("death_x_pos") is not None
    ]
    if death_x_positions:
        payload["eval/death_x_pos_histogram"] = wandb.Histogram(death_x_positions)
    if video_path is not None and video_path.is_file():
        payload["eval/best_episode_video"] = wandb.Video(str(video_path), format="mp4")
    wandb_run.log(payload, step=int(metrics["checkpoint_step"]))


def promote_best_artifact(
    wandb_run,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    model_path: Path,
) -> None:
    if wandb_run is None or args.no_promote_best:
        return
    import wandb

    artifact = wandb.Artifact(
        f"{slug(args.run_name)}-best",
        type="model",
        metadata={
            "run_name": args.run_name,
            "kind": "best",
            "source": "local_checkpoint_eval",
            "checkpoint_step": metrics["checkpoint_step"],
            "checkpoint_artifact": metrics["checkpoint_artifact"],
            "completion_rate": metrics["completion_rate"],
            "max_x_max": metrics["max_x_max"],
            "reward_mean": metrics["reward_mean"],
            "hud_crop_top": metrics["hud_crop_top"],
        },
    )
    artifact.add_file(str(model_path), name="best_model.zip")
    wandb_run.log_artifact(
        artifact,
        aliases=["best", "latest", f"step-{metrics['checkpoint_step']}"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pending W&B Mario PPO checkpoints")
    parser.add_argument("run_name", nargs="?", help="Training run name / artifact prefix")
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH, help="W&B entity/project")
    parser.add_argument("--artifact", action="append", help="Explicit checkpoint artifact ref")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--eval-dir", default="runs/local_evals")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate checkpoints already logged"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
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
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--completion-x-threshold", type=int, default=3160)
    parser.add_argument("--record-best-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-scale", type=int, default=4)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-run-path", help="W&B run path, e.g. entity/project/runs/<id>")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--no-wandb-log", action="store_true")
    parser.add_argument("--no-promote-best", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if not args.run_name and args.artifact:
        args.run_name = slug(args.artifact[0].split("/")[-1].split("-checkpoint", 1)[0])
    if not args.run_name:
        raise SystemExit("run_name is required")

    assert_rom_imported()
    artifacts = find_checkpoint_artifacts(args)
    if not artifacts:
        print("No checkpoint artifacts found")
        return

    history_path = Path(args.eval_dir) / args.run_name / "checkpoint_eval_metrics.jsonl"
    history = load_eval_history(history_path)
    evaluated_steps = {int(row["checkpoint_step"]) for row in history}
    wandb_run = init_wandb_run(args, artifacts)

    try:
        for artifact in artifacts:
            artifact_name = getattr(artifact, "qualified_name", None) or getattr(
                artifact, "name", "artifact"
            )
            checkpoint_step = checkpoint_step_from_artifact(artifact)
            if (
                checkpoint_step is not None
                and checkpoint_step in evaluated_steps
                and not args.force
            ):
                print(f"Skipping step {checkpoint_step}: already evaluated")
                continue

            model_path = download_artifact(artifact, Path(args.root))
            checkpoint_step = checkpoint_step or checkpoint_step_from_artifact(artifact, model_path)
            if checkpoint_step is None:
                print(f"Skipping {artifact_name}: cannot infer checkpoint step", file=sys.stderr)
                continue
            if checkpoint_step in evaluated_steps and not args.force:
                print(f"Skipping step {checkpoint_step}: already evaluated")
                continue

            print(f"Evaluating checkpoint step {checkpoint_step}: {artifact_name}", flush=True)
            previous_best = best_metrics(history)
            metrics, video_path = evaluate_checkpoint(
                args, model_path, checkpoint_step, artifact_name
            )
            append_eval_history(history_path, metrics)
            history.append(metrics)
            evaluated_steps.add(checkpoint_step)
            log_wandb_eval(wandb_run, metrics, video_path)

            current_best = best_metrics(history)
            if current_best is metrics and (
                previous_best is None or score(metrics) > score(previous_best)
            ):
                promote_best_artifact(wandb_run, args, metrics, model_path)
                print(
                    "promoted best "
                    f"step={checkpoint_step} "
                    f"completion_rate={metrics['completion_rate']:.3f} "
                    f"max_x_max={metrics['max_x_max']} "
                    f"reward_mean={metrics['reward_mean']:.2f}",
                    flush=True,
                )

            print(
                "eval "
                f"step={checkpoint_step} "
                f"reward_mean={metrics['reward_mean']:.2f} "
                f"max_x_mean={metrics['max_x_mean']:.2f} "
                f"max_x_max={metrics['max_x_max']} "
                f"completion_rate={metrics['completion_rate']:.3f} "
                f"death_rate={metrics['death_rate']:.3f}",
                flush=True,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
