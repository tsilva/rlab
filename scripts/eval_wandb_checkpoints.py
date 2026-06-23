from __future__ import annotations

# ruff: noqa: E402

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import (
    apply_model_config_defaults,
    explicit_arg_dests,
)
from stable_retro_ppo.cli_args import add_env_config_args
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import assert_rom_imported, resolve_env_config
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.eval_metrics import flat_numeric_metrics
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.json_utils import json_safe
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    artifact_qualified_name,
    checkpoint_step_from_artifact,
    download_artifact_model,
    safe_artifact_stem,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def slug(value: str) -> str:
    return safe_artifact_stem(value)


def split_project(value: str) -> tuple[str | None, str]:
    parts = value.split("/", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


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
    name = artifact_qualified_name(artifact)
    return download_artifact_model(artifact, artifact_download_dir(root, name))


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
        file.write(json.dumps(json_safe(metrics)) + "\n")


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


def eval_seed_for_checkpoint(args: argparse.Namespace, checkpoint_step: int) -> int:
    if args.seed_offset_by_checkpoint_step:
        return args.seed + checkpoint_step
    return args.seed


def evaluate_checkpoint(
    args: argparse.Namespace,
    model_path: Path,
    checkpoint_step: int,
    artifact_name: str,
) -> tuple[dict[str, Any], Path | None]:
    model = PPO.load(model_path, device=resolve_sb3_device(args.device))
    config = resolve_env_config(env_config_from_args(args, max_episode_steps_attr="max_steps"))
    eval_seed = eval_seed_for_checkpoint(args, checkpoint_step)
    video_path = (
        Path(args.eval_dir) / args.run_name / "videos" / f"best_episode_{checkpoint_step}_steps.mp4"
        if args.record_best_video
        else None
    )
    metrics, video_path = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=args.episodes,
        seed=eval_seed,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        completion_x_threshold=config.completion_x_threshold,
        capture_best_video=args.record_best_video,
        video_path=video_path,
        video_fps=args.video_fps,
        video_scale=args.video_scale,
        extra={
            "checkpoint_step": checkpoint_step,
            "checkpoint_artifact": artifact_name,
            "model": str(model_path),
            "hud_crop_top": args.hud_crop_top,
            "eval_seed": eval_seed,
        },
    )
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
    payload.update(flat_numeric_metrics(metrics, "eval/"))
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
    parser = argparse.ArgumentParser(
        description="Evaluate pending W&B Stable Retro PPO checkpoints"
    )
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
    add_env_config_args(parser, max_steps_default=2500)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument(
        "--seed-offset-by-checkpoint-step",
        action="store_true",
        help=(
            "Use the legacy eval seed schedule of --seed + checkpoint_step. "
            "By default, all checkpoints use the same eval seed schedule for fair comparison."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--deterministic", action="store_true", help="Use greedy policy actions")
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
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    explicit_dests.update(
        {"terminate_on_life_loss", "terminate_on_level_change", "terminate_on_completion"}
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if not args.run_name and args.artifact:
        args.run_name = slug(args.artifact[0].split("/")[-1].split("-checkpoint", 1)[0])
    if not args.run_name:
        raise SystemExit("run_name is required")

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

            checkpoint_args = copy.copy(args)
            apply_model_config_defaults(
                checkpoint_args,
                model_path,
                parser_defaults,
                explicit_dests,
            )
            assert_rom_imported(checkpoint_args.game)
            print(f"Evaluating checkpoint step {checkpoint_step}: {artifact_name}", flush=True)
            previous_best = best_metrics(history)
            metrics, video_path = evaluate_checkpoint(
                checkpoint_args, model_path, checkpoint_step, artifact_name
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
