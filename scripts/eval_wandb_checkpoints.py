from __future__ import annotations

# ruff: noqa: E402

import argparse
import copy
import json
import os
import re
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
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import EnvConfig, assert_rom_imported, resolve_env_config
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.eval_metrics import flat_numeric_metrics
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.wandb_artifacts import (
    model_zip_from_download,
    safe_artifact_stem,
    write_downloaded_artifact_metadata,
)
from stable_retro_ppo.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def slug(value: str) -> str:
    return safe_artifact_stem(value)


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
    model_path = model_zip_from_download(Path(artifact.download(root=str(download_root))))
    write_downloaded_artifact_metadata(model_path, artifact)
    return model_path


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
        }
    return value


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
    defaults = EnvConfig()
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
    parser.add_argument("--game", default=defaults.game)
    parser.add_argument("--state", default=defaults.state)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sticky-action-prob",
        type=float,
        default=defaults.sticky_action_prob,
        help="Probability of replaying the previous high-level action; 0 disables sticky actions.",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--observation-size", type=int, default=defaults.observation_size)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=defaults.hud_crop_top,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--obs-resize-algorithm", default=defaults.obs_resize_algorithm)
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
    parser.add_argument("--action-set", default=defaults.action_set)
    parser.add_argument("--use-retro-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-rewards", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--score-progress-clipped", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--no-progress-timeout-steps", type=int, default=0)
    parser.add_argument("--no-progress-min-delta", type=int, default=0)
    parser.add_argument(
        "--terminate-on-life-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--terminate-on-level-change", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--terminate-on-completion", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=defaults.completion_x_threshold,
        help="Deprecated no-op; level completion is detected from stable-retro level changes.",
    )
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
