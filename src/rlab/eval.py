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

import numpy as np
from stable_baselines3 import PPO

from rlab.artifacts import apply_model_config_defaults, explicit_arg_dests
from rlab.cli_args import add_env_config_args
from rlab.device import resolve_sb3_device
from rlab.env import (
    action_names_for_set,
    assert_rom_imported,
    make_eval_vec_env,
    resolve_env_config,
)
from rlab.env_config import env_config_from_args
from rlab.eval_metrics import (
    flat_numeric_metrics,
    is_level_complete,
    run_eval_episode,
    summarize_episode_results,
)
from rlab.eval_runner import evaluate_model_episodes
from rlab.json_utils import json_safe
from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_BEST_VIDEO,
    EVAL_BEST_X,
    EVAL_CHECKPOINT_ARTIFACT,
    EVAL_CHECKPOINT_STEP,
    EVAL_CONFIG_HUD_CROP_TOP,
    EVAL_DEATH_COUNT,
    EVAL_DEATH_RATE,
    EVAL_DEATH_X_HIST,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    EVAL_PROGRESS_LEVEL_X_MAX,
    EVAL_PROGRESS_LEVEL_X_MEAN,
    EVAL_PROGRESS_X_MAX,
    EVAL_PROGRESS_X_MEAN,
    EVAL_REWARD_MAX,
    EVAL_REWARD_MEAN,
    EVAL_REWARD_STD,
)
from rlab.wandb_artifacts import (
    artifact_download_dir,
    artifact_qualified_name,
    checkpoint_step_from_artifact,
    download_artifact_model,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH, load_wandb_env


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def slug(value: str) -> str:
    return safe_artifact_stem(value)


def split_project(value: str) -> tuple[str | None, str]:
    parts = value.split("/", 1)
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def artifact_eval_name(args: argparse.Namespace) -> str:
    if args.artifact_run:
        return slug(args.artifact_run)
    if args.artifact:
        leaf = args.artifact[0].split("/")[-1].split(":", 1)[0]
        for suffix in ("-checkpoint", "-final", "-best"):
            if suffix in leaf:
                leaf = leaf.split(suffix, 1)[0]
                break
        return slug(leaf)
    raise ValueError("artifact eval requires --artifact or --artifact-run")


def checkpoint_artifact_ref(args: argparse.Namespace) -> str:
    if not args.artifact_run:
        raise SystemExit("--artifact-run is required unless --artifact is provided")
    if args.checkpoint_series:
        return f"{args.artifact_project}/{slug(args.artifact_run)}-checkpoint"
    return model_artifact_ref(
        project=args.artifact_project,
        run_name=args.artifact_run,
        kind=args.artifact_kind,
        version=args.artifact_version,
    )


def find_checkpoint_artifacts(args: argparse.Namespace):
    load_wandb_env()

    import wandb

    api = wandb.Api()
    if args.artifact:
        return [api.artifact(ref, type="model") for ref in args.artifact]

    ref = checkpoint_artifact_ref(args)
    if not args.checkpoint_series:
        return [api.artifact(ref, type="model")]

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
        float(metrics.get(EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN, metrics["completion_rate"])),
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
        Path(args.eval_dir)
        / args.eval_run_name
        / "videos"
        / f"best_episode_{checkpoint_step}_steps.mp4"
        if args.record_best_video
        else None
    )
    metrics, video_path = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=args.episodes,
        seed=eval_seed,
        max_steps=args.max_steps,
        deterministic=eval_deterministic(args),
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
    entity, project = split_project(args.artifact_project)
    return wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=args.eval_run_name,
        resume="allow",
        mode=args.wandb_mode,
    )


def log_wandb_eval(wandb_run, metrics: dict[str, Any], video_path: Path | None) -> None:
    if wandb_run is None:
        return
    import wandb

    payload: dict[str, Any] = {
        EVAL_REWARD_MEAN: metrics["reward_mean"],
        EVAL_REWARD_STD: metrics["reward_std"],
        EVAL_REWARD_MAX: metrics["reward_max"],
        EVAL_PROGRESS_X_MEAN: metrics["max_x_mean"],
        EVAL_PROGRESS_X_MAX: metrics["max_x_max"],
        EVAL_PROGRESS_LEVEL_X_MEAN: metrics["max_level_x_mean"],
        EVAL_PROGRESS_LEVEL_X_MAX: metrics["max_level_x_max"],
        EVAL_DEATH_COUNT: metrics["death_count"],
        EVAL_DEATH_RATE: metrics["death_rate"],
        EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
        EVAL_BEST_X: metrics["best_episode"]["max_x_pos"],
        EVAL_CHECKPOINT_STEP: metrics["checkpoint_step"],
        EVAL_CHECKPOINT_ARTIFACT: metrics["checkpoint_artifact"],
        EVAL_CONFIG_HUD_CROP_TOP: metrics["hud_crop_top"],
    }
    payload.update(flat_numeric_metrics(metrics, "eval/done/"))
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in metrics["episode_results"]
        if episode.get("death_x_pos") is not None
    ]
    if death_x_positions:
        payload[EVAL_DEATH_X_HIST] = wandb.Histogram(death_x_positions)
    if video_path is not None and video_path.is_file():
        payload[EVAL_BEST_VIDEO] = wandb.Video(str(video_path), format="mp4")
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
        f"{slug(args.eval_run_name)}-best",
        type="model",
        metadata={
            "run_name": args.eval_run_name,
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


def eval_deterministic(args: argparse.Namespace) -> bool:
    return bool(args.deterministic)


def scripted_action(policy: str, step_idx: int, action_names: tuple[str, ...]) -> int:
    if policy == "random":
        raise ValueError("random policy is sampled from the env action space")
    if policy == "noop":
        return action_names.index("noop")
    if policy == "right":
        # Mostly sprint right, with periodic jumps to clear early obstacles.
        if step_idx % 55 in range(30, 42):
            return action_names.index("right_a_b")
        return action_names.index("right_b")
    raise ValueError(f"unknown policy: {policy}")


def run_scripted_episode(
    env,
    policy: str,
    max_steps: int,
    action_names: tuple[str, ...],
    completion_x_threshold: int,
    default_start_state: str | None = None,
):
    obs = env.reset()
    total_reward = 0.0
    max_x = 0
    max_level_x = 0
    final_info = {}
    for step_idx in range(max_steps):
        if policy == "random":
            action = env.action_space.sample()
        else:
            action = scripted_action(policy, step_idx, action_names)
        obs, rewards, dones, infos = env.step([action])
        info = dict(infos[0])
        total_reward += float(rewards[0])
        max_x = max(max_x, int(info.get("max_x_pos", 0)))
        max_level_x = max(max_level_x, int(info.get("level_max_x_pos", 0)))
        final_info = info
        if bool(dones[0]):
            break
    completed = is_level_complete(final_info, max_x, completion_x_threshold)
    died = bool(final_info.get("died", False))
    death_x_pos = final_info.get("death_x_pos")
    if died and death_x_pos is None:
        death_x_pos = max_x
    return {
        "start_state": final_info.get("start_state")
        or final_info.get("state")
        or default_start_state,
        "reward": total_reward,
        "max_x_pos": max_x,
        "max_level_x_pos": max_level_x,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "steps": step_idx + 1,
        "level_complete": completed,
        "died": died,
        "death_x_pos": int(death_x_pos) if death_x_pos is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO or scripted Stable Retro baselines")
    parser.add_argument("--model", help="Path to PPO .zip model")
    parser.add_argument(
        "--artifact",
        action="append",
        help="Full W&B model artifact ref to evaluate. May be passed more than once.",
    )
    parser.add_argument(
        "--artifact-run",
        help="Training run name used to build a W&B checkpoint artifact ref.",
    )
    parser.add_argument("--artifact-project", default=DEFAULT_WANDB_PROJECT_PATH)
    parser.add_argument(
        "--artifact-kind",
        choices=["final", "best", "checkpoint"],
        default="checkpoint",
    )
    parser.add_argument("--artifact-version", default="latest")
    parser.add_argument("--artifact-root", default="runs/wandb_artifacts")
    parser.add_argument(
        "--checkpoint-series",
        action="store_true",
        help="With --artifact-run, evaluate every checkpoint version instead of one version.",
    )
    parser.add_argument("--eval-dir", default="runs/local_evals")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate checkpoint steps already logged locally."
    )
    parser.add_argument("--policy", choices=["random", "right", "noop"], default="random")
    parser.add_argument("--episodes", type=int, default=20)
    add_env_config_args(parser, max_steps_default=4500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seed-offset-by-checkpoint-step",
        action="store_true",
        help=(
            "Use the legacy eval seed schedule of --seed + checkpoint_step. "
            "By default, checkpoint artifacts use the same eval seed schedule for fair comparison."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic argmax actions instead of stochastic policy sampling.",
    )
    parser.add_argument("--record-best-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-scale", type=int, default=4)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-run-path", help="W&B run path, e.g. entity/project/runs/<id>")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--no-wandb-log", action="store_true")
    parser.add_argument("--no-promote-best", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print one progress line per completed episode to stderr.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-episode details from stdout JSON.",
    )
    return parser


def run_checkpoint_artifact_eval(
    args: argparse.Namespace,
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
) -> None:
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    args.eval_run_name = artifact_eval_name(args)
    artifacts = find_checkpoint_artifacts(args)
    if not artifacts:
        print("No checkpoint artifacts found")
        return

    history_path = Path(args.eval_dir) / args.eval_run_name / "checkpoint_eval_metrics.jsonl"
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

            model_path = download_artifact(artifact, Path(args.artifact_root))
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


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    explicit_dests.add("done_on_info_json")
    args = parser.parse_args()
    if args.artifact or args.artifact_run:
        run_checkpoint_artifact_eval(args, parser_defaults, explicit_dests)
        return
    if args.model:
        apply_model_config_defaults(args, Path(args.model), parser_defaults, explicit_dests)
    assert_rom_imported(args.game)
    config = resolve_env_config(
        env_config_from_args(
            args,
            max_episode_steps_attr="max_steps",
            include_states=True,
        )
    )
    model = PPO.load(args.model, device=resolve_sb3_device(args.device)) if args.model else None

    if model is not None:
        env = make_eval_vec_env(config=config, n_envs=1, seed=args.seed)
        try:
            episodes = []
            for episode_idx in range(args.episodes):
                result = run_eval_episode(
                    env,
                    model=model,
                    max_steps=args.max_steps,
                    deterministic=eval_deterministic(args),
                    seed=args.seed + episode_idx,
                    completion_x_threshold=config.completion_x_threshold,
                    default_start_state=config.state,
                )
                result.pop("actions")
                episodes.append(result)
                if args.progress:
                    print(
                        "eval_episode="
                        f"{episode_idx + 1}/{args.episodes} "
                        f"state={result.get('start_state')} "
                        f"complete={bool(result.get('level_complete'))} "
                        f"max_x={int(result.get('max_x_pos', 0))} "
                        f"steps={int(result.get('steps', 0))}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            env.close()
    else:
        action_names = action_names_for_set(args.action_set, game=args.game)
        env = make_eval_vec_env(config=config, n_envs=1, seed=args.seed)
        episodes = [
            run_scripted_episode(
                env,
                policy=args.policy,
                max_steps=args.max_steps,
                action_names=action_names,
                completion_x_threshold=config.completion_x_threshold,
                default_start_state=config.state,
            )
            for _ in range(args.episodes)
        ]
        env.close()

    summary = summarize_episode_results(
        episodes,
        deterministic=bool(args.model and eval_deterministic(args)),
        extra={
            "model": args.model,
            "policy": "ppo" if args.model else args.policy,
            "hud_crop_top": args.hud_crop_top,
        },
    )
    if args.summary_only:
        summary.pop("episode_results", None)
    print(json.dumps(summary, indent=2, default=json_default))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
