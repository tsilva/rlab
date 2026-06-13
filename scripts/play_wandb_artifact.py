from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a W&B model artifact and play it locally")
    parser.add_argument("run_name", nargs="?", help="Training run name, e.g. modal_gpu_short_improve")
    parser.add_argument("--project", default="tsilva/mario-ppo", help="W&B entity/project")
    parser.add_argument("--artifact", help="Full artifact ref, overriding run_name/kind/project")
    parser.add_argument("--kind", choices=["final", "best", "checkpoint"], default="final")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--root", default="runs/wandb_artifacts")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-seeds", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--reward-mode", choices=["bounded", "additive", "score"], default="bounded")
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=30.0)
    parser.add_argument("--reward-scale", type=float, default=30.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--completion-x-threshold", type=int, default=0)
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", choices=["simple", "right"], default="simple")
    parser.add_argument("--download-only", action="store_true")
    return parser


def artifact_ref(args: argparse.Namespace) -> str:
    if args.artifact:
        return args.artifact
    if not args.run_name:
        raise SystemExit("run_name is required unless --artifact is provided")
    return f"{args.project}/{args.run_name}-{args.kind}:{args.version}"


def download_artifact(ref: str, root: Path) -> Path:
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


def play_model(model_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mario_ppo.play",
        "--model",
        str(model_path),
        "--episodes",
        str(args.episodes),
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
        "--completion-x-threshold",
        str(args.completion_x_threshold),
    ]
    if args.stochastic:
        cmd.append("--stochastic")
    if not args.max_pool_frames:
        cmd.append("--no-max-pool-frames")
    if args.random_seeds:
        cmd.append("--random-seeds")
    if args.no_terminate_on_life_loss:
        cmd.append("--no-terminate-on-life-loss")
    if args.terminate_on_level_change:
        cmd.append("--terminate-on-level-change")
    if args.terminate_on_completion:
        cmd.append("--terminate-on-completion")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = build_parser().parse_args()
    ref = artifact_ref(args)
    download_root = Path(args.root) / slug(ref.replace("/", "_").replace(":", "_"))
    print(f"Downloading {ref} to {download_root}")
    model_path = download_artifact(ref, download_root)
    print(f"Downloaded model: {model_path}")
    if not args.download_only:
        play_model(model_path, args)


if __name__ == "__main__":
    main()
