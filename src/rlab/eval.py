from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np

from rlab.cli_args import explicit_arg_dests, parse_json_value
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_provider_runtime_available,
    resolve_env_config,
    task_action_set,
)
from rlab.env_config import env_config_from_args, parse_obs_crop
from rlab.eval_runner import evaluate_model_episodes
from rlab.model_sources import (
    add_model_source_args,
    apply_model_source_defaults,
    model_source_ref,
    resolve_single_model_source,
)
from rlab.sb3_models import load_sb3_model
from rlab.seeds import DEFAULT_EVAL_SEED, validate_eval_seed
from rlab.targets import target_for_game
from rlab.train_config import add_env_config_args


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def default_eval_n_envs() -> int:
    return max(os.cpu_count() or 1, 1)


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


class ScriptedPolicy:
    """A one-lane policy adapter for the shared model-evaluation driver."""

    def __init__(self, policy: str, action_names: tuple[str, ...]) -> None:
        self.policy = policy
        self.action_names = action_names
        self.action_space = None
        self.step_idx = 0

    def bind_action_space(self, action_space) -> None:
        self.action_space = action_space

    def reset_episode(self) -> None:
        self.step_idx = 0

    def predict(self, _obs, deterministic: bool):
        del deterministic
        if self.policy == "random":
            if self.action_space is None:
                raise RuntimeError("scripted policy action space was not bound")
            action = self.action_space.sample()
        else:
            action = scripted_action(self.policy, self.step_idx, self.action_names)
        self.step_idx += 1
        return np.asarray([action]), None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab eval",
        description="Evaluate PPO or scripted provider baselines",
    )
    add_model_source_args(
        parser,
        positional_artifact=True,
        model_help="Path to PPO .zip model",
        default_kind="checkpoint",
        include_wandb_artifacts=False,
    )
    parser.add_argument("--policy", choices=["random", "right", "noop"], default="random")
    parser.add_argument("--episodes", type=int, default=20)
    add_env_config_args(
        parser,
        max_steps_default=4500,
        parse_json_value=parse_json_value,
        parse_obs_crop=parse_obs_crop,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base eval seed. The default lives in the eval-reserved seed range "
            ">= 10000; train seeds are forbidden from that range."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--n-envs",
        type=int,
        default=default_eval_n_envs(),
        help="Number of vectorized eval envs; defaults to the logical CPU core count.",
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-scale", type=int, default=4)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, argv_list)
    args = parser.parse_args(argv_list)
    args.seed = validate_eval_seed(args.seed)
    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")
    ref = model_source_ref(args)
    if ref is not None or args.model:
        if ref is not None:
            print(f"Downloading {ref}", flush=True)
        source = resolve_single_model_source(args)
        args.model = str(source.model_path)
        if ref is not None:
            print(f"Downloaded model: {args.model}", flush=True)
        apply_model_source_defaults(
            args,
            source,
            parser,
            parser_defaults,
            explicit_dests,
        )
    config = resolve_env_config(
        env_config_from_args(
            args,
            include_states=True,
        )
    )
    assert_provider_runtime_available(config)
    model = (
        load_sb3_model(args.model, device=resolve_sb3_device(args.device)) if args.model else None
    )

    if model is not None:
        summary, _ = evaluate_model_episodes(
            model=model,
            config=config,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            deterministic=False,
            n_envs=args.n_envs,
            progress=args.progress,
            progress_description="eval model",
            extra={
                "model": args.model,
                "policy": "ppo",
                "hud_crop_top": args.hud_crop_top,
            },
        )
    else:
        action_names = target_for_game(config.game).action_names_for_set(task_action_set(config))
        policy = ScriptedPolicy(args.policy, action_names)
        summary, _ = evaluate_model_episodes(
            model=policy,
            config=config,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            deterministic=False,
            n_envs=1,
            progress=args.progress,
            progress_description=f"eval {args.policy}",
            extra={
                "model": args.model,
                "policy": args.policy,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
