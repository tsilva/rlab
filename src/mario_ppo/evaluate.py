from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO

from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import (
    DEFAULT_HUD_CROP_TOP,
    action_names_for_set,
    assert_rom_imported,
    make_eval_vec_env,
)
from mario_ppo.env_config import env_config_from_args
from mario_ppo.eval_metrics import is_level_complete, run_eval_episode, summarize_episode_results


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
    parser = argparse.ArgumentParser(description="Evaluate Mario PPO or scripted baselines")
    parser.add_argument("--model", help="Path to PPO .zip model")
    parser.add_argument("--policy", choices=["random", "right", "noop"], default="random")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample from the policy; use --no-stochastic for deterministic argmax eval.",
    )
    parser.add_argument("--use-retro-reward", action="store_true")
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
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", choices=["simple", "right", "native"], default="simple")
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=3160,
        help="Treat an episode as level-complete if max_x_pos reaches this value; set <=0 to disable.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    config = env_config_from_args(args, max_episode_steps_attr="max_steps")
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
                    deterministic=not args.stochastic,
                    seed=args.seed + episode_idx,
                    completion_x_threshold=args.completion_x_threshold,
                )
                result.pop("actions")
                episodes.append(result)
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
                completion_x_threshold=args.completion_x_threshold,
            )
            for _ in range(args.episodes)
        ]
        env.close()

    summary = summarize_episode_results(
        episodes,
        deterministic=bool(args.model and not args.stochastic),
        extra={
            "model": args.model,
            "policy": "ppo" if args.model else args.policy,
            "hud_crop_top": args.hud_crop_top,
        },
    )
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
