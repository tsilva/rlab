from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import apply_model_config_defaults, explicit_arg_dests
from stable_retro_ppo.cli_args import add_env_config_args
from stable_retro_ppo.device import resolve_sb3_device
from stable_retro_ppo.env import (
    action_names_for_set,
    assert_rom_imported,
    make_eval_vec_env,
    resolve_env_config,
)
from stable_retro_ppo.env_config import env_config_from_args
from stable_retro_ppo.eval_metrics import (
    is_level_complete,
    run_eval_episode,
    summarize_episode_results,
)


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
    parser.add_argument("--policy", choices=["random", "right", "noop"], default="random")
    parser.add_argument("--episodes", type=int, default=20)
    add_env_config_args(parser, max_steps_default=4500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample from the policy; use --no-stochastic for deterministic argmax eval.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, sys.argv[1:])
    explicit_dests.update(
        {"terminate_on_life_loss", "terminate_on_level_change", "terminate_on_completion"}
    )
    args = parser.parse_args()
    if args.model:
        apply_model_config_defaults(args, Path(args.model), parser_defaults, explicit_dests)
    assert_rom_imported(args.game)
    config = resolve_env_config(env_config_from_args(args, max_episode_steps_attr="max_steps"))
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
                    completion_x_threshold=config.completion_x_threshold,
                    default_start_state=config.state,
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
                completion_x_threshold=config.completion_x_threshold,
                default_start_state=config.state,
            )
            for _ in range(args.episodes)
        ]
        env.close()

    summary = summarize_episode_results(
        episodes,
        deterministic=bool(args.model and not args.stochastic),
        state_metric_root="eval",
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
