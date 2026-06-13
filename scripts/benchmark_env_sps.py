from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_retro import StableRetroSubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from mario_ppo.env import (
    GAME,
    DiscreteMarioActions,
    EnvConfig,
    FrameSkip,
    MarioProgressInfo,
    action_names_for_set,
    assert_rom_imported,
    make_fast_mario_env,
    make_mario_env,
)


def make_raw_env(seed: int) -> gym.Env:
    env = retro.make(GAME, state="Level1-1", render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def make_python_preprocessed_env(seed: int) -> gym.Env:
    config = EnvConfig(max_episode_steps=4500, terminate_on_life_loss=False)
    env = make_mario_env(config=config, seed=seed)
    env.reset(seed=seed)
    return env


def make_retro_preprocessed_env(seed: int, resize_algorithm: str) -> gym.Env:
    config = EnvConfig(max_episode_steps=4500, terminate_on_life_loss=False)
    env = retro.make(
        config.game,
        state=config.state,
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=resize_algorithm,
    )
    env = DiscreteMarioActions(env, action_set=config.action_set)
    env = FrameSkip(env, config.frame_skip)
    env = MarioProgressInfo(
        env,
        use_retro_reward=config.use_retro_reward,
        reward_mode=config.reward_mode,
        progress_reward_cap=config.progress_reward_cap,
        progress_reward_scale=config.progress_reward_scale,
        terminal_reward=config.terminal_reward,
        reward_scale=config.reward_scale,
        time_penalty=config.time_penalty,
        death_penalty=config.death_penalty,
        completion_reward=config.completion_reward,
        completion_x_threshold=config.completion_x_threshold,
        terminate_on_life_loss=config.terminate_on_life_loss,
        terminate_on_level_change=config.terminate_on_level_change,
        terminate_on_completion=config.terminate_on_completion,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env


def make_fast_preprocessed_env(seed: int) -> gym.Env:
    config = EnvConfig(max_episode_steps=4500, terminate_on_life_loss=False)
    env = make_fast_mario_env(config=config, seed=seed)
    env.reset(seed=seed)
    return env


def make_env(mode: str, seed: int, resize_algorithm: str) -> gym.Env:
    if mode == "python":
        return make_python_preprocessed_env(seed)
    if mode == "retro":
        return make_retro_preprocessed_env(seed, resize_algorithm)
    if mode == "fast":
        return make_fast_preprocessed_env(seed)
    return make_raw_env(seed)


def sample_action(space: gym.Space, rng: np.random.Generator):
    if isinstance(space, gym.spaces.Discrete):
        return int(rng.integers(space.n))
    if isinstance(space, gym.spaces.MultiBinary):
        return rng.integers(0, 2, size=space.shape, dtype=space.dtype)
    return space.sample()


def bench_single(
    mode: str,
    steps: int,
    warmup: int,
    seed: int,
    resize_algorithm: str,
) -> dict[str, object]:
    env = make_env(mode, seed, resize_algorithm)
    rng = np.random.default_rng(seed)
    action_space = env.action_space
    obs, _ = env.reset(seed=seed)
    for _ in range(warmup):
        obs, _, terminated, truncated, _ = env.step(sample_action(action_space, rng))
        if terminated or truncated:
            obs, _ = env.reset()

    start = time.perf_counter()
    resets = 0
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(sample_action(action_space, rng))
        if terminated or truncated:
            resets += 1
            obs, _ = env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "mode": mode,
        "envs": 1,
        "steps": steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps / elapsed,
        "resets": resets,
        "obs_shape": tuple(obs.shape),
        "obs_dtype": str(obs.dtype),
    }


def make_vec_env_fn(mode: str, rank: int, seed: int):
    def _init() -> gym.Env:
        env_seed = seed + rank
        return make_env(mode, env_seed, make_vec_env_fn.resize_algorithm)

    return _init


make_vec_env_fn.resize_algorithm = "area"


def bench_vector(
    mode: str,
    envs: int,
    steps_per_env: int,
    warmup: int,
    seed: int,
    resize_algorithm: str,
) -> dict[str, object]:
    make_vec_env_fn.resize_algorithm = resize_algorithm
    vec_env_cls = StableRetroSubprocVecEnv if mode == "fast" else SubprocVecEnv
    vec_env = vec_env_cls(
        [make_vec_env_fn(mode, rank, seed) for rank in range(envs)],
        start_method="fork",
    )
    rng = np.random.default_rng(seed)
    obs = vec_env.reset()

    if mode in {"python", "retro", "fast"}:
        action_count = len(action_names_for_set("simple"))

        def make_actions():
            return rng.integers(0, action_count, size=(envs,))

    else:
        action_shape = vec_env.action_space.shape
        action_dtype = vec_env.action_space.dtype

        def make_actions():
            return rng.integers(0, 2, size=(envs, *action_shape), dtype=action_dtype)

    for _ in range(warmup):
        obs, _, _, _ = vec_env.step(make_actions())

    vec_steps = steps_per_env
    total_env_steps = envs * vec_steps
    start = time.perf_counter()
    for _ in range(vec_steps):
        obs, _, _, _ = vec_env.step(make_actions())
    elapsed = time.perf_counter() - start
    vec_env.close()
    return {
        "mode": mode,
        "envs": envs,
        "steps": total_env_steps,
        "vec_steps": vec_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": total_env_steps / elapsed,
        "obs_shape": tuple(obs.shape),
        "obs_dtype": str(obs.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SuperMarioBros-Nes-v0 env steps/sec.")
    parser.add_argument("--mode", choices=["raw", "python", "retro", "fast"], default="python")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--retro-resize-algorithm",
        choices=["nearest", "bilinear", "area"],
        default="area",
    )
    args = parser.parse_args()

    assert_rom_imported()
    if args.envs == 1:
        result = bench_single(
            args.mode,
            args.steps,
            args.warmup,
            args.seed,
            args.retro_resize_algorithm,
        )
    else:
        result = bench_vector(
            args.mode,
            args.envs,
            args.steps,
            args.warmup,
            args.seed,
            args.retro_resize_algorithm,
        )
    result["package_version"] = getattr(retro, "__version__", "").strip()
    if args.mode == "retro":
        result["retro_resize_algorithm"] = args.retro_resize_algorithm
    result["config"] = {
        "game": GAME,
        "benchmark": asdict(EnvConfig(max_episode_steps=4500, terminate_on_life_loss=False)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
