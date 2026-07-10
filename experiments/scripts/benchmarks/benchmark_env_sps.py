from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from rlab.env import (
    EnvConfig,
    make_provider_vec_env,
    make_training_vec_env,
    native_obs_crop,
    resolve_env_config,
    state_weight_mapping,
)
from rlab.env_providers import provider_native_vec_kwargs
from rlab.targets import target_for_game


def benchmark_config(args: argparse.Namespace) -> EnvConfig:
    return resolve_env_config(
        EnvConfig(
            env_provider=args.env_provider,
            game=args.game,
            state=args.state,
            frame_skip=4,
            max_pool_frames=False,
            sticky_action_prob=0.0,
            observation_size=84,
            obs_crop=(32, 0, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
            obs_resize_algorithm="area",
            task={
                "id": "mario",
                "action": {"set": "simple"},
                "signals": {
                    "x": ["xscrollHi", "xscrollLo"],
                    "score": "score",
                    "lives": "lives",
                    "level": ["levelHi", "levelLo"],
                },
                "events": {
                    "life_loss": {"signal": "lives", "operation": "decrease"},
                    "level_change": {"signal": "level", "operation": "change"},
                },
                # Engine-only endings keep provider and runtime reset workloads aligned.
                "termination": {"failure": [], "success": [], "max_episode_steps": 0},
                "reward": {"reward_mode": "native"},
            },
        )
    )


def action_batches(config: EnvConfig, *, envs: int, count: int, seed: int):
    masks = np.asarray(target_for_game(config.game).action_masks_for_set("simple"), dtype=np.int8)
    rng = np.random.default_rng(seed)
    policy_actions = rng.integers(0, len(masks), size=(count, envs), dtype=np.int64)
    return policy_actions, masks[policy_actions]


def reset_seeds(seed: int, envs: int, episode_indices: np.ndarray) -> list[int]:
    return [
        int(np.random.SeedSequence([seed, lane, int(episode_indices[lane])]).generate_state(1)[0])
        for lane in range(envs)
    ]


def run_provider(
    config: EnvConfig,
    *,
    envs: int,
    seed: int,
    warmup_actions: np.ndarray,
    measured_actions: np.ndarray,
) -> dict[str, Any]:
    kwargs = provider_native_vec_kwargs(
        config,
        n_envs=envs,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=state_weight_mapping,
    )
    env = make_provider_vec_env(config, native_kwargs=kwargs)
    episode_indices = np.zeros(envs, dtype=np.int64)
    reset_count = 0

    def step(actions: np.ndarray) -> None:
        nonlocal reset_count
        _obs, _rewards, terminated, truncated, _infos = env.step(actions)
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        if not np.any(done):
            return
        episode_indices[done] += 1
        seeds = [
            value if bool(done[lane]) else None
            for lane, value in enumerate(reset_seeds(seed, envs, episode_indices))
        ]
        env.reset(seed=seeds, options={"reset_mask": done.copy()})
        reset_count += int(np.sum(done))

    try:
        env.reset(seed=reset_seeds(seed, envs, episode_indices))
        for actions in warmup_actions:
            step(actions)
        started_at = time.perf_counter()
        for actions in measured_actions:
            step(actions)
        elapsed = time.perf_counter() - started_at
        return {
            "elapsed_seconds": elapsed,
            "steps_per_second": envs * len(measured_actions) / elapsed,
            "reset_lanes": reset_count,
        }
    finally:
        env.close()


def run_runtime(
    config: EnvConfig,
    *,
    envs: int,
    seed: int,
    warmup_actions: np.ndarray,
    measured_actions: np.ndarray,
) -> dict[str, Any]:
    env = make_training_vec_env(config, n_envs=envs, seed=seed)
    reset_count = 0

    def step(actions: np.ndarray) -> None:
        nonlocal reset_count
        _obs, _rewards, dones, _infos = env.step(actions)
        reset_count += int(np.sum(dones))
        env.drain_records()

    try:
        env.reset()
        for actions in warmup_actions:
            step(actions)
        started_at = time.perf_counter()
        for actions in measured_actions:
            step(actions)
        elapsed = time.perf_counter() - started_at
        return {
            "elapsed_seconds": elapsed,
            "steps_per_second": envs * len(measured_actions) / elapsed,
            "reset_lanes": reset_count,
        }
    finally:
        env.close()


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "median_elapsed_seconds": statistics.median(
            float(sample["elapsed_seconds"]) for sample in samples
        ),
        "median_steps_per_second": statistics.median(
            float(sample["steps_per_second"]) for sample in samples
        ),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark native provider and consolidated batch-runtime steps/sec."
    )
    parser.add_argument("--env-provider", default="supermariobrosnes-turbo")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--mode", choices=("provider", "runtime", "compare"), default="compare")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-overhead", type=float, default=0.05)
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 1 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("envs, steps, and repeats must be positive; warmup must be non-negative")

    config = benchmark_config(args)
    policy_actions, native_actions = action_batches(
        config,
        envs=args.envs,
        count=args.warmup + args.steps,
        seed=args.seed,
    )
    policy_warmup, policy_measured = policy_actions[: args.warmup], policy_actions[args.warmup :]
    native_warmup, native_measured = native_actions[: args.warmup], native_actions[args.warmup :]
    runners = {
        "provider": lambda: run_provider(
            config,
            envs=args.envs,
            seed=args.seed,
            warmup_actions=native_warmup,
            measured_actions=native_measured,
        ),
        "runtime": lambda: run_runtime(
            config,
            envs=args.envs,
            seed=args.seed,
            warmup_actions=policy_warmup,
            measured_actions=policy_measured,
        ),
    }
    modes = ("provider", "runtime") if args.mode == "compare" else (args.mode,)
    samples: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            samples[mode].append(runners[mode]())

    result: dict[str, Any] = {
        "benchmark_contract": "rlab.native-vector-runtime.v2",
        "environment_contract": {
            "schema_version": 2,
            "task_termination_boundary": "vector_step",
        },
        "mode": args.mode,
        "envs": args.envs,
        "steps_per_env": args.steps,
        "warmup_steps_per_env": args.warmup,
        "repeats": args.repeats,
        "config": asdict(config),
        "results": {mode: summarize(values) for mode, values in samples.items()},
    }
    if args.mode == "compare":
        provider_elapsed = result["results"]["provider"]["median_elapsed_seconds"]
        runtime_elapsed = result["results"]["runtime"]["median_elapsed_seconds"]
        overhead = runtime_elapsed / provider_elapsed - 1.0
        result["runtime_overhead_fraction"] = overhead
        result["max_runtime_overhead_fraction"] = args.max_overhead
        result["overhead_gate_passed"] = overhead <= args.max_overhead
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
