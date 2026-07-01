from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from typing import Any

import numpy as np
from stable_retro import RetroVecEnv

from rlab.env import EnvConfig, action_names_for_set, make_fast_retro_env, make_vec_envs
from rlab.targets import SuperMarioBrosNesV0Target, target_for_game


def sha_array(array: Any) -> str:
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def normalize_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "xscrollHi",
        "xscrollLo",
        "x_pos",
        "level_x_pos",
        "max_x_pos",
        "level_max_x_pos",
        "progress_delta",
        "score",
        "score_delta",
        "lives",
        "levelHi",
        "levelLo",
        "died",
        "level_complete",
        "completion_event",
        "level_changed",
        "raw_reward",
        "shaped_reward",
        "reward_mode",
        "TimeLimit.truncated",
        "_native_done",
        "global_reset",
    )
    normalized: dict[str, Any] = {}
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, np.generic):
            value = value.item()
        normalized[key] = value
    return normalized


def config_from_args(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
        game=SuperMarioBrosNesV0Target.game,
        state=SuperMarioBrosNesV0Target.default_state,
        hud_crop_top=SuperMarioBrosNesV0Target.default_hud_crop_top,
        frame_skip=4,
        max_pool_frames=True,
        max_episode_steps=4500,
        reward_mode="score",
        terminal_reward=50.0,
        reward_scale=10.0,
        action_set="simple",
        info_events={"level_change": (("levelHi", "levelLo"), "change")},
        done_on_events=("level_change",),
        env_threads=args.env_threads,
    )


def action_sequence(name: str, length: int, n_actions: int, seed: int) -> list[int]:
    if name == "noop":
        return [0] * length
    if name == "right":
        return [1] * length
    if name == "right_b":
        return [2] * length
    if name == "right_ab":
        return [4] * length
    if name == "cycle":
        return [idx % n_actions for idx in range(length)]
    if name == "random":
        rng = np.random.default_rng(seed)
        return [int(x) for x in rng.integers(0, n_actions, size=length)]
    raise ValueError(f"unknown sequence {name!r}")


def run_single_trace(config: EnvConfig, sequence_name: str, length: int, seed: int) -> dict[str, Any]:
    action_names = action_names_for_set(config.action_set, game=config.game)
    actions = action_sequence(sequence_name, length, len(action_names), seed)
    env = make_fast_retro_env(config=config, seed=seed)
    obs, info = env.reset(seed=seed)
    obs_hashes = {"reset": sha_array(obs)}
    reward_sum = 0.0
    done_step = None
    final_info: dict[str, Any] = dict(info)
    samples: dict[str, Any] = {}
    action_counts: Counter[int] = Counter()
    for step_idx, action in enumerate(actions, start=1):
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        final_info = dict(info)
        if step_idx in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, length}:
            obs_hashes[str(step_idx)] = sha_array(obs)
            samples[str(step_idx)] = {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": normalize_info(dict(info)),
            }
        if terminated or truncated:
            done_step = step_idx
            break
    env.close()
    return {
        "sequence": sequence_name,
        "steps_run": step_idx,
        "done_step": done_step,
        "reward_sum": reward_sum,
        "action_counts": {action_names[key]: value for key, value in sorted(action_counts.items())},
        "obs_hashes": obs_hashes,
        "final_info": normalize_info(final_info),
        "samples": samples,
    }


def run_vector_trace(config: EnvConfig, sequence_name: str, length: int, seed: int, n_envs: int):
    action_names = action_names_for_set(config.action_set, game=config.game)
    base_actions = action_sequence(sequence_name, length * n_envs, len(action_names), seed)
    env = make_vec_envs(config=config, n_envs=n_envs, seed=seed)
    obs = env.reset()
    obs_hashes = {"reset": sha_array(obs)}
    reward_sums = np.zeros(n_envs, dtype=np.float64)
    done_counts = np.zeros(n_envs, dtype=np.int64)
    completion_counts = np.zeros(n_envs, dtype=np.int64)
    terminal_counts = np.zeros(n_envs, dtype=np.int64)
    first_done_step = [None] * n_envs
    final_infos: list[dict[str, Any]] = [{} for _ in range(n_envs)]
    samples: dict[str, Any] = {}

    for step_idx in range(1, length + 1):
        actions = np.asarray(
            [base_actions[(step_idx - 1) * n_envs + env_idx] for env_idx in range(n_envs)],
            dtype=np.int64,
        )
        obs, rewards, dones, infos = env.step(actions)
        reward_sums += np.asarray(rewards, dtype=np.float64)
        for env_idx, (done, info) in enumerate(zip(dones, infos, strict=False)):
            info = dict(info)
            final_infos[env_idx] = info
            if bool(info.get("completion_event", info.get("level_complete", False))):
                completion_counts[env_idx] += 1
            if bool(done):
                done_counts[env_idx] += 1
                if first_done_step[env_idx] is None:
                    first_done_step[env_idx] = step_idx
            if bool(info.get("_native_done", False)):
                terminal_counts[env_idx] += 1
        if step_idx in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, length}:
            obs_hashes[str(step_idx)] = sha_array(obs)
            samples[str(step_idx)] = {
                "reward_sum": float(np.sum(rewards)),
                "reward_min": float(np.min(rewards)),
                "reward_max": float(np.max(rewards)),
                "done_count": int(np.sum(dones)),
                "first_info": normalize_info(dict(infos[0])),
            }
    env.close()
    return {
        "sequence": sequence_name,
        "n_envs": n_envs,
        "steps_run": length,
        "reward_sums": [float(x) for x in reward_sums],
        "done_counts": [int(x) for x in done_counts],
        "first_done_step": first_done_step,
        "completion_counts": [int(x) for x in completion_counts],
        "native_terminal_counts": [int(x) for x in terminal_counts],
        "obs_hashes": obs_hashes,
        "final_infos": [normalize_info(info) for info in final_infos[: min(4, n_envs)]],
        "samples": samples,
    }


def run_raw_native_vector_trace(
    config: EnvConfig,
    sequence_name: str,
    length: int,
    seed: int,
    n_envs: int,
) -> dict[str, Any]:
    action_names = action_names_for_set(config.action_set, game=config.game)
    target = target_for_game(config.game)
    index_to_mask = [target.action_library[name] for name in action_names]
    action_indices = action_sequence(sequence_name, length * n_envs, len(action_names), seed)
    env = RetroVecEnv(
        config.game,
        num_envs=n_envs,
        state=config.state or None,
        num_threads=config.env_threads if config.env_threads > 0 else min(n_envs, 16),
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=config.obs_resize_algorithm,
        frame_skip=config.frame_skip,
        frame_stack=4,
        frame_maxpool=config.max_pool_frames,
        obs_copy="safe_view",
        obs_layout="chw",
    )
    env.seed(seed)
    obs = env.reset()
    reset_infos = getattr(env, "reset_infos", None)
    samples: dict[str, Any] = {
        "reset": {
            "obs_hash": sha_array(obs),
            "reset_info0": normalize_info(dict(reset_infos[0])) if reset_infos else {},
            "reset_info0_keys": sorted(reset_infos[0].keys()) if reset_infos else [],
        },
    }
    reward_sums = np.zeros(n_envs, dtype=np.float64)
    done_counts = np.zeros(n_envs, dtype=np.int64)
    final_infos: list[dict[str, Any]] = [{} for _ in range(n_envs)]
    for step_idx in range(1, length + 1):
        actions = np.stack(
            [index_to_mask[action_indices[(step_idx - 1) * n_envs + idx]] for idx in range(n_envs)]
        )
        obs, rewards, dones, infos = env.step(actions)
        reward_sums += np.asarray(rewards, dtype=np.float64)
        done_counts += np.asarray(dones, dtype=np.int64)
        final_infos = [dict(info) for info in infos]
        if step_idx in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, length}:
            samples[str(step_idx)] = {
                "obs_hash": sha_array(obs),
                "reward_sum": float(np.sum(rewards)),
                "rewards_first4": [float(x) for x in np.asarray(rewards)[:4]],
                "dones_first4": [bool(x) for x in np.asarray(dones)[:4]],
                "info0": normalize_info(dict(infos[0])),
                "info0_keys": sorted(infos[0].keys()),
            }
    env.close()
    return {
        "sequence": sequence_name,
        "n_envs": n_envs,
        "steps_run": length,
        "reward_sums": [float(x) for x in reward_sums],
        "done_counts": [int(x) for x in done_counts],
        "final_infos": [normalize_info(info) for info in final_infos[: min(4, n_envs)]],
        "final_info_keys": [sorted(info.keys()) for info in final_infos[: min(4, n_envs)]],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-steps", type=int, default=1200)
    parser.add_argument("--vector-steps", type=int, default=700)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--sequences",
        default="right,right_b,right_ab,random",
        help="Comma-separated action traces to run.",
    )
    args = parser.parse_args()
    config = config_from_args(args)
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    result = {
        "stable_retro_turbo": importlib.metadata.version("stable-retro-turbo"),
        "config": {
            "single_steps": args.single_steps,
            "vector_steps": args.vector_steps,
            "n_envs": args.n_envs,
            "env_threads": args.env_threads,
            "seed": args.seed,
            "sequences": sequences,
        },
        "single": [
            run_single_trace(config=config, sequence_name=sequence, length=args.single_steps, seed=args.seed)
            for sequence in sequences
        ],
        "raw_vector": [
            run_raw_native_vector_trace(
                config=config,
                sequence_name=sequence,
                length=args.vector_steps,
                seed=args.seed,
                n_envs=args.n_envs,
            )
            for sequence in sequences
        ],
        "vector": [
            run_vector_trace(
                config=config,
                sequence_name=sequence,
                length=args.vector_steps,
                seed=args.seed,
                n_envs=args.n_envs,
            )
            for sequence in sequences
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
