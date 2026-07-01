from __future__ import annotations

import hashlib
import importlib.metadata
import json
from typing import Any

import numpy as np
from stable_retro import RetroVecEnv

from rlab.env import EnvConfig, action_names_for_set, make_vec_envs
from rlab.targets import SuperMarioBrosNesV0Target, target_for_game


def sha_array(array: Any) -> str:
    arr = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def ptr(array: Any) -> int:
    arr = np.asarray(array)
    return int(arr.__array_interface__["data"][0])


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "hash": sha_array(arr),
        "ptr": ptr(arr),
        "strides": list(arr.strides),
        "c_contiguous": bool(arr.flags.c_contiguous),
        "writeable": bool(arr.flags.writeable),
    }


def mutation_check(reset_obs: np.ndarray, step_fn, actions: np.ndarray, steps: int = 3):
    prior_refs = [reset_obs]
    records = [
        {
            "event": "reset",
            "new_obs": array_summary(reset_obs),
            "prior_hashes": [sha_array(obs) for obs in prior_refs],
        }
    ]
    for step in range(1, steps + 1):
        before = [sha_array(obs) for obs in prior_refs]
        obs, rewards, dones, infos = step_fn(actions)
        after = [sha_array(old_obs) for old_obs in prior_refs]
        prior_refs.append(obs)
        records.append(
            {
                "event": f"step_{step}",
                "new_obs": array_summary(obs),
                "new_obs_shares_memory_with_prior": [
                    bool(np.shares_memory(obs, old_obs)) for old_obs in prior_refs[:-1]
                ],
                "prior_hashes_before": before,
                "prior_hashes_after": after,
                "prior_mutated_by_step": [before_i != after_i for before_i, after_i in zip(before, after, strict=True)],
                "reward_sum": float(np.sum(rewards)),
                "done_count": int(np.sum(dones)),
                "info0_keys": sorted(dict(infos[0]).keys()) if infos else [],
                "info0_sample": {
                    key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in dict(infos[0]).items()
                    if key in {"xscrollHi", "xscrollLo", "score", "lives", "levelHi", "levelLo"}
                }
                if infos
                else {},
            }
        )
    return records


def run_raw(obs_copy: str) -> dict[str, Any]:
    config = EnvConfig(
        game=SuperMarioBrosNesV0Target.game,
        state=SuperMarioBrosNesV0Target.default_state,
        hud_crop_top=SuperMarioBrosNesV0Target.default_hud_crop_top,
        reward_mode="score",
        terminal_reward=50.0,
        reward_scale=10.0,
        action_set="simple",
        info_events={"level_change": (("levelHi", "levelLo"), "change")},
        done_on_events=("level_change",),
        env_threads=4,
    )
    n_envs = 16
    env = RetroVecEnv(
        config.game,
        num_envs=n_envs,
        state=config.state or None,
        num_threads=4,
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=config.obs_resize_algorithm,
        frame_skip=config.frame_skip,
        frame_stack=4,
        frame_maxpool=config.max_pool_frames,
        obs_copy=obs_copy,
        obs_layout="chw",
        done_on={
            name: config.info_events[name]
            for name in config.done_on_events
            if name in config.info_events
        },
    )
    env.seed(23)
    obs = env.reset()
    action_names = action_names_for_set(config.action_set, game=config.game)
    target = target_for_game(config.game)
    action_masks = np.stack([target.action_library[name] for name in action_names]).astype(np.int8)
    actions = np.asarray([action_masks[index % len(action_masks)] for index in range(n_envs)])
    try:
        records = mutation_check(obs, env.step, actions)
    finally:
        env.close()
    return {
        "obs_copy": obs_copy,
        "records": records,
    }


def run_wrapped() -> dict[str, Any]:
    config = EnvConfig(
        game=SuperMarioBrosNesV0Target.game,
        state=SuperMarioBrosNesV0Target.default_state,
        hud_crop_top=SuperMarioBrosNesV0Target.default_hud_crop_top,
        reward_mode="score",
        terminal_reward=50.0,
        reward_scale=10.0,
        action_set="simple",
        info_events={"level_change": (("levelHi", "levelLo"), "change")},
        done_on_events=("level_change",),
        env_threads=4,
    )
    n_envs = 16
    env = make_vec_envs(config=config, n_envs=n_envs, seed=23)
    obs = env.reset()
    actions = np.arange(n_envs, dtype=np.int64) % len(action_names_for_set(config.action_set, game=config.game))

    def step_fn(discrete_actions):
        return env.step(discrete_actions)

    try:
        records = mutation_check(obs, step_fn, actions)
    finally:
        env.close()
    return {"records": records}


def main() -> None:
    result = {
        "stable_retro_turbo": importlib.metadata.version("stable-retro-turbo"),
        "raw_safe_view": run_raw(obs_copy="safe_view"),
        "raw_copy": run_raw(obs_copy="copy"),
        "wrapped_training_env": run_wrapped(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
