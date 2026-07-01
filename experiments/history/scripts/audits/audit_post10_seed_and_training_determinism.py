from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_retro import RetroVecEnv

from rlab.device import resolve_sb3_device
from rlab.env import EnvConfig, action_names_for_set, assert_rom_imported, make_vec_envs
from rlab.targets import SuperMarioBrosNesV0Target, target_for_game


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        "score",
        "lives",
        "levelHi",
        "levelLo",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def make_raw_env(config: EnvConfig, n_envs: int, env_threads: int, env_kwargs: dict[str, Any]):
    return RetroVecEnv(
        config.game,
        num_envs=n_envs,
        state=config.state or None,
        num_threads=env_threads,
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
        **env_kwargs,
    )


def probe_noop_keyword(config: EnvConfig, n_envs: int, env_threads: int) -> dict[str, Any]:
    candidates = ("reset_noops",)
    results: dict[str, Any] = {}
    for name in candidates:
        try:
            env = make_raw_env(config, n_envs=n_envs, env_threads=env_threads, env_kwargs={name: 30})
            env.seed(23)
            obs = env.reset()
            results[name] = {
                "accepted": True,
                "reset_hash": sha_array(obs),
            }
            env.close()
        except Exception as exc:  # noqa: BLE001 - this is an API probe.
            results[name] = {"accepted": False, "error": repr(exc)}
    return results


def action_indices(seed: int, steps: int, n_envs: int, n_actions: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_actions, size=(steps, n_envs), dtype=np.int64)


def env_trace(
    *,
    config: EnvConfig,
    seed: int,
    action_seed: int,
    steps: int,
    n_envs: int,
    env_threads: int,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    action_names = action_names_for_set(config.action_set, game=config.game)
    target = target_for_game(config.game)
    action_masks = np.stack([target.action_library[name] for name in action_names]).astype(np.int8)
    action_table = action_indices(action_seed, steps, n_envs, len(action_names))
    sample_steps = {1, 2, 4, 8, 16, 32, 64, 128, steps}

    env = make_raw_env(config, n_envs=n_envs, env_threads=env_threads, env_kwargs=env_kwargs)
    try:
        env.seed(seed)
        obs = env.reset()
        reset_infos = getattr(env, "reset_infos", None)
        samples: dict[str, Any] = {
            "reset": {
                "obs_hash": sha_array(obs),
                "info0": normalize_info(dict(reset_infos[0])) if reset_infos else {},
            }
        }
        reward_sums = np.zeros(n_envs, dtype=np.float64)
        done_counts = np.zeros(n_envs, dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(np.asarray(obs).tobytes())
        for step in range(1, steps + 1):
            actions = action_masks[action_table[step - 1]]
            obs, rewards, dones, infos = env.step(actions)
            rewards_arr = np.asarray(rewards)
            dones_arr = np.asarray(dones)
            reward_sums += rewards_arr.astype(np.float64)
            done_counts += dones_arr.astype(np.int64)
            digest.update(np.asarray(obs).tobytes())
            digest.update(rewards_arr.tobytes())
            digest.update(dones_arr.tobytes())
            digest.update(json.dumps([normalize_info(dict(info)) for info in infos[:4]], sort_keys=True).encode())
            if step in sample_steps:
                samples[str(step)] = {
                    "obs_hash": sha_array(obs),
                    "reward_sum": float(np.sum(rewards_arr)),
                    "done_count": int(np.sum(dones_arr)),
                    "info0": normalize_info(dict(infos[0])),
                }
        return {
            "seed": seed,
            "action_seed": action_seed,
            "steps": steps,
            "n_envs": n_envs,
            "env_threads": env_threads,
            "env_kwargs": env_kwargs,
            "fingerprint": digest.hexdigest(),
            "reward_sums": [float(x) for x in reward_sums],
            "done_counts": [int(x) for x in done_counts],
            "samples": samples,
        }
    finally:
        env.close()


def compare_env_traces(args: argparse.Namespace, config: EnvConfig) -> dict[str, Any]:
    noop_probe = probe_noop_keyword(config, args.n_envs, args.env_threads)
    noop_key = next((key for key, value in noop_probe.items() if value.get("accepted")), None)
    env_kwargs_variants: list[tuple[str, dict[str, Any]]] = [("default", {})]
    if noop_key is not None:
        env_kwargs_variants.append((f"{noop_key}=30", {noop_key: 30}))

    cases: list[dict[str, Any]] = []
    for label, env_kwargs in env_kwargs_variants:
        for seed in (args.seed, args.seed + 1):
            traces = [
                env_trace(
                    config=config,
                    seed=seed,
                    action_seed=args.action_seed,
                    steps=args.env_steps,
                    n_envs=args.n_envs,
                    env_threads=args.env_threads,
                    env_kwargs=env_kwargs,
                )
                for _ in range(args.env_repeats)
            ]
            cases.append(
                {
                    "label": label,
                    "seed": seed,
                    "fingerprints": [trace["fingerprint"] for trace in traces],
                    "deterministic": len({trace["fingerprint"] for trace in traces}) == 1,
                    "first_trace": traces[0],
                }
            )
    baseline = cases[0]["fingerprints"][0]
    default_next_seed = cases[1]["fingerprints"][0] if len(cases) > 1 else None
    noop_same_seed = cases[2]["fingerprints"][0] if noop_key is not None and len(cases) > 2 else None
    return {
        "noop_probe": noop_probe,
        "selected_noop_key": noop_key,
        "cases": cases,
        "same_seed_repeats_all_deterministic": all(case["deterministic"] for case in cases),
        "different_seed_changes_trace": default_next_seed is not None and baseline != default_next_seed,
        "noop_changes_trace_for_same_seed": noop_same_seed is not None and baseline != noop_same_seed,
    }


def set_training_determinism(seed: int, deterministic_torch: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def model_tensor_hash(model: PPO) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def train_once(
    *,
    args: argparse.Namespace,
    config: EnvConfig,
    run_root: Path,
    repeat_idx: int,
    deterministic_torch: bool,
) -> dict[str, Any]:
    set_training_determinism(args.train_seed, deterministic_torch)
    set_random_seed(args.train_seed)
    run_dir = run_root / f"repeat_{repeat_idx}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    env = make_vec_envs(config=config, n_envs=args.train_n_envs, seed=args.train_seed)
    try:
        device = resolve_sb3_device(args.device)
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.train_learning_rate,
            n_steps=args.train_n_steps,
            batch_size=args.train_batch_size,
            n_epochs=args.train_n_epochs,
            gamma=0.9,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=1.0,
            clip_range=0.2,
            normalize_advantage=False,
            policy_kwargs={"optimizer_kwargs": {"eps": 1e-8}},
            tensorboard_log=str(run_dir),
            device=device,
            seed=args.train_seed,
            verbose=0,
        )
        initial_hash = model_tensor_hash(model)
        model.learn(total_timesteps=args.train_timesteps, progress_bar=False)
        final_hash = model_tensor_hash(model)
        model.save(run_dir / "final_model")
        return {
            "repeat_idx": repeat_idx,
            "initial_policy_hash": initial_hash,
            "final_policy_hash": final_hash,
            "num_timesteps": int(model.num_timesteps),
            "run_dir": str(run_dir),
            "device": str(device),
        }
    finally:
        env.close()


def compare_training(args: argparse.Namespace, config: EnvConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for deterministic_torch in (False, True):
        label = "torch_deterministic" if deterministic_torch else "default"
        run_root = Path(args.output_dir) / "training" / label
        repeats = [
            train_once(
                args=args,
                config=config,
                run_root=run_root,
                repeat_idx=idx,
                deterministic_torch=deterministic_torch,
            )
            for idx in range(args.train_repeats)
        ]
        results[label] = {
            "repeats": repeats,
            "initial_hashes": [item["initial_policy_hash"] for item in repeats],
            "final_hashes": [item["final_policy_hash"] for item in repeats],
            "initial_deterministic": len({item["initial_policy_hash"] for item in repeats}) == 1,
            "final_deterministic": len({item["final_policy_hash"] for item in repeats}) == 1,
        }
        if results[label]["final_deterministic"]:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="logs/post10_determinism_audit")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--action-seed", type=int, default=12345)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--env-threads", type=int, default=4)
    parser.add_argument("--env-steps", type=int, default=512)
    parser.add_argument("--env-repeats", type=int, default=3)
    parser.add_argument("--train-seed", type=int, default=23)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--train-timesteps", type=int, default=32768)
    parser.add_argument("--train-n-envs", type=int, default=16)
    parser.add_argument("--train-n-steps", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--train-n-epochs", type=int, default=2)
    parser.add_argument("--train-learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    assert_rom_imported(SuperMarioBrosNesV0Target.game)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(
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
    result = {
        "stable_retro_turbo": importlib.metadata.version("stable-retro-turbo"),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "env_config": {
            "n_envs": args.n_envs,
            "env_threads": args.env_threads,
            "env_steps": args.env_steps,
            "env_repeats": args.env_repeats,
            "preprocessing": {
                "obs_resize": [config.observation_size, config.observation_size],
                "obs_crop": [config.hud_crop_top, 0, 0, 0],
                "obs_grayscale": True,
                "obs_resize_algorithm": config.obs_resize_algorithm,
                "frame_skip": config.frame_skip,
                "frame_stack": 4,
                "frame_maxpool": config.max_pool_frames,
                "obs_copy": "safe_view",
            },
        },
        "env_determinism": compare_env_traces(args, config),
        "training_config": {
            "train_repeats": args.train_repeats,
            "train_timesteps": args.train_timesteps,
            "train_n_envs": args.train_n_envs,
            "train_n_steps": args.train_n_steps,
            "train_batch_size": args.train_batch_size,
            "train_n_epochs": args.train_n_epochs,
            "device": args.device,
        },
        "training_determinism": compare_training(args, config),
    }
    out_path = output_dir / "post10_seed_and_training_determinism.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
