from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import stable_retro as retro


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _shared_objects() -> list[dict[str, Any]]:
    package_dir = Path(retro.__file__).resolve().parent
    objects = []
    for path in sorted(package_dir.rglob("*.so")) + sorted(package_dir.rglob("*.dylib")):
        objects.append(
            {
                "path": str(path.relative_to(package_dir)),
                "size_bytes": path.stat().st_size,
            },
        )
    return objects


def _make_action(env: Any) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=getattr(env.action_space, "dtype", np.int8))


def _bench_single(
    name: str,
    make_env: Callable[[], Any],
    steps: int,
    warmup: int,
) -> dict[str, Any]:
    env = make_env()
    try:
        obs, _info = env.reset()
        action = _make_action(env)
        for _ in range(warmup):
            env.step(action)
        start = time.perf_counter()
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset()
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "mode": "single_env",
            "steps": steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps / elapsed,
            "obs_shape": tuple(int(v) for v in obs.shape),
            "obs_dtype": str(obs.dtype),
        }
    finally:
        env.close()


def _bench_vector(
    name: str,
    make_env: Callable[[], Any],
    n_envs: int,
    vec_steps: int,
    warmup: int,
    start_method: str,
) -> dict[str, Any]:
    from stable_retro import StableRetroSubprocVecEnv

    env = StableRetroSubprocVecEnv([make_env for _ in range(n_envs)], start_method=start_method)
    try:
        obs = env.reset()
        probe_env = make_env()
        try:
            action_shape = probe_env.action_space.shape
            action_dtype = getattr(probe_env.action_space, "dtype", np.int8)
        finally:
            probe_env.close()
        action = np.zeros((n_envs,) + action_shape, dtype=action_dtype)
        for _ in range(warmup):
            env.step(action)
        start = time.perf_counter()
        for _ in range(vec_steps):
            obs, rewards, dones, infos = env.step(action)
        elapsed = time.perf_counter() - start
        total_steps = n_envs * vec_steps
        return {
            "name": name,
            "mode": "subproc_vec",
            "envs": n_envs,
            "vec_steps": vec_steps,
            "total_steps": total_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": total_steps / elapsed,
            "per_env_steps_per_sec": (total_steps / elapsed) / n_envs,
            "obs_shape": tuple(int(v) for v in obs.shape),
            "obs_dtype": str(obs.dtype),
        }
    finally:
        env.close()


def run_diagnostics(
    single_steps: int,
    vec_steps: int,
    warmup: int,
    vector_envs: list[int],
    start_method: str,
) -> dict[str, Any]:
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")

    def raw_rgb_env() -> Any:
        return retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array")

    def skip_only_env() -> Any:
        return retro.make("SuperMarioBros-Nes-v0", render_mode="rgb_array", frame_skip=4)

    def resize_gray_env() -> Any:
        return retro.make(
            "SuperMarioBros-Nes-v0",
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_crop=(32, 0, 0, 0),
            obs_resize_algorithm="area",
            obs_grayscale=True,
        )

    def atari_env() -> Any:
        return retro.make(
            "SuperMarioBros-Nes-v0",
            render_mode="rgb_array",
            obs_resize=(84, 84),
            obs_crop=(32, 0, 0, 0),
            obs_resize_algorithm="area",
            obs_grayscale=True,
            frame_skip=4,
            frame_stack=4,
            maxpool_last_two=True,
        )

    results = [
        _bench_single("raw_rgb", raw_rgb_env, single_steps, warmup),
        _bench_single("native_frame_skip_4", skip_only_env, single_steps, warmup),
        _bench_single("native_resize_gray", resize_gray_env, single_steps, warmup),
        _bench_single("native_atari_preproc", atari_env, single_steps, warmup),
    ]
    for n_envs in vector_envs:
        results.append(
            _bench_vector(
                f"native_atari_preproc_vec_{n_envs}",
                atari_env,
                n_envs,
                vec_steps,
                warmup,
                start_method,
            ),
        )

    return {
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "stable_retro_version": getattr(retro, "__version__", ""),
            "stable_retro_file": str(Path(retro.__file__).resolve()),
            "shared_objects": _shared_objects(),
        },
        "benchmarks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-steps", type=int, default=3_000)
    parser.add_argument("--vec-steps", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--vector-envs", default="1,16,32")
    parser.add_argument("--start-method", default="spawn")
    args = parser.parse_args()

    vector_envs = [int(value.strip()) for value in args.vector_envs.split(",") if value.strip()]
    result = run_diagnostics(
        single_steps=args.single_steps,
        vec_steps=args.vec_steps,
        warmup=args.warmup,
        vector_envs=vector_envs,
        start_method=args.start_method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
