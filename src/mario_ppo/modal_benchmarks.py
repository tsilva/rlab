from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from mario_ppo.modal_core import (
    PROJECT_ROOT,
    VOLUME_NAME,
    VOLUME_ROOT,
    app,
    ensure_remote_roms,
    image,
    volume,
)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=30 * 60,
    cpu=16.0,
    memory=32768,
)
def benchmark_env_remote(
    n_envs: int = 16,
    vec_steps: int = 2_000,
    warmup: int = 200,
    start_method: str = "spawn",
) -> dict[str, object]:
    import numpy as np
    import stable_retro as retro
    from stable_baselines3.common.vec_env import VecTransposeImage

    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    ensure_remote_roms("benchmarking")

    def make_mario_env():
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

    try:
        from stable_retro import StableRetroSubprocVecEnv
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"StableRetroSubprocVecEnv unavailable: {exc}",
            "package_version": getattr(retro, "__version__", "").strip(),
            "python": os.sys.version.split()[0],
        }

    env = StableRetroSubprocVecEnv(
        [make_mario_env for _ in range(n_envs)], start_method=start_method
    )
    hwc_obs = env.reset().copy()
    action = np.zeros((n_envs, 9), dtype=np.int8)
    for _ in range(warmup):
        env.step(action)
    start = time.perf_counter()
    for _ in range(vec_steps):
        env.step(action)
    elapsed = time.perf_counter() - start
    env.close()

    transposed = VecTransposeImage(
        StableRetroSubprocVecEnv([make_mario_env for _ in range(n_envs)], start_method=start_method)
    )
    chw_obs = transposed.reset().copy()
    transposed.close()

    result = {
        "ok": True,
        "package_version": getattr(retro, "__version__", "").strip(),
        "python": os.sys.version.split()[0],
        "envs": n_envs,
        "warmup_vec_steps": warmup,
        "vec_steps": vec_steps,
        "total_agent_steps": n_envs * vec_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": (n_envs * vec_steps) / elapsed,
        "hwc_obs_shape": tuple(int(v) for v in hwc_obs.shape),
        "hwc_obs_dtype": str(hwc_obs.dtype),
        "chw_obs_shape": tuple(int(v) for v in chw_obs.shape),
        "chw_obs_dtype": str(chw_obs.dtype),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=45 * 60,
    cpu=16.0,
    memory=32768,
)
def benchmark_env_sweep_remote(
    env_counts: list[int],
    vec_steps: int = 3_000,
    warmup: int = 100,
    start_method: str = "spawn",
) -> dict[str, object]:
    import numpy as np
    import stable_retro as retro

    os.environ["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    ensure_remote_roms("benchmarking")

    from stable_retro import StableRetroSubprocVecEnv

    def make_mario_env():
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

    results = []
    for n_envs in env_counts:
        env = StableRetroSubprocVecEnv(
            [make_mario_env for _ in range(n_envs)], start_method=start_method
        )
        try:
            hwc_obs = env.reset().copy()
            action = np.zeros((n_envs, 9), dtype=np.int8)
            for _ in range(warmup):
                env.step(action)
            start = time.perf_counter()
            for _ in range(vec_steps):
                env.step(action)
            elapsed = time.perf_counter() - start
        finally:
            env.close()

        result = {
            "envs": n_envs,
            "warmup_vec_steps": warmup,
            "vec_steps": vec_steps,
            "total_agent_steps": n_envs * vec_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": (n_envs * vec_steps) / elapsed,
            "per_env_steps_per_sec": ((n_envs * vec_steps) / elapsed) / n_envs,
            "hwc_obs_shape": tuple(int(v) for v in hwc_obs.shape),
            "hwc_obs_dtype": str(hwc_obs.dtype),
        }
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    base_sps = results[0]["steps_per_sec"] if results else 0.0
    for result in results:
        result["scaling_vs_first"] = result["steps_per_sec"] / base_sps if base_sps else 0.0
        result["parallel_efficiency_vs_first"] = (
            result["scaling_vs_first"] / result["envs"] if result["envs"] else 0.0
        )

    summary = {
        "ok": True,
        "package_version": getattr(retro, "__version__", "").strip(),
        "python": os.sys.version.split()[0],
        "start_method": start_method,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=45 * 60,
    cpu=16.0,
    memory=32768,
)
def benchmark_env_diagnostics_remote(
    single_steps: int = 3_000,
    vec_steps: int = 2_000,
    warmup: int = 100,
    vector_envs: str = "1,16,32",
    start_method: str = "spawn",
) -> dict[str, object]:
    ensure_remote_roms("benchmarking")
    cmd = [
        "python",
        "scripts/benchmarks/benchmark_retro_env_diagnostics.py",
        "--single-steps",
        str(single_steps),
        "--vec-steps",
        str(vec_steps),
        "--warmup",
        str(warmup),
        "--vector-envs",
        vector_envs,
        "--start-method",
        start_method,
    ]
    env = os.environ.copy()
    env["STABLE_RETRO_DISABLE_AUDIO"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, flush=True)
    return json.loads(completed.stdout)


@app.local_entrypoint()
def upload_roms(rom_dir: str = "~/Desktop/roms") -> None:
    local_rom_dir = Path(rom_dir).expanduser()
    if not local_rom_dir.is_dir():
        raise NotADirectoryError(local_rom_dir)

    rom_files = sorted(
        path
        for path in local_rom_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".nes", ".zip"}
    )
    if not rom_files:
        raise FileNotFoundError(f"No .nes or .zip ROMs found in {local_rom_dir}")

    with volume.batch_upload(force=True) as batch:
        for rom_file in rom_files:
            batch.put_file(rom_file, f"/roms/{rom_file.name}")
    print(f"Uploaded {len(rom_files)} ROM files to modal volume {VOLUME_NAME}:/roms")


@app.local_entrypoint()
def benchmark_env(
    n_envs: int = 16,
    vec_steps: int = 2_000,
    warmup: int = 200,
    cpu: float = 16.0,
    memory: int = 32768,
    start_method: str = "spawn",
) -> None:
    result = benchmark_env_remote.with_options(cpu=cpu, memory=memory).remote(
        n_envs=n_envs,
        vec_steps=vec_steps,
        warmup=warmup,
        start_method=start_method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def benchmark_env_sweep(
    env_counts: str = "1,2,4,8,16,32",
    vec_steps: int = 3_000,
    warmup: int = 100,
    cpu: float = 16.0,
    memory: int = 32768,
    start_method: str = "spawn",
) -> None:
    parsed_env_counts = [int(value.strip()) for value in env_counts.split(",") if value.strip()]
    result = benchmark_env_sweep_remote.with_options(cpu=cpu, memory=memory).remote(
        env_counts=parsed_env_counts,
        vec_steps=vec_steps,
        warmup=warmup,
        start_method=start_method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def benchmark_env_diagnostics(
    single_steps: int = 3_000,
    vec_steps: int = 2_000,
    warmup: int = 100,
    vector_envs: str = "1,16,32",
    cpu: float = 16.0,
    memory: int = 32768,
    start_method: str = "spawn",
) -> None:
    result = benchmark_env_diagnostics_remote.with_options(cpu=cpu, memory=memory).remote(
        single_steps=single_steps,
        vec_steps=vec_steps,
        warmup=warmup,
        vector_envs=vector_envs,
        start_method=start_method,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
