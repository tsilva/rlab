from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "mario-ppo"
VOLUME_NAME = "mario-ppo-data"
PROJECT_ROOT = Path("/root/mario-ppo")
VOLUME_ROOT = Path("/vol")
ROM_DIR = VOLUME_ROOT / "roms"
RUNS_DIR = VOLUME_ROOT / "runs"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("ffmpeg", "git")
    .pip_install_from_pyproject(
        "pyproject.toml",
        optional_dependencies=[],
        extra_options="--only-binary=:all:",
    )
    .workdir(str(PROJECT_ROOT))
    .env(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "WANDB_DIR": str(RUNS_DIR),
            "WANDB_CACHE_DIR": str(RUNS_DIR / ".wandb-cache"),
            "WANDB_CONFIG_DIR": str(RUNS_DIR / ".wandb-config"),
            "WANDB_DATA_DIR": str(RUNS_DIR / ".wandb-data"),
            "WANDB_ARTIFACT_DIR": str(RUNS_DIR / ".wandb-artifacts"),
        },
    )
    .add_local_dir(
        ".",
        remote_path=str(PROJECT_ROOT),
        ignore=[
            ".git",
            ".venv",
            ".uv-cache",
            ".matplotlib",
            "__pycache__",
            "runs",
            "logs",
            "models",
            "videos",
            "wandb",
        ],
    )
)


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"


def _download_wandb_model_artifact(ref: str) -> Path:
    import wandb

    download_root = RUNS_DIR / "wandb_artifacts" / _safe_path_name(ref)
    download_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading W&B artifact {ref} to {download_root}", flush=True)
    artifact = wandb.Api().artifact(ref, type="model")
    artifact_path = Path(artifact.download(root=str(download_root)))
    zip_files = sorted(artifact_path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in {artifact_path}")
    print(f"Using resumed model {zip_files[0]}", flush=True)
    return zip_files[0]


def _latest_checkpoint(run_name: str) -> Path | None:
    checkpoint_dir = RUNS_DIR / run_name / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"_(\d+)_steps(?:\.zip)?$", path.name)
        return int(match.group(1)) if match else -1

    checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=checkpoint_step)
    return checkpoints[-1] if checkpoints else None


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): volume},
    timeout=24 * 60 * 60,
    gpu="T4",
    cpu=8.0,
    memory=16384,
    secrets=[wandb_secret],
)
def train_remote(
    timesteps: int = 1_000_000,
    n_envs: int = 4,
    seed: int = 7,
    run_name: str = "modal_ppo_level1_1",
    state: str = "Level1-1",
    states: str = "",
    batch_size: int = 256,
    n_steps: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eval_freq: int = 0,
    eval_episodes: int = 0,
    eval_stochastic: bool = True,
    completion_x_threshold: int = 3160,
    no_eval_videos: bool = False,
    eval_video_fps: float = 30.0,
    eval_video_scale: int = 4,
    frame_skip: int = 4,
    max_pool_frames: bool = True,
    max_episode_steps: int = 4500,
    hud_crop_top: int = 32,
    checkpoint_freq: int = 100_000,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    target_kl: float = 0.0,
    reward_mode: str = "bounded",
    progress_reward_cap: float = 30.0,
    progress_reward_scale: float = 1.0,
    terminal_reward: float = 30.0,
    reward_scale: float = 30.0,
    time_penalty: float = 0.0,
    death_penalty: float = 25.0,
    completion_reward: float = 0.0,
    no_terminate_on_life_loss: bool = False,
    terminate_on_level_change: bool = False,
    terminate_on_completion: bool = False,
    action_set: str = "simple",
    resume: str | None = None,
    resume_artifact: str | None = None,
    auto_resume_latest: bool = False,
    device: str = "cuda",
    wandb: bool = True,
    wandb_project: str = "mario-ppo",
    wandb_mode: str = "online",
) -> dict[str, str | int | bool | None]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROM_DIR.exists() or not any(ROM_DIR.iterdir()):
        raise FileNotFoundError(
            f"No ROMs found in {ROM_DIR}. Run upload_roms before training.",
        )
    if resume and resume_artifact:
        raise ValueError("Use only one of resume or resume_artifact")

    _run(["python", "-m", "stable_retro.import", str(ROM_DIR)])
    resolved_resume = str(_download_wandb_model_artifact(resume_artifact)) if resume_artifact else resume
    if auto_resume_latest and resolved_resume is None:
        latest_checkpoint = _latest_checkpoint(run_name)
        if latest_checkpoint is not None:
            resolved_resume = str(latest_checkpoint)
            print(f"Auto-resuming from latest checkpoint {resolved_resume}", flush=True)

    cmd = [
        "python",
        "-m",
        "mario_ppo.train",
        "--timesteps",
        str(timesteps),
        "--n-envs",
        str(n_envs),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--runs-dir",
        str(RUNS_DIR),
        "--state",
        state,
        "--states",
        states,
        "--batch-size",
        str(batch_size),
        "--n-steps",
        str(n_steps),
        "--n-epochs",
        str(n_epochs),
        "--learning-rate",
        str(learning_rate),
        "--gamma",
        str(gamma),
        "--gae-lambda",
        str(gae_lambda),
        "--device",
        device,
        "--eval-freq",
        str(eval_freq),
        "--eval-episodes",
        str(eval_episodes),
        "--completion-x-threshold",
        str(completion_x_threshold),
        "--eval-video-fps",
        str(eval_video_fps),
        "--eval-video-scale",
        str(eval_video_scale),
        "--frame-skip",
        str(frame_skip),
        "--max-episode-steps",
        str(max_episode_steps),
        "--hud-crop-top",
        str(hud_crop_top),
        "--checkpoint-freq",
        str(checkpoint_freq),
        "--ent-coef",
        str(ent_coef),
        "--clip-range",
        str(clip_range),
        "--reward-mode",
        reward_mode,
        "--progress-reward-cap",
        str(progress_reward_cap),
        "--progress-reward-scale",
        str(progress_reward_scale),
        "--terminal-reward",
        str(terminal_reward),
        "--reward-scale",
        str(reward_scale),
        "--time-penalty",
        str(time_penalty),
        "--death-penalty",
        str(death_penalty),
        "--completion-reward",
        str(completion_reward),
        "--action-set",
        action_set,
    ]
    if max_pool_frames:
        cmd.append("--max-pool-frames")
    else:
        cmd.append("--no-max-pool-frames")
    if target_kl > 0:
        cmd.extend(["--target-kl", str(target_kl)])
    if eval_stochastic:
        cmd.append("--eval-stochastic")
    if no_eval_videos:
        cmd.append("--no-eval-videos")
    if no_terminate_on_life_loss:
        cmd.append("--no-terminate-on-life-loss")
    if terminate_on_level_change:
        cmd.append("--terminate-on-level-change")
    if terminate_on_completion:
        cmd.append("--terminate-on-completion")
    if resolved_resume:
        cmd.extend(["--resume", resolved_resume])
    if wandb:
        cmd.extend(["--wandb", "--wandb-project", wandb_project, "--wandb-mode", wandb_mode])

    env = os.environ.copy()
    if not wandb:
        env["WANDB_MODE"] = "disabled"
    _run(cmd, env=env)
    volume.commit()

    run_dir = RUNS_DIR / run_name
    wandb_url_path = run_dir / "wandb_url.txt"
    wandb_run_id_path = run_dir / "wandb_run_id.txt"
    wandb_run_path_path = run_dir / "wandb_run_path.txt"
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_model": str(run_dir / "final_model.zip"),
        "wandb_url": wandb_url_path.read_text().strip() if wandb_url_path.is_file() else None,
        "wandb_run_id": wandb_run_id_path.read_text().strip() if wandb_run_id_path.is_file() else None,
        "wandb_run_path": (
            wandb_run_path_path.read_text().strip() if wandb_run_path_path.is_file() else None
        ),
        "wandb_enabled": wandb,
        "timesteps": timesteps,
        "n_envs": n_envs,
    }


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
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROM_DIR.exists() or not any(ROM_DIR.iterdir()):
        raise FileNotFoundError(
            f"No ROMs found in {ROM_DIR}. Run upload_roms before benchmarking.",
        )
    _run(["python", "-m", "stable_retro.import", str(ROM_DIR)])

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

    env = StableRetroSubprocVecEnv([make_mario_env for _ in range(n_envs)], start_method=start_method)
    hwc_obs = env.reset().copy()
    action = np.zeros((n_envs, 9), dtype=np.int8)
    for _ in range(warmup):
        env.step(action)
    start = time.perf_counter()
    for _ in range(vec_steps):
        obs, rewards, dones, infos = env.step(action)
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
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROM_DIR.exists() or not any(ROM_DIR.iterdir()):
        raise FileNotFoundError(
            f"No ROMs found in {ROM_DIR}. Run upload_roms before benchmarking.",
        )
    _run(["python", "-m", "stable_retro.import", str(ROM_DIR)])

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
        env = StableRetroSubprocVecEnv([make_mario_env for _ in range(n_envs)], start_method=start_method)
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
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROM_DIR.exists() or not any(ROM_DIR.iterdir()):
        raise FileNotFoundError(
            f"No ROMs found in {ROM_DIR}. Run upload_roms before benchmarking.",
        )
    _run(["python", "-m", "stable_retro.import", str(ROM_DIR)])
    cmd = [
        "python",
        "scripts/benchmark_retro_env_diagnostics.py",
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
def train(
    timesteps: int = 512,
    n_envs: int = 1,
    seed: int = 7,
    run_name: str = "modal_smoke",
    states: str = "",
    cpu: float = 8.0,
    memory: int = 16384,
    gpu: str = "T4",
    n_steps: int = 128,
    n_epochs: int = 4,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    batch_size: int = 0,
    eval_freq: int = 0,
    eval_episodes: int = 0,
    eval_stochastic: bool = True,
    completion_x_threshold: int = 3160,
    no_eval_videos: bool = False,
    eval_video_fps: float = 30.0,
    eval_video_scale: int = 4,
    frame_skip: int = 4,
    max_pool_frames: bool = True,
    max_episode_steps: int = 600,
    hud_crop_top: int = 32,
    checkpoint_freq: int = 100_000,
    ent_coef: float = 0.01,
    clip_range: float = 0.2,
    target_kl: float = 0.0,
    reward_mode: str = "bounded",
    progress_reward_cap: float = 30.0,
    progress_reward_scale: float = 1.0,
    terminal_reward: float = 30.0,
    reward_scale: float = 30.0,
    time_penalty: float = 0.0,
    death_penalty: float = 25.0,
    completion_reward: float = 0.0,
    no_terminate_on_life_loss: bool = False,
    terminate_on_level_change: bool = False,
    terminate_on_completion: bool = False,
    action_set: str = "simple",
    resume: str = "",
    resume_artifact: str = "",
    auto_resume_latest: bool = False,
    wandb: bool = False,
    wandb_project: str = "mario-ppo",
    wandb_mode: str = "offline",
) -> None:
    result = train_remote.with_options(cpu=cpu, memory=memory, gpu=gpu).remote(
        timesteps=timesteps,
        n_envs=n_envs,
        seed=seed,
        run_name=run_name,
        states=states,
        batch_size=batch_size or (128 if n_envs == 1 else 256),
        n_steps=n_steps,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_stochastic=eval_stochastic,
        completion_x_threshold=completion_x_threshold,
        no_eval_videos=no_eval_videos,
        eval_video_fps=eval_video_fps,
        eval_video_scale=eval_video_scale,
        frame_skip=frame_skip,
        max_pool_frames=max_pool_frames,
        max_episode_steps=max_episode_steps,
        hud_crop_top=hud_crop_top,
        checkpoint_freq=checkpoint_freq,
        ent_coef=ent_coef,
        clip_range=clip_range,
        target_kl=target_kl,
        reward_mode=reward_mode,
        progress_reward_cap=progress_reward_cap,
        progress_reward_scale=progress_reward_scale,
        terminal_reward=terminal_reward,
        reward_scale=reward_scale,
        time_penalty=time_penalty,
        death_penalty=death_penalty,
        completion_reward=completion_reward,
        no_terminate_on_life_loss=no_terminate_on_life_loss,
        terminate_on_level_change=terminate_on_level_change,
        terminate_on_completion=terminate_on_completion,
        action_set=action_set,
        resume=resume or None,
        resume_artifact=resume_artifact or None,
        auto_resume_latest=auto_resume_latest,
        device="cuda",
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )
    print(result)


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
