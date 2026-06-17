from __future__ import annotations

import os
import re
from pathlib import Path

from mario_ppo.cli import TRAIN_COMMAND_FIELDS, build_train_command
from mario_ppo.modal_core import (
    RUNS_DIR,
    VOLUME_ROOT,
    app,
    ensure_remote_roms,
    image,
    run_cmd,
    safe_path_name,
    volume,
    wandb_secret,
)
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT


def _download_wandb_model_artifact(ref: str) -> Path:
    import wandb

    download_root = RUNS_DIR / "wandb_artifacts" / safe_path_name(ref)
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
    n_envs: int = 8,
    env_threads: int = 0,
    torch_num_threads: int = 0,
    seed: int = 123,
    run_name: str = "modal_ppo_level1_1",
    run_description: str = "",
    game: str = "SuperMarioBros-Nes-v0",
    state: str = "Level1-1",
    states: str = "",
    batch_size: int = 256,
    n_steps: int = 512,
    n_epochs: int = 10,
    learning_rate: float = 1e-4,
    gamma: float = 0.9,
    gae_lambda: float = 1.0,
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
    vf_coef: float = 1.0,
    clip_range: float = 0.2,
    normalize_advantage: bool = False,
    adam_eps: float = 1e-8,
    target_kl: float = 0.0,
    reward_mode: str = "baseline",
    progress_reward_cap: float = 30.0,
    progress_reward_scale: float = 1.0,
    terminal_reward: float = 50.0,
    reward_scale: float = 10.0,
    time_penalty: float = 0.0,
    death_penalty: float = 25.0,
    completion_reward: float = 0.0,
    score_progress_clipped: bool = False,
    no_progress_timeout_steps: int = 0,
    no_progress_min_delta: int = 0,
    no_terminate_on_life_loss: bool = False,
    terminate_on_level_change: bool = False,
    terminate_on_completion: bool = False,
    action_set: str = "simple",
    resume: str | None = None,
    resume_artifact: str | None = None,
    auto_resume_latest: bool = False,
    device: str = "cuda",
    wandb: bool = True,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_mode: str = "online",
    wandb_artifact_storage_uri: str = "",
) -> dict[str, str | int | bool | None]:
    ensure_remote_roms("training")
    if resume and resume_artifact:
        raise ValueError("Use only one of resume or resume_artifact")

    resolved_resume = (
        str(_download_wandb_model_artifact(resume_artifact)) if resume_artifact else resume
    )
    if auto_resume_latest and resolved_resume is None:
        latest_checkpoint = _latest_checkpoint(run_name)
        if latest_checkpoint is not None:
            resolved_resume = str(latest_checkpoint)
            print(f"Auto-resuming from latest checkpoint {resolved_resume}", flush=True)

    local_values = locals().copy()
    train_options = {key: local_values[key] for key in TRAIN_COMMAND_FIELDS if key in local_values}
    train_options["runs_dir"] = str(RUNS_DIR)
    train_options["resume"] = resolved_resume
    cmd = build_train_command(train_options)

    env = os.environ.copy()
    if not wandb:
        env["WANDB_MODE"] = "disabled"
    run_cmd(cmd, env=env)
    volume.commit()

    run_dir = RUNS_DIR / run_name
    wandb_url_path = run_dir / "wandb_url.txt"
    wandb_run_id_path = run_dir / "wandb_run_id.txt"
    wandb_run_path_path = run_dir / "wandb_run_path.txt"
    run_description_path = run_dir / "run_description.txt"
    return {
        "run_name": run_name,
        "run_description": run_description_path.read_text().strip()
        if run_description_path.is_file()
        else None,
        "run_dir": str(run_dir),
        "final_model": str(run_dir / "final_model.zip"),
        "wandb_url": wandb_url_path.read_text().strip() if wandb_url_path.is_file() else None,
        "wandb_run_id": wandb_run_id_path.read_text().strip()
        if wandb_run_id_path.is_file()
        else None,
        "wandb_run_path": wandb_run_path_path.read_text().strip()
        if wandb_run_path_path.is_file()
        else None,
        "wandb_enabled": wandb,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "env_threads": env_threads,
        "torch_num_threads": torch_num_threads,
    }


@app.local_entrypoint()
def train(
    timesteps: int = 512,
    n_envs: int = 8,
    env_threads: int = 0,
    torch_num_threads: int = 0,
    seed: int = 123,
    run_name: str = "modal_smoke",
    run_description: str = "",
    game: str = "SuperMarioBros-Nes-v0",
    state: str = "Level1-1",
    states: str = "",
    cpu: float = 8.0,
    memory: int = 16384,
    gpu: str = "T4",
    n_steps: int = 512,
    n_epochs: int = 10,
    learning_rate: float = 1e-4,
    gamma: float = 0.9,
    gae_lambda: float = 1.0,
    batch_size: int = 256,
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
    vf_coef: float = 1.0,
    clip_range: float = 0.2,
    normalize_advantage: bool = False,
    adam_eps: float = 1e-8,
    target_kl: float = 0.0,
    reward_mode: str = "baseline",
    progress_reward_cap: float = 30.0,
    progress_reward_scale: float = 1.0,
    terminal_reward: float = 50.0,
    reward_scale: float = 10.0,
    time_penalty: float = 0.0,
    death_penalty: float = 25.0,
    completion_reward: float = 0.0,
    score_progress_clipped: bool = False,
    no_progress_timeout_steps: int = 0,
    no_progress_min_delta: int = 0,
    no_terminate_on_life_loss: bool = False,
    terminate_on_level_change: bool = False,
    terminate_on_completion: bool = False,
    action_set: str = "simple",
    resume: str = "",
    resume_artifact: str = "",
    auto_resume_latest: bool = False,
    wandb: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_mode: str = "offline",
    wandb_artifact_storage_uri: str = "",
) -> None:
    result = train_remote.with_options(cpu=cpu, memory=memory, gpu=gpu).remote(
        timesteps=timesteps,
        n_envs=n_envs,
        env_threads=env_threads,
        torch_num_threads=torch_num_threads,
        seed=seed,
        run_name=run_name,
        run_description=run_description,
        game=game,
        state=state,
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
        vf_coef=vf_coef,
        clip_range=clip_range,
        normalize_advantage=normalize_advantage,
        adam_eps=adam_eps,
        target_kl=target_kl,
        reward_mode=reward_mode,
        progress_reward_cap=progress_reward_cap,
        progress_reward_scale=progress_reward_scale,
        terminal_reward=terminal_reward,
        reward_scale=reward_scale,
        time_penalty=time_penalty,
        death_penalty=death_penalty,
        completion_reward=completion_reward,
        score_progress_clipped=score_progress_clipped,
        no_progress_timeout_steps=no_progress_timeout_steps,
        no_progress_min_delta=no_progress_min_delta,
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
        wandb_artifact_storage_uri=wandb_artifact_storage_uri,
    )
    print(result)
