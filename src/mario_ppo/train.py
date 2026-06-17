from __future__ import annotations

# ruff: noqa: E402

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from mario_ppo.artifacts import (
    init_wandb,
    log_wandb_model_artifact,
    write_run_description,
    write_wandb_url,
)
from mario_ppo.callbacks import (
    RollingCompletionStopCallback,
    TrainingCompletionRateStopCallback,
    WandbCheckpointArtifactCallback,
)
from mario_ppo.cli import apply_preset, build_parser
from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import assert_rom_imported, default_run_dir, make_training_vec_env
from mario_ppo.env_config import env_config_from_args
from mario_ppo.eval_metrics import MarioEvalCallback
from mario_ppo.schedules import (
    EntropyCoefficientScheduleCallback,
    apply_resume_hyperparameters,
    learning_rate_schedule,
)


def main() -> None:
    args = apply_preset(build_parser().parse_args())
    assert_rom_imported()
    set_random_seed(args.seed)

    run_dir = default_run_dir(args.run_name, args.runs_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    write_run_description(args, run_dir)
    if args.run_description.strip():
        print(f"run description: {args.run_description.strip()}", flush=True)
    else:
        print("warning: --run-description is empty", flush=True)

    config = env_config_from_args(args, include_states=True, include_env_threads=True)
    wandb_run = init_wandb(args, run_dir, config)

    env = make_training_vec_env(config=config, n_envs=args.n_envs, seed=args.seed)
    device = resolve_sb3_device(args.device)
    if args.torch_num_threads > 0:
        import torch

        torch.set_num_threads(args.torch_num_threads)
        print(f"Using torch num threads: {torch.get_num_threads()}", flush=True)
    print(f"Using torch device: {device}", flush=True)

    lr_schedule = learning_rate_schedule(args)
    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=run_dir, device=device)
        apply_resume_hyperparameters(model, args)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            clip_range=args.clip_range,
            normalize_advantage=args.normalize_advantage,
            target_kl=args.target_kl,
            policy_kwargs={"optimizer_kwargs": {"eps": args.adam_eps}},
            tensorboard_log=run_dir,
            device=device,
            verbose=1,
        )

    checkpoint_save_freq = max(args.checkpoint_freq // max(args.n_envs, 1), 1)
    artifact_callback = WandbCheckpointArtifactCallback(
        wandb_run,
        args,
        checkpoint_dir,
        scan_freq=checkpoint_save_freq,
    )
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo_mario",
        ),
        artifact_callback,
    ]
    if args.ent_coef_final is not None:
        callbacks.append(
            EntropyCoefficientScheduleCallback(
                initial_value=args.ent_coef,
                final_value=args.ent_coef_final,
                schedule_timesteps=args.ent_coef_schedule_timesteps
                if args.ent_coef_schedule_timesteps > 0
                else args.timesteps,
            ),
        )
    if args.stop_completion_episode_window > 0 and args.stop_completion_rate_threshold > 0:
        callbacks.append(
            TrainingCompletionRateStopCallback(
                episode_window=args.stop_completion_episode_window,
                rate_threshold=args.stop_completion_rate_threshold,
                run_dir=run_dir,
                wandb_run=wandb_run,
            ),
        )
    if args.stop_completion_rolling_window > 0 and args.stop_completion_rolling_threshold > 0:
        callbacks.append(
            RollingCompletionStopCallback(
                rolling_window=args.stop_completion_rolling_window,
                threshold=args.stop_completion_rolling_threshold,
                run_dir=run_dir,
                wandb_run=wandb_run,
            ),
        )
    if args.eval_freq > 0 and args.eval_episodes > 0:
        callbacks.append(
            MarioEvalCallback(
                config=config,
                run_dir=run_dir,
                best_model_save_path=best_dir,
                eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                n_eval_episodes=args.eval_episodes,
                deterministic=not args.eval_stochastic,
                seed=args.seed + 10_000,
                completion_x_threshold=args.completion_x_threshold,
                wandb_run=wandb_run,
                record_video=not args.no_eval_videos,
                video_fps=args.eval_video_fps,
                video_scale=args.eval_video_scale,
            ),
        )
    else:
        print("training-loop eval disabled; evaluate checkpoint artifacts out of process")

    final_model_path = Path(run_dir, "final_model.zip")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        model.save(os.path.join(run_dir, "final_model"))
        artifact_callback.log_new_checkpoints()
        for best_model_path in sorted(Path(best_dir).glob("*.zip")):
            log_wandb_model_artifact(
                wandb_run,
                args,
                best_model_path,
                kind="best",
                aliases=["best", "latest"],
            )
        log_wandb_model_artifact(
            wandb_run,
            args,
            final_model_path,
            kind="final",
            aliases=["final", "latest"],
        )
        write_wandb_url(wandb_run, run_dir)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f"saved {final_model_path}")


if __name__ == "__main__":
    main()
