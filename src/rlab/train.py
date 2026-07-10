from __future__ import annotations

# ruff: noqa: E402

import os
import re
import signal
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.utils import set_random_seed

from rlab.artifacts import (
    init_wandb,
    write_model_metadata,
    write_run_description,
    write_wandb_url,
)
from rlab.callbacks import (
    DoneCounterCallback,
    LedgerCheckpointCallback,
    LevelCompleteInfoCallback,
    MetricStoreMirrorCallback,
    MetricThresholdStopCallback,
    RewardComponentDiagnosticsCallback,
    RolloutDiagnosticsCallback,
    ThroughputCallback,
    TimeElapsedCallback,
)
from rlab.cli import effective_n_envs, parse_train_args
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_provider_runtime_available,
    default_run_dir,
    make_training_vec_env,
    resolve_env_config,
    resolve_mixed_state_config,
)
from rlab.env_config import env_config_from_args
from rlab.metric_store import MetricStore, file_sha256, metric_store_path
from rlab.schedules import (
    EntropyCoefficientScheduleCallback,
    apply_resume_hyperparameters,
    learning_rate_schedule,
)
from rlab.task_advantage import PerTaskAdvantagePPO, resolve_advantage_normalization_mode


SB3_HUMAN_OUTPUT_MAX_LENGTH = 512
GRACEFUL_STOP_SIGNAL = getattr(signal, "SIGUSR1", None)


def checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"ppo_{slug or 'retro'}"


def parse_net_arch(value: str) -> list[int]:
    if not str(value).strip():
        return []
    layers: list[int] = []
    for part in str(value).split(","):
        layer = part.strip()
        if not layer:
            continue
        size = int(layer)
        if size <= 0:
            raise ValueError("--policy-net-arch/--value-net-arch sizes must be positive")
        layers.append(size)
    return layers


def policy_kwargs_from_args(args) -> dict[str, object]:
    policy_kwargs: dict[str, object] = {"optimizer_kwargs": {"eps": args.adam_eps}}
    pi_arch = parse_net_arch(args.policy_net_arch)
    vf_arch = parse_net_arch(args.value_net_arch)
    if pi_arch or vf_arch:
        policy_kwargs["net_arch"] = {"pi": pi_arch, "vf": vf_arch}
    return policy_kwargs


def checkpoint_save_frequency(checkpoint_freq: int, n_envs: int) -> int | None:
    if checkpoint_freq <= 0:
        return None
    return max(checkpoint_freq // max(n_envs, 1), 1)


def save_model_bundle(
    *,
    model,
    args,
    config,
    model_path: Path,
    kind: str,
    step: int | None,
    store: MetricStore,
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.parent / f".{model_path.stem}.{uuid.uuid4().hex}.zip"
    model.save(str(temp_path))
    temp_path.replace(model_path)
    metadata_path = write_model_metadata(
        model_path,
        args,
        config,
        kind,
        checkpoint_step_value=step,
    )
    checkpoint_id = store.record_checkpoint(
        run_name=str(getattr(args, "run_name", "")),
        kind=kind,
        step=step,
        path=model_path,
        metadata_path=metadata_path,
        sha256=file_sha256(model_path),
    )
    print(f"{kind} model ready: id={checkpoint_id} step={step} path={model_path}", flush=True)
    return model_path


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal-{signum}"


class GracefulStopFlag:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def request(self, reason: str) -> None:
        self.requested = True
        self.reason = reason


def install_graceful_stop_handler(stop_flag: GracefulStopFlag) -> int | None:
    if GRACEFUL_STOP_SIGNAL is None:
        return None

    def handle_graceful_stop(signum, _frame) -> None:
        stop_flag.request(signal_name(signum))

    signal.signal(GRACEFUL_STOP_SIGNAL, handle_graceful_stop)
    return int(GRACEFUL_STOP_SIGNAL)


def disable_sb3_human_output_truncation(
    model, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH
) -> None:
    logger = getattr(model, "_logger", None)
    logger_attr = getattr(type(model), "logger", None)
    if logger is None and not isinstance(logger_attr, property):
        logger = getattr(model, "logger", None)
    if logger is None:
        return
    for output_format in getattr(logger, "output_formats", ()):
        if isinstance(output_format, HumanOutputFormat):
            output_format.max_length = max_length


class Sb3HumanOutputFormatCallback(BaseCallback):
    def __init__(self, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH) -> None:
        super().__init__()
        self.max_length = max_length

    def _on_training_start(self) -> None:
        disable_sb3_human_output_truncation(self.model, max_length=self.max_length)

    def _on_step(self) -> bool:
        return True


class GracefulStopCallback(BaseCallback):
    def __init__(self, stop_flag: GracefulStopFlag) -> None:
        super().__init__()
        self.stop_flag = stop_flag
        self.logged = False

    def _on_step(self) -> bool:
        if not self.stop_flag.requested:
            return True
        if not self.logged:
            reason = self.stop_flag.reason or "graceful stop"
            print(
                f"graceful stop requested by {reason}; "
                f"stopping at num_timesteps={self.num_timesteps}",
                flush=True,
            )
            self.logged = True
        return False


def main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    set_random_seed(args.seed)

    run_dir = default_run_dir(args.run_name, args.runs_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    store_path = metric_store_path(run_dir)
    metric_store = MetricStore(store_path)
    metric_store.init()
    write_run_description(args, run_dir)
    if args.run_description.strip():
        print(f"run description: {args.run_description.strip()}", flush=True)
    else:
        print("warning: --run-description is empty", flush=True)

    config = resolve_env_config(
        env_config_from_args(args, include_states=True)
    )
    n_envs = effective_n_envs(args)
    config = resolve_mixed_state_config(config, n_envs=n_envs)
    assert_provider_runtime_available(config)
    wandb_run = init_wandb(args, run_dir, config)
    write_wandb_url(wandb_run, run_dir)
    graceful_stop_flag = GracefulStopFlag()
    graceful_stop_signal = install_graceful_stop_handler(graceful_stop_flag)
    if graceful_stop_signal is not None:
        print(f"graceful stop signal: {signal_name(graceful_stop_signal)}", flush=True)

    env = make_training_vec_env(config=config, n_envs=n_envs, seed=args.seed)
    device = resolve_sb3_device(args.device)
    print(f"Using torch device: {device}", flush=True)

    lr_schedule = learning_rate_schedule(args)
    advantage_normalization = resolve_advantage_normalization_mode(args)
    if advantage_normalization == "per-task" and not config.task_conditioning:
        raise ValueError("--advantage-normalization per-task requires --task-conditioning")
    sb3_normalize_advantage = advantage_normalization == "global"
    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=run_dir, device=device)
        if advantage_normalization == "per-task":
            raise ValueError("--advantage-normalization per-task is not supported with --resume")
        apply_resume_hyperparameters(model, args)
        model.normalize_advantage = sb3_normalize_advantage
    else:
        policy_name = "MultiInputPolicy" if config.task_conditioning else "CnnPolicy"
        model_cls = PerTaskAdvantagePPO if advantage_normalization == "per-task" else PPO
        model = model_cls(
            policy_name,
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
            clip_range_vf=args.clip_range_vf,
            normalize_advantage=sb3_normalize_advantage,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs_from_args(args),
            tensorboard_log=run_dir,
            device=device,
            verbose=1,
        )
    callbacks = [
        GracefulStopCallback(graceful_stop_flag),
        Sb3HumanOutputFormatCallback(),
        TimeElapsedCallback(wandb_run=wandb_run),
        ThroughputCallback(),
        DoneCounterCallback(
            wandb_run=wandb_run,
            default_state=config.state,
            done_on_info={
                name: config.info_events[name]
                for name in config.done_on_events
                if name in config.info_events
            },
        ),
        LevelCompleteInfoCallback(
            wandb_run=wandb_run,
            info_events=config.info_events,
        ),
        MetricStoreMirrorCallback(store_path),
        *(
            [
                MetricThresholdStopCallback(
                    marker_path=Path(run_dir) / "early_stop.txt",
                    detector=args.early_stop,
                    metric_store_path=store_path,
                )
            ]
            if args.early_stop
            else []
        ),
        RolloutDiagnosticsCallback(wandb_run=wandb_run),
        RewardComponentDiagnosticsCallback(),
    ]
    checkpoint_save_freq = checkpoint_save_frequency(args.checkpoint_freq, n_envs)
    if checkpoint_save_freq is not None:
        callbacks.append(
            LedgerCheckpointCallback(
                args=args,
                config=config,
                save_freq=checkpoint_save_freq,
                save_path=checkpoint_dir,
                name_prefix=checkpoint_prefix(config.game),
                metric_store_path=store_path,
            )
        )
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
    print("training-loop eval disabled; async checkpoint eval handles promotion metrics")

    final_model_path = Path(run_dir, "final_model.zip")
    env_closed = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        if graceful_stop_flag.requested and checkpoint_save_freq is not None:
            interrupted_checkpoint_path = (
                Path(checkpoint_dir)
                / f"{checkpoint_prefix(config.game)}_interrupted_{model.num_timesteps}_steps.zip"
            )
            save_model_bundle(
                model=model,
                args=args,
                config=config,
                model_path=interrupted_checkpoint_path,
                kind="interrupted",
                step=model.num_timesteps,
                store=metric_store,
            )
            print(f"saved interrupted checkpoint {interrupted_checkpoint_path}", flush=True)
        save_model_bundle(
            model=model,
            args=args,
            config=config,
            model_path=final_model_path,
            kind="final",
            step=model.num_timesteps,
            store=metric_store,
        )
        env.close()
        env_closed = True
        write_wandb_url(wandb_run, run_dir)
    finally:
        if not env_closed:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f"saved {final_model_path}")


if __name__ == "__main__":
    main()
