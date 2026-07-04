from __future__ import annotations

# ruff: noqa: E402

import os
import re
import signal
import time
import json
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.utils import set_random_seed

from rlab.artifacts import (
    checkpoint_step,
    init_wandb,
    log_wandb_model_artifact,
    sanitize_artifact_name,
    write_run_description,
    write_wandb_url,
)
from rlab.callbacks import (
    CheckpointArtifactTimingState,
    DoneCounterCallback,
    LevelCompleteInfoCallback,
    MetricThresholdStopCallback,
    RewardComponentDiagnosticsCallback,
    RolloutDiagnosticsCallback,
    ThroughputCallback,
    TimeElapsedCallback,
    TimedCheckpointCallback,
    WandbCheckpointArtifactCallback,
)
from rlab.cli import parse_train_args
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_rom_imported,
    default_run_dir,
    make_training_vec_env,
    resolve_env_config,
    resolve_mixed_state_config,
)
from rlab.env_config import env_config_from_args
from rlab.eval_metrics import flat_numeric_metrics
from rlab.eval_runner import evaluate_model_episodes
from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_BEST_X,
    EVAL_CHECKPOINT_ARTIFACT,
    EVAL_CHECKPOINT_STEP,
    EVAL_CONFIG_HUD_CROP_TOP,
    EVAL_DEATH_COUNT,
    EVAL_DEATH_RATE,
    EVAL_PROGRESS_LEVEL_X_MAX,
    EVAL_PROGRESS_LEVEL_X_MEAN,
    EVAL_PROGRESS_X_MAX,
    EVAL_PROGRESS_X_MEAN,
    EVAL_REWARD_MAX,
    EVAL_REWARD_MEAN,
    EVAL_REWARD_STD,
    GLOBAL_STEP,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_COMPLETION_RATE,
    LEADER_CHECKPOINT_COMPLETION_RATE_MEAN,
    LEADER_CHECKPOINT_EVAL_SOURCE,
    LEADER_CHECKPOINT_LOCAL_PATH,
    LEADER_CHECKPOINT_MAX_X_MAX,
    LEADER_CHECKPOINT_REWARD_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL,
    LEADER_CHECKPOINT_UPDATED_AT,
)
from rlab.schedules import (
    EntropyCoefficientScheduleCallback,
    apply_resume_hyperparameters,
    learning_rate_schedule,
)
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.wandb_utils import DEFAULT_WANDB_ENTITY, resolve_wandb_project
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


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    step = checkpoint_step(path)
    return (step if step is not None else 2**63 - 1, path.name)


def checkpoint_paths_for_eval(checkpoint_dir: str | Path) -> list[Path]:
    return sorted(Path(checkpoint_dir).glob("*.zip"), key=checkpoint_sort_key)


def eval_checkpoint_artifact_ref(args, checkpoint_path: Path, step: int) -> str:
    if getattr(args, "no_wandb_artifacts", False):
        return str(checkpoint_path)
    entity = str(getattr(args, "wandb_entity", "") or DEFAULT_WANDB_ENTITY).strip()
    project = resolve_wandb_project(
        getattr(args, "wandb_project", None),
        str(getattr(args, "game", "") or ""),
    )
    if entity and project:
        name = f"{sanitize_artifact_name(args.run_name)}-checkpoint"
        return f"{entity}/{project}/{name}:step-{step}"
    return str(checkpoint_path)


COMPLETION_GOAL_RATE = 0.99


def eval_score(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    def metric_float(key: str, default: float = float("-inf")) -> float:
        value = metrics.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    completion_min = metric_float(
        "eval/done/level_change/from_rate/min",
        metric_float("eval/done/level_change/rate", metric_float("completion_rate")),
    )
    completion_mean = metric_float(
        "eval/done/level_change/from_rate/mean",
        metric_float("eval/done/level_change/rate", metric_float("completion_rate")),
    )
    checkpoint_step_value = metric_float("checkpoint_step")
    steps_to_goal = (
        checkpoint_step_value
        if completion_min >= COMPLETION_GOAL_RATE and checkpoint_step_value > float("-inf")
        else float("inf")
    )
    return (completion_min, completion_mean, -steps_to_goal, metric_float("reward_mean"))


def update_best_checkpoint_summary(
    wandb_run,
    *,
    metrics: dict[str, object],
    checkpoint_path: Path,
    checkpoint_step_value: int,
    artifact_ref: str,
) -> None:
    if wandb_run is None:
        return

    def remove_summary_key(key: str) -> None:
        try:
            del wandb_run.summary[key]
        except (AttributeError, KeyError):
            pass

    def summary_float(key: str) -> float:
        try:
            value = wandb_run.summary.get(key)
        except AttributeError:
            value = None
        if value is None:
            return float("-inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    score = eval_score(metrics)
    previous_completion = summary_float(LEADER_CHECKPOINT_COMPLETION_RATE)
    previous_completion_mean = summary_float(LEADER_CHECKPOINT_COMPLETION_RATE_MEAN)
    previous_steps_to_goal = summary_float(LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL)
    if previous_steps_to_goal == float("-inf") and previous_completion >= COMPLETION_GOAL_RATE:
        previous_steps_to_goal = summary_float(LEADER_CHECKPOINT_STEP)
    previous_step_score = (
        -previous_steps_to_goal
        if previous_steps_to_goal > float("-inf")
        else float("-inf")
    )
    previous = (
        previous_completion,
        previous_completion_mean,
        previous_step_score,
        summary_float(LEADER_CHECKPOINT_REWARD_MEAN),
    )
    if score < previous:
        return

    wandb_run.summary[LEADER_CHECKPOINT_COMPLETION_RATE] = score[0]
    wandb_run.summary[LEADER_CHECKPOINT_COMPLETION_RATE_MEAN] = score[1]
    wandb_run.summary[LEADER_CHECKPOINT_REWARD_MEAN] = score[3]
    wandb_run.summary[LEADER_CHECKPOINT_MAX_X_MAX] = metrics.get("max_x_max")
    wandb_run.summary[LEADER_CHECKPOINT_STEP] = checkpoint_step_value
    if score[2] > float("-inf"):
        wandb_run.summary[LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL] = checkpoint_step_value
    else:
        remove_summary_key(LEADER_CHECKPOINT_STEPS_TO_COMPLETION_GOAL)
    wandb_run.summary[LEADER_CHECKPOINT_ARTIFACT_REF] = artifact_ref
    wandb_run.summary[LEADER_CHECKPOINT_LOCAL_PATH] = str(checkpoint_path)
    wandb_run.summary[LEADER_CHECKPOINT_EVAL_SOURCE] = "post_train_inline"
    wandb_run.summary[LEADER_CHECKPOINT_UPDATED_AT] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )


def log_checkpoint_eval_metrics(
    wandb_run,
    *,
    args,
    metrics: dict[str, object],
    checkpoint_path: Path,
    checkpoint_step_value: int,
    artifact_ref: str,
) -> None:
    if wandb_run is None:
        return
    payload: dict[str, object] = {
        GLOBAL_STEP: checkpoint_step_value,
        EVAL_CHECKPOINT_STEP: checkpoint_step_value,
        EVAL_CHECKPOINT_ARTIFACT: artifact_ref,
        EVAL_REWARD_MEAN: metrics["reward_mean"],
        EVAL_REWARD_STD: metrics["reward_std"],
        EVAL_REWARD_MAX: metrics["reward_max"],
        EVAL_PROGRESS_X_MEAN: metrics["max_x_mean"],
        EVAL_PROGRESS_X_MAX: metrics["max_x_max"],
        EVAL_PROGRESS_LEVEL_X_MEAN: metrics["max_level_x_mean"],
        EVAL_PROGRESS_LEVEL_X_MAX: metrics["max_level_x_max"],
        EVAL_DEATH_COUNT: metrics["death_count"],
        EVAL_DEATH_RATE: metrics["death_rate"],
        EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
        EVAL_BEST_X: metrics["best_episode"]["max_x_pos"],
        EVAL_CONFIG_HUD_CROP_TOP: args.hud_crop_top,
        "eval/source": "post_train_inline",
    }
    payload.update(flat_numeric_metrics(metrics, "eval/done/"))
    payload.update(flat_numeric_metrics(metrics, "eval/info/"))
    # Keep W&B's internal history cursor monotonic after training. `global_step`
    # and `eval/checkpoint/step` carry the checkpoint identity for charts/queries.
    wandb_run.log(payload)
    update_best_checkpoint_summary(
        wandb_run,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        checkpoint_step_value=checkpoint_step_value,
        artifact_ref=artifact_ref,
    )


def eval_config_from_training_config(config):
    if not config.done_on_events:
        return config
    return replace(config, done_on_events=())


def evaluate_checkpoints_after_training(
    *,
    args,
    config,
    checkpoint_dir: str | Path,
    wandb_run,
) -> list[dict[str, object]]:
    if not args.post_train_eval:
        print("post-training checkpoint eval disabled", flush=True)
        return []
    paths = checkpoint_paths_for_eval(checkpoint_dir)
    if not paths:
        print("post-training checkpoint eval skipped: no checkpoints found", flush=True)
        return []
    episodes = int(args.post_train_eval_episodes)
    n_envs = int(args.post_train_eval_n_envs)
    max_steps = int(args.post_train_eval_max_steps or args.max_episode_steps)
    if episodes < 1:
        raise ValueError("--post-train-eval-episodes must be >= 1")
    if n_envs < 1:
        raise ValueError("--post-train-eval-n-envs must be >= 1")
    if max_steps < 1:
        raise ValueError("--post-train-eval-max-steps/--max-episode-steps must be >= 1")

    eval_config = eval_config_from_training_config(config)
    results: list[dict[str, object]] = []
    for checkpoint_path in paths:
        step = checkpoint_step(checkpoint_path)
        if step is None:
            print(f"post-training checkpoint eval skipped: unknown step {checkpoint_path}", flush=True)
            continue
        print(f"post-training checkpoint eval step={step} path={checkpoint_path}", flush=True)
        eval_model = PPO.load(checkpoint_path, device=resolve_sb3_device(args.device))
        metrics, _video_path = evaluate_model_episodes(
            model=eval_model,
            config=eval_config,
            episodes=episodes,
            seed=DEFAULT_EVAL_SEED,
            max_steps=max_steps,
            deterministic=not bool(args.post_train_eval_stochastic),
            n_envs=n_envs,
            progress=True,
            progress_description=f"eval checkpoint {step}",
            extra={
                "checkpoint_step": step,
                "checkpoint_artifact": str(checkpoint_path),
                "eval_source": "post_train_inline",
            },
        )
        artifact_ref = eval_checkpoint_artifact_ref(args, checkpoint_path, step)
        log_checkpoint_eval_metrics(
            wandb_run,
            args=args,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_step_value=step,
            artifact_ref=artifact_ref,
        )
        summary = {
            "checkpoint_step": step,
            "checkpoint_path": str(checkpoint_path),
            "completion_min": eval_score(metrics)[0],
            "completion_mean": eval_score(metrics)[1],
            "steps_to_completion_goal": step if eval_score(metrics)[2] > float("-inf") else None,
            "reward_mean": float(metrics["reward_mean"]),
        }
        results.append(summary)
        print(
            "post-training checkpoint eval "
            f"step={step} completion_min={summary['completion_min']:.3f} "
            f"completion_mean={summary['completion_mean']:.3f} "
            f"reward_mean={summary['reward_mean']:.2f}",
            flush=True,
        )
    return results


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
    assert_rom_imported(args.game)
    set_random_seed(args.seed)

    run_dir = default_run_dir(args.run_name, args.runs_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    write_run_description(args, run_dir)
    if args.run_description.strip():
        print(f"run description: {args.run_description.strip()}", flush=True)
    else:
        print("warning: --run-description is empty", flush=True)

    config = resolve_env_config(
        env_config_from_args(args, include_states=True, include_env_threads=True)
    )
    config = resolve_mixed_state_config(config, n_envs=args.n_envs)
    wandb_run = init_wandb(args, run_dir, config)
    graceful_stop_flag = GracefulStopFlag()
    graceful_stop_signal = install_graceful_stop_handler(graceful_stop_flag)
    if graceful_stop_signal is not None:
        print(f"graceful stop signal: {signal_name(graceful_stop_signal)}", flush=True)

    env = make_training_vec_env(config=config, n_envs=args.n_envs, seed=args.seed)
    device = resolve_sb3_device(args.device)
    if args.torch_num_threads > 0:
        import torch

        torch.set_num_threads(args.torch_num_threads)
        print(f"Using torch num threads: {torch.get_num_threads()}", flush=True)
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
        *(
            [
                MetricThresholdStopCallback(
                    marker_path=Path(run_dir) / "early_stop.txt",
                    detector=args.early_stop,
                    metric_name=args.early_stop_metric,
                    threshold=args.early_stop_threshold,
                    operator=args.early_stop_operator,
                )
            ]
            if args.early_stop or args.early_stop_metric
            else []
        ),
        RolloutDiagnosticsCallback(wandb_run=wandb_run),
        RewardComponentDiagnosticsCallback(),
    ]
    artifact_callback = None
    checkpoint_timing_state = None
    checkpoint_save_freq = checkpoint_save_frequency(args.checkpoint_freq, args.n_envs)
    if checkpoint_save_freq is not None:
        checkpoint_timing_state = CheckpointArtifactTimingState()
        artifact_callback = WandbCheckpointArtifactCallback(
            wandb_run,
            args,
            config,
            checkpoint_dir,
            scan_freq=checkpoint_save_freq,
            timing_state=checkpoint_timing_state,
        )
        callbacks.extend(
            [
                TimedCheckpointCallback(
                    save_freq=checkpoint_save_freq,
                    save_path=checkpoint_dir,
                    name_prefix=checkpoint_prefix(config.game),
                    timing_state=checkpoint_timing_state,
                ),
                artifact_callback,
            ]
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
    print("training-loop eval disabled; evaluate checkpoint artifacts out of process")

    final_model_path = Path(run_dir, "final_model.zip")
    env_closed = False
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        if graceful_stop_flag.requested and checkpoint_save_freq is not None:
            interrupted_checkpoint_path = (
                Path(checkpoint_dir)
                / f"{checkpoint_prefix(config.game)}_interrupted_{model.num_timesteps}_steps.zip"
            )
            save_started_at = time.perf_counter()
            if checkpoint_timing_state is not None:
                checkpoint_timing_state.begin(model.num_timesteps, save_started_at)
            model.save(interrupted_checkpoint_path)
            if checkpoint_timing_state is not None:
                checkpoint_timing_state.record_save(
                    model.num_timesteps,
                    time.perf_counter() - save_started_at,
                )
            print(f"saved interrupted checkpoint {interrupted_checkpoint_path}", flush=True)
        final_save_started_at = time.perf_counter()
        model.save(os.path.join(run_dir, "final_model"))
        final_save_seconds = time.perf_counter() - final_save_started_at
        if artifact_callback is not None:
            artifact_callback.log_new_checkpoints()
        final_aliases = ["final", "latest"]
        if graceful_stop_flag.requested:
            final_aliases.append("interrupted")
        log_wandb_model_artifact(
            wandb_run,
            args,
            config,
            final_model_path,
            kind="final",
            aliases=final_aliases,
            metric_step=model.num_timesteps,
            local_save_seconds=final_save_seconds,
        )
        env.close()
        env_closed = True
        checkpoint_eval_results = evaluate_checkpoints_after_training(
            args=args,
            config=config,
            checkpoint_dir=checkpoint_dir,
            wandb_run=wandb_run,
        )
        if checkpoint_eval_results:
            (Path(run_dir) / "checkpoint_eval_summary.json").write_text(
                json.dumps(checkpoint_eval_results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        write_wandb_url(wandb_run, run_dir)
    finally:
        if not env_closed:
            env.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f"saved {final_model_path}")


if __name__ == "__main__":
    main()
