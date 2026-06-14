from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed

from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import (
    DEFAULT_HUD_CROP_TOP,
    EnvConfig,
    assert_rom_imported,
    default_run_dir,
    make_vec_envs,
)
from mario_ppo.eval_metrics import MarioEvalCallback
from mario_ppo.wandb_utils import DEFAULT_WANDB_PROJECT, load_wandb_env


def parse_states(value: str) -> tuple[str, ...]:
    return tuple(state.strip() for state in value.split(",") if state.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on SuperMarioBros-Nes-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--env-threads",
        type=int,
        default=0,
        help="Native stable-retro env threads; <=0 keeps min(n_envs, 16).",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="PyTorch CPU intra-op threads; <=0 leaves the torch default.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-name", default="ppo_level1_1")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument(
        "--states",
        default="",
        help="Comma-separated training states. If set, vector workers cycle through these states by rank.",
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument(
        "--max-pool-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Max-pool over the last two raw frames inside each frame-skip step.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=4500)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize; 32 removes the Mario HUD.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Training-loop eval frequency. Keep 0 to evaluate checkpoints out of process.",
    )
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument(
        "--completion-x-threshold",
        type=int,
        default=3160,
        help="Treat an episode as level-complete if max_x_pos reaches this value; set <=0 to disable.",
    )
    parser.add_argument(
        "--no-eval-videos", action="store_true", help="Disable best-episode eval videos"
    )
    parser.add_argument("--eval-video-fps", type=float, default=30.0)
    parser.add_argument("--eval-video-scale", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--stop-completion-episode-window",
        type=int,
        default=0,
        help=(
            "Stop when completion rate over the last N completed training episodes "
            "reaches --stop-completion-rate-threshold; <=0 disables this early stop."
        ),
    )
    parser.add_argument(
        "--stop-completion-rate-threshold",
        type=float,
        default=0.0,
        help="Completion-rate threshold over completed training episodes for early stopping.",
    )
    parser.add_argument(
        "--stop-completion-rolling-window",
        type=int,
        default=0,
        help=(
            "Stop when rolling mean completion events per PPO rollout reaches the "
            "configured threshold; <=0 disables this early stop."
        ),
    )
    parser.add_argument(
        "--stop-completion-rolling-threshold",
        type=float,
        default=0.0,
        help="Rolling mean completion-events-per-rollout threshold for early stopping.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate-final",
        type=float,
        default=None,
        help="If set, linearly decay learning rate from --learning-rate to this value over training.",
    )
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--ent-coef-final",
        type=float,
        default=None,
        help="If set, linearly decay entropy coefficient from --ent-coef to this value.",
    )
    parser.add_argument(
        "--ent-coef-schedule-timesteps",
        type=int,
        default=0,
        help=("Timesteps over which to decay entropy coefficient; <=0 decays over --timesteps."),
    )
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize PPO advantages before policy updates.",
    )
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--use-retro-reward", action="store_true")
    parser.add_argument("--clip-rewards", action="store_true")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "bounded", "additive", "score"],
        default="baseline",
        help="baseline matches the upstream score/env reward; bounded uses capped progress; additive is legacy shaping; score adds emulator reward and score deltas.",
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument(
        "--terminate-on-level-change",
        action="store_true",
        help="End the episode when stable-retro reports a new level via levelHi/levelLo.",
    )
    parser.add_argument(
        "--terminate-on-completion",
        action="store_true",
        help="End the episode on either real level change or the configured completion x-threshold.",
    )
    parser.add_argument("--action-set", choices=["simple", "right"], default="simple")
    parser.add_argument("--resume", help="Path to an existing PPO .zip checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--no-wandb-artifacts", action="store_true", help="Disable W&B model uploads"
    )
    return parser


def init_wandb(args: argparse.Namespace, run_dir: str, config: EnvConfig):
    if not args.wandb:
        return None

    load_wandb_env()

    wandb_dir = os.path.abspath(run_dir)
    wandb_aux_dir = os.path.join(wandb_dir, "wandb")
    wandb_env_dirs = {
        "WANDB_DIR": wandb_dir,
        "WANDB_CACHE_DIR": os.path.join(wandb_aux_dir, "cache"),
        "WANDB_CONFIG_DIR": os.path.join(wandb_aux_dir, "config"),
        "WANDB_DATA_DIR": os.path.join(wandb_aux_dir, "data"),
        "WANDB_ARTIFACT_DIR": os.path.join(wandb_aux_dir, "artifacts"),
    }
    for env_name, path in wandb_env_dirs.items():
        os.environ.setdefault(env_name, path)
        os.makedirs(os.environ[env_name], exist_ok=True)

    import wandb

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    wandb_config: dict[str, Any] = {
        **vars(args),
        "game": config.game,
        "state": config.state,
        "states": list(config.states),
        "frame_skip": config.frame_skip,
        "max_pool_frames": config.max_pool_frames,
        "max_episode_steps": config.max_episode_steps,
        "observation_size": config.observation_size,
        "hud_crop_top": config.hud_crop_top,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "use_retro_reward": config.use_retro_reward,
        "reward_mode": config.reward_mode,
        "progress_reward_cap": config.progress_reward_cap,
        "progress_reward_scale": config.progress_reward_scale,
        "terminal_reward": config.terminal_reward,
        "reward_scale": config.reward_scale,
        "time_penalty": config.time_penalty,
        "death_penalty": config.death_penalty,
        "completion_reward": config.completion_reward,
        "completion_x_threshold": config.completion_x_threshold,
        "terminate_on_life_loss": config.terminate_on_life_loss,
        "terminate_on_level_change": config.terminate_on_level_change,
        "terminate_on_completion": config.terminate_on_completion,
        "action_set": config.action_set,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.run_name,
        tags=tags,
        config=wandb_config,
        dir=wandb_dir,
        sync_tensorboard=True,
        save_code=True,
        mode=args.wandb_mode,
    )


def wandb_artifacts_enabled(wandb_run, args: argparse.Namespace) -> bool:
    return wandb_run is not None and not args.no_wandb_artifacts


def sanitize_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "mario-ppo"


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def format_wandb_run_path(run_path) -> str:
    if isinstance(run_path, (list, tuple)):
        return "/".join(str(part) for part in run_path)
    return str(run_path)


def log_wandb_model_artifact(
    wandb_run,
    args: argparse.Namespace,
    model_path: Path,
    kind: str,
    aliases: list[str] | None = None,
) -> None:
    if not wandb_artifacts_enabled(wandb_run, args):
        return
    if not model_path.is_file():
        return

    import wandb

    artifact_name = f"{sanitize_artifact_name(args.run_name)}-{kind}"
    step = checkpoint_step(model_path)
    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "kind": kind,
        "filename": model_path.name,
        "checkpoint_step": step,
    }
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        metadata["wandb_run_id"] = run_id
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        metadata["wandb_run_path"] = format_wandb_run_path(run_path)

    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata=metadata,
    )
    artifact.add_file(str(model_path), name=model_path.name)
    wandb_run.log_artifact(artifact, aliases=aliases)
    print(f"wandb artifact logged: {artifact_name} ({model_path})")


class WandbCheckpointArtifactCallback(BaseCallback):
    def __init__(
        self,
        wandb_run,
        args: argparse.Namespace,
        checkpoint_dir: str,
        scan_freq: int,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scan_freq = scan_freq
        self.logged_paths: set[Path] = set()

    def _on_step(self) -> bool:
        if self.scan_freq <= 1 or self.n_calls % self.scan_freq == 0:
            self.log_new_checkpoints()
        return True

    def log_new_checkpoints(self) -> None:
        if not wandb_artifacts_enabled(self.wandb_run, self.args):
            return

        for checkpoint_path in sorted(self.checkpoint_dir.glob("*.zip")):
            resolved_path = checkpoint_path.resolve()
            if resolved_path in self.logged_paths:
                continue
            step = checkpoint_step(checkpoint_path)
            aliases = ["latest"]
            if step is not None:
                aliases.append(f"step-{step}")
            log_wandb_model_artifact(
                self.wandb_run,
                self.args,
                checkpoint_path,
                kind="checkpoint",
                aliases=aliases,
            )
            self.logged_paths.add(resolved_path)


class RollingCompletionStopCallback(BaseCallback):
    def __init__(
        self,
        rolling_window: int,
        threshold: float,
        run_dir: str,
        wandb_run=None,
    ):
        super().__init__()
        self.rolling_window = rolling_window
        self.threshold = threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.rollout_completion_count = 0
        self.total_completion_count = 0
        self.rollout_counts: deque[int] = deque(maxlen=rolling_window)
        self.rolling_mean = 0.0
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        for info in infos:
            if bool(info.get("completion_event", info.get("level_complete", False))):
                self.rollout_completion_count += 1
                self.total_completion_count += 1

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_counts.append(self.rollout_completion_count)
        self.rolling_mean = sum(self.rollout_counts) / len(self.rollout_counts)
        window_full = len(self.rollout_counts) >= self.rolling_window

        self.logger.record("train/completion_events_rollout", self.rollout_completion_count)
        self.logger.record("train/completion_events_rolling_mean", self.rolling_mean)
        self.logger.record("train/completion_events_total", self.total_completion_count)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "train/completion_events_rollout": self.rollout_completion_count,
                    "train/completion_events_rolling_mean": self.rolling_mean,
                    "train/completion_events_total": self.total_completion_count,
                    "global_step": self.num_timesteps,
                },
                step=self.num_timesteps,
            )

        print(
            "completion rolling: "
            f"rollout={self.rollout_completion_count} "
            f"mean={self.rolling_mean:.3f}/{self.threshold:g} "
            f"window={len(self.rollout_counts)}/{self.rolling_window} "
            f"total={self.total_completion_count}",
            flush=True,
        )

        if window_full and self.rolling_mean >= self.threshold:
            self.stop_requested = True
            stop_path = self.run_dir / "early_stop.txt"
            stop_path.write_text(
                "\n".join(
                    [
                        "reason=rolling_completion_threshold",
                        f"timesteps={self.num_timesteps}",
                        f"rolling_window={self.rolling_window}",
                        f"rolling_mean={self.rolling_mean:.6f}",
                        f"threshold={self.threshold:.6f}",
                        f"total_completion_count={self.total_completion_count}",
                    ],
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "early stop requested: "
                f"rolling completion mean {self.rolling_mean:.3f} >= {self.threshold:g}",
                flush=True,
            )

        self.rollout_completion_count = 0


class TrainingCompletionRateStopCallback(BaseCallback):
    def __init__(
        self,
        episode_window: int,
        rate_threshold: float,
        run_dir: str,
        wandb_run=None,
    ):
        super().__init__()
        if not 0.0 < rate_threshold <= 1.0:
            raise ValueError("rate_threshold must be in (0, 1]")
        self.episode_window = episode_window
        self.rate_threshold = rate_threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.completed_episode_outcomes: deque[int] = deque(maxlen=episode_window)
        self.total_terminal_episodes = 0
        self.total_completed_episodes = 0
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos, strict=False):
            if not bool(done) or bool(info.get("global_reset", False)):
                continue

            completed = bool(info.get("completion_event", info.get("level_complete", False)))
            self.completed_episode_outcomes.append(int(completed))
            self.total_terminal_episodes += 1
            if completed:
                self.total_completed_episodes += 1

            completion_rate = self.completion_rate
            self.logger.record("train/completion_episode_rate", completion_rate)
            self.logger.record(
                "train/completion_episode_window_size", len(self.completed_episode_outcomes)
            )
            self.logger.record("train/completion_episodes_total", self.total_completed_episodes)
            self.logger.record("train/terminal_episodes_total", self.total_terminal_episodes)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/completion_episode_rate": completion_rate,
                        "train/completion_episode_window_size": len(
                            self.completed_episode_outcomes,
                        ),
                        "train/completion_episodes_total": self.total_completed_episodes,
                        "train/terminal_episodes_total": self.total_terminal_episodes,
                        "global_step": self.num_timesteps,
                    },
                    step=self.num_timesteps,
                )

            if (
                len(self.completed_episode_outcomes) >= self.episode_window
                and completion_rate >= self.rate_threshold
            ):
                self.request_stop(completion_rate)
                return False

        return True

    @property
    def completion_rate(self) -> float:
        if not self.completed_episode_outcomes:
            return 0.0
        return sum(self.completed_episode_outcomes) / len(self.completed_episode_outcomes)

    def request_stop(self, completion_rate: float) -> None:
        self.stop_requested = True
        stop_path = self.run_dir / "early_stop.txt"
        stop_path.write_text(
            "\n".join(
                [
                    "reason=training_completion_rate_threshold",
                    f"timesteps={self.num_timesteps}",
                    f"episode_window={self.episode_window}",
                    f"completion_rate={completion_rate:.6f}",
                    f"threshold={self.rate_threshold:.6f}",
                    f"total_terminal_episodes={self.total_terminal_episodes}",
                    f"total_completed_episodes={self.total_completed_episodes}",
                ],
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "early stop requested: "
            f"training completion rate {completion_rate:.3f} >= {self.rate_threshold:g} "
            f"over last {self.episode_window} completed episodes",
            flush=True,
        )


def write_wandb_url(wandb_run, run_dir: str) -> None:
    if wandb_run is None:
        return

    run_url = getattr(wandb_run, "url", None)
    if run_url:
        Path(run_dir, "wandb_url.txt").write_text(f"{run_url}\n", encoding="utf-8")
    run_id = getattr(wandb_run, "id", None)
    if run_id:
        Path(run_dir, "wandb_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    run_path = getattr(wandb_run, "path", None)
    if run_path:
        Path(run_dir, "wandb_run_path.txt").write_text(
            f"{format_wandb_run_path(run_path)}\n",
            encoding="utf-8",
        )


def linear_decay_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


def learning_rate_schedule(args: argparse.Namespace) -> float | Callable[[float], float]:
    if args.learning_rate_final is None:
        return args.learning_rate
    return linear_decay_schedule(args.learning_rate, args.learning_rate_final)


class EntropyCoefficientScheduleCallback(BaseCallback):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        schedule_timesteps: int,
    ):
        super().__init__()
        if schedule_timesteps <= 0:
            raise ValueError("schedule_timesteps must be positive")
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps

    def _current_value(self) -> float:
        progress = min(max(self.num_timesteps / self.schedule_timesteps, 0.0), 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._current_value()

    def _on_step(self) -> bool:
        ent_coef = self._current_value()
        self.model.ent_coef = ent_coef
        self.logger.record("train/ent_coef", ent_coef)
        return True


def apply_resume_hyperparameters(model: PPO, args: argparse.Namespace) -> None:
    lr_schedule = learning_rate_schedule(args)
    model.learning_rate = lr_schedule
    model.lr_schedule = get_schedule_fn(lr_schedule)
    model.ent_coef = args.ent_coef
    model.vf_coef = args.vf_coef
    model.n_epochs = args.n_epochs
    model.batch_size = args.batch_size
    model.clip_range = get_schedule_fn(args.clip_range)
    model.normalize_advantage = args.normalize_advantage
    model.target_kl = args.target_kl
    model.policy.optimizer.defaults["eps"] = args.adam_eps
    for param_group in model.policy.optimizer.param_groups:
        param_group["eps"] = args.adam_eps


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    set_random_seed(args.seed)

    run_dir = default_run_dir(args.run_name, args.runs_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    config = EnvConfig(
        state=args.state,
        states=parse_states(args.states),
        frame_skip=args.frame_skip,
        max_pool_frames=args.max_pool_frames,
        max_episode_steps=args.max_episode_steps,
        hud_crop_top=args.hud_crop_top,
        use_retro_reward=args.use_retro_reward,
        clip_rewards=args.clip_rewards,
        reward_mode=args.reward_mode,
        progress_reward_cap=args.progress_reward_cap,
        progress_reward_scale=args.progress_reward_scale,
        terminal_reward=args.terminal_reward,
        reward_scale=args.reward_scale,
        time_penalty=args.time_penalty,
        death_penalty=args.death_penalty,
        completion_reward=args.completion_reward,
        completion_x_threshold=args.completion_x_threshold,
        terminate_on_life_loss=not args.no_terminate_on_life_loss,
        terminate_on_level_change=args.terminate_on_level_change,
        terminate_on_completion=args.terminate_on_completion,
        action_set=args.action_set,
        env_threads=args.env_threads,
    )
    wandb_run = init_wandb(args, run_dir, config)

    env = make_vec_envs(config=config, n_envs=args.n_envs, seed=args.seed)
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
