from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from rlab.artifacts import env_config_from_model_metadata, write_model_metadata
from rlab.env import EnvConfig, make_eval_vec_env, resolve_env_config
from rlab.eval_metrics import is_level_complete, single_env_action
from rlab.model_sources import add_model_source_args, resolve_single_model_source
from rlab.play import model_observation, task_info_value_from_info
from rlab.train import resolve_sb3_device
from rlab.train_config import DEVICE_CHOICES


DEFAULT_TEACHERS = (
    "Level1-1=hf://tsilva/SuperMarioBros-NES_Level1-1",
    "Level1-2=hf://tsilva/SuperMarioBros-NES_Level1-2",
)
DEFAULT_STATES = ("Level1-1", "Level1-2")
DEFAULT_INFO_VALUES = ((0, 0), (0, 1))


def mario_student_config(*, done_on_events: Sequence[str] = ()) -> EnvConfig:
    return resolve_env_config(
        EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            states=DEFAULT_STATES,
            state_probs=(0.5, 0.5),
            task_conditioning=True,
            task_conditioning_info_vars=("levelHi", "levelLo"),
            task_conditioning_info_values=DEFAULT_INFO_VALUES,
            frame_skip=4,
            max_pool_frames=False,
            sticky_action_prob=0.0,
            max_episode_steps=0,
            observation_size=84,
            obs_crop=(32, 0, 0, 0),
            obs_resize_algorithm="area",
            reward_mode="score",
            progress_reward_scale=1.0,
            terminal_reward=50.0,
            death_penalty=25.0,
            completion_reward=0.0,
            reward_scale=10.0,
            score_progress_clipped=False,
            done_on_events=tuple(done_on_events),
            env_threads=4,
        )
    )


def parse_teacher(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("teacher must be STATE=MODEL_REF")
    state, ref = value.split("=", 1)
    state = state.strip()
    ref = ref.strip()
    if not state or not ref:
        raise argparse.ArgumentTypeError("teacher must be STATE=MODEL_REF")
    return state, ref


def resolve_model_ref(ref: str, *, hf_model_root: Path, device: str) -> tuple[PPO, Path, EnvConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    add_model_source_args(parser, positional_artifact=True, default_kind="final")
    args = parser.parse_args([ref])
    args.hf_model_root = str(hf_model_root)
    source = resolve_single_model_source(args)
    model = PPO.load(source.model_path, device=resolve_sb3_device(device))
    fallback = EnvConfig(game="SuperMarioBros-Nes-v0")
    config = env_config_from_model_metadata(source.model_path, fallback=fallback) or fallback
    return model, source.model_path, resolve_env_config(config)


def single_state_student_config(config: EnvConfig, state: str) -> EnvConfig:
    return replace(
        config,
        state=state,
        states=(),
        state_probs=(),
        done_on_events=(),
        max_episode_steps=0,
    )


def teacher_env_config(config: EnvConfig, state: str) -> EnvConfig:
    return replace(
        config,
        state=state,
        states=(),
        state_probs=(),
        max_episode_steps=0,
    )


def append_sample(
    images: list[np.ndarray],
    tasks: list[np.ndarray],
    actions: list[int],
    obs: dict[str, np.ndarray],
    action: int,
) -> None:
    images.append(np.asarray(obs["image"][0], dtype=np.uint8).copy())
    tasks.append(np.asarray(obs["task"][0], dtype=np.float32).copy())
    actions.append(int(action))


def collect_teacher_samples(
    *,
    state: str,
    teacher: PPO,
    teacher_config: EnvConfig,
    student_config: EnvConfig,
    seed: int,
    max_samples: int,
    max_steps_per_episode: int,
    deterministic: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    teacher_env = make_eval_vec_env(config=teacher_env_config(teacher_config, state), n_envs=1, seed=seed)
    student_env = make_eval_vec_env(config=single_state_student_config(student_config, state), n_envs=1, seed=seed)
    images: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    actions: list[int] = []
    episodes = 0
    completions = 0
    deaths = 0
    resets = 0

    try:
        teacher_obs = teacher_env.reset()
        student_obs = student_env.reset()
        steps_this_episode = 0
        while len(actions) < max_samples:
            action, _ = teacher.predict(teacher_obs, deterministic=deterministic)
            action_value = single_env_action(action)
            append_sample(images, tasks, actions, student_obs, int(action_value))
            teacher_obs, _, teacher_dones, teacher_infos = teacher_env.step(np.asarray([action_value]))
            student_obs, _, student_dones, student_infos = student_env.step(np.asarray([action_value]))
            steps_this_episode += 1
            teacher_info = dict(teacher_infos[0])
            student_info = dict(student_infos[0])
            completed = is_level_complete(student_info) or is_level_complete(teacher_info)
            died = bool(student_info.get("died") or teacher_info.get("died"))
            timed_out = steps_this_episode >= max_steps_per_episode
            terminated = bool(student_dones[0]) or bool(teacher_dones[0])
            if completed or died or timed_out or terminated:
                episodes += 1
                completions += int(completed)
                deaths += int(died)
                resets += 1
                teacher_env.seed(seed + resets)
                student_env.seed(seed + resets)
                teacher_obs = teacher_env.reset()
                student_obs = student_env.reset()
                steps_this_episode = 0
    finally:
        teacher_env.close()
        student_env.close()

    return (
        {
            "image": np.stack(images, axis=0),
            "task": np.stack(tasks, axis=0),
            "action": np.asarray(actions, dtype=np.int64),
        },
        {
            "state": state,
            "samples": len(actions),
            "episodes": episodes,
            "completions": completions,
            "deaths": deaths,
        },
    )


def concatenate_datasets(datasets: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        "image": np.concatenate([dataset["image"] for dataset in datasets], axis=0),
        "task": np.concatenate([dataset["task"] for dataset in datasets], axis=0),
        "action": np.concatenate([dataset["action"] for dataset in datasets], axis=0),
    }


def train_student(
    *,
    dataset: dict[str, np.ndarray],
    env_config: EnvConfig,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    net_arch: Sequence[int],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_eval_vec_env(config=env_config, n_envs=1, seed=seed)
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=512,
            batch_size=batch_size,
            n_epochs=1,
            gamma=0.9,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=0.0,
            normalize_advantage=False,
            policy_kwargs={"net_arch": {"pi": list(net_arch), "vf": list(net_arch)}},
            device=resolve_sb3_device(device),
            seed=seed,
            verbose=0,
        )
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate, eps=1e-8)
        sample_count = int(dataset["action"].shape[0])
        losses: list[float] = []
        accuracies: list[float] = []
        for _epoch in range(epochs):
            indices = rng.permutation(sample_count)
            for start in range(0, sample_count, batch_size):
                batch_idx = indices[start : start + batch_size]
                obs_batch = {
                    "image": dataset["image"][batch_idx],
                    "task": dataset["task"][batch_idx],
                }
                obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
                action_tensor = torch.as_tensor(
                    dataset["action"][batch_idx],
                    dtype=torch.long,
                    device=model.policy.device,
                )
                distribution = model.policy.get_distribution(obs_tensor).distribution
                loss = F.cross_entropy(distribution.logits, action_tensor)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    pred = torch.argmax(distribution.logits, dim=1)
                    accuracies.append(float((pred == action_tensor).float().mean().item()))
                    losses.append(float(loss.item()))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        args = argparse.Namespace(
            run_name=output_path.stem,
            run_description="Behavior-cloned Mario Level1-1/Level1-2 distilled policy",
            goal_slug="Levels_1-1_1-2",
            recipe_slug="distill-bc",
            recipe_path="",
            queue_train_job_id=0,
            runtime_image_ref="",
            run_target="local-macbook",
        )
        write_model_metadata(output_path, args, env_config, "final")
    finally:
        env.close()

    return {
        "samples": int(dataset["action"].shape[0]),
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": float(np.mean(losses[-10:])) if losses else None,
        "accuracy": float(np.mean(accuracies[-10:])) if accuracies else None,
        "model": str(output_path),
    }


def run_sequence_episode(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = make_eval_vec_env(config=replace(config, state="Level1-1", states=(), state_probs=()), n_envs=1, seed=seed)
    try:
        obs = env.reset()
        active_info_value: tuple[int | str, ...] | None = (0, 0)
        reward_total = 0.0
        max_x = 0
        clears: list[str] = []
        max_completed_level_count = 0
        final_info: dict[str, Any] = {}
        died = False
        terminated = False
        for step in range(1, max_steps + 1):
            if isinstance(obs, dict):
                model_obs = obs
            else:
                model_obs = model_observation(
                    model,
                    np.asarray(obs),
                    config,
                    active_info_value=active_info_value,
                )
            action, _ = model.predict(model_obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            info = dict(infos[0])
            final_info = info
            reward_total += float(rewards[0])
            max_x = max(max_x, int(info.get("max_x_pos", 0)))
            next_info_value = task_info_value_from_info(info, config)
            if next_info_value is not None:
                active_info_value = next_info_value
            if bool(info.get("died")):
                died = True
            terminated = bool(dones[0])
            completed_count = int(info.get("completed_level_count") or max_completed_level_count)
            max_completed_level_count = max(max_completed_level_count, completed_count)
            while len(clears) < max_completed_level_count:
                clears.append(f"Level1-{len(clears) + 1}")
            if max_completed_level_count >= 2 or died or terminated:
                return {
                    "seed": seed,
                    "steps": step,
                    "reward": reward_total,
                    "max_x_pos": max_x,
                    "clears": clears,
                    "success": max_completed_level_count >= 2 and not died,
                    "died": died,
                    "terminated": terminated,
                    "final_info": final_info,
                }
        return {
            "seed": seed,
            "steps": max_steps,
            "reward": reward_total,
            "max_x_pos": max_x,
            "clears": clears,
            "success": False,
            "died": died,
            "terminated": terminated,
            "truncated": True,
            "final_info": final_info,
        }
    finally:
        env.close()


def cmd_train(args: argparse.Namespace) -> int:
    teachers = [parse_teacher(value) for value in args.teacher]
    student_config = mario_student_config()
    datasets: list[dict[str, np.ndarray]] = []
    collection_summaries: list[dict[str, Any]] = []
    teacher_paths: dict[str, str] = {}
    for idx, (state, ref) in enumerate(teachers):
        teacher, model_path, config = resolve_model_ref(
            ref,
            hf_model_root=Path(args.hf_model_root),
            device=args.device,
        )
        dataset, summary = collect_teacher_samples(
            state=state,
            teacher=teacher,
            teacher_config=config,
            student_config=student_config,
            seed=args.seed + idx * 1000,
            max_samples=args.samples_per_state,
            max_steps_per_episode=args.max_steps_per_episode,
            deterministic=args.deterministic_teacher,
        )
        datasets.append(dataset)
        collection_summaries.append(summary)
        teacher_paths[state] = str(model_path)
        print(json.dumps(summary, sort_keys=True), flush=True)
    dataset = concatenate_datasets(datasets)
    started_at = time.perf_counter()
    train_summary = train_student(
        dataset=dataset,
        env_config=student_config,
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        net_arch=tuple(args.net_arch),
    )
    summary = {
        "kind": "distill_train",
        "elapsed_seconds": time.perf_counter() - started_at,
        "teachers": teacher_paths,
        "collection": collection_summaries,
        "train": train_summary,
    }
    summary_path = Path(args.output).with_suffix(".distill.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def cmd_sequence_eval(args: argparse.Namespace) -> int:
    model = PPO.load(args.model, device=resolve_sb3_device(args.device))
    config = mario_student_config()
    results = [
        run_sequence_episode(
            model=model,
            config=config,
            seed=args.seed + idx,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        for idx in range(args.episodes)
    ]
    success_count = sum(1 for result in results if result["success"])
    summary = {
        "kind": "sequence_eval",
        "model": args.model,
        "episodes": args.episodes,
        "success_count": success_count,
        "success_rate": success_count / args.episodes if args.episodes else 0.0,
        "results": results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlab distill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="behavior-clone teacher actions into one student")
    train.add_argument("--teacher", action="append", default=list(DEFAULT_TEACHERS))
    train.add_argument("--hf-model-root", default="artifacts/hf-models")
    train.add_argument("--output", default="runs/distill/levels_1_1_1_2_bc/model.zip")
    train.add_argument("--samples-per-state", type=int, default=2048)
    train.add_argument("--max-steps-per-episode", type=int, default=4500)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    train.add_argument("--net-arch", type=int, nargs="+", default=[512, 512])
    train.add_argument("--deterministic-teacher", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(func=cmd_train)

    seq = subparsers.add_parser("sequence-eval", help="verify Level1-1 then Level1-2 without event resets")
    seq.add_argument("--model", required=True)
    seq.add_argument("--episodes", type=int, default=20)
    seq.add_argument("--max-steps", type=int, default=9000)
    seq.add_argument("--seed", type=int, default=10000)
    seq.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    seq.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    seq.add_argument("--output")
    seq.set_defaults(func=cmd_sequence_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
