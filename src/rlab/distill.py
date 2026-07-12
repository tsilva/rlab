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
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from rlab.artifacts import env_config_from_model_metadata, write_model_metadata
from rlab.env import EnvConfig, make_eval_vec_env, resolve_env_config, with_task_termination
from rlab.eval_metrics import (
    batch_metrics_for_lane,
    drain_runtime_records,
    episode_records,
    single_env_action,
)
from rlab.model_sources import add_model_source_args, resolve_single_model_source
from rlab.device import resolve_sb3_device
from rlab.policy_observation import model_observation, task_info_value_from_info
from rlab.train_config import DEVICE_CHOICES
from rlab.task_kernels import default_task_document


DEFAULT_TEACHERS = (
    "Level1-1=hf://tsilva/SuperMarioBros-NES_Level1-1",
    "Level1-2=hf://tsilva/SuperMarioBros-NES_Level1-2",
)
DEFAULT_STATES = ("Level1-1", "Level1-2")
DEFAULT_INFO_VALUES = ((0, 0), (0, 1))


def mario_student_config(*, termination_events: Sequence[str] = ()) -> EnvConfig:
    failure = [name for name in termination_events if name != "level_change"]
    success = [name for name in termination_events if name == "level_change"]
    task = default_task_document("mario")
    task["termination"] = {
        "failure": failure,
        "success": success,
        "max_episode_steps": 0,
    }
    task["reward"].update(
        {
            "reward_mode": "score",
            "progress_reward_scale": 1.0,
            "terminal_reward": 50.0,
            "death_penalty": 25.0,
            "completion_reward": 0.0,
            "reward_scale": 10.0,
            "score_progress_clipped": False,
        }
    )
    task["conditioning"] = {
        "enabled": True,
        "signal": "level",
        "values": [list(value) for value in DEFAULT_INFO_VALUES],
    }
    return resolve_env_config(
        EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            states=DEFAULT_STATES,
            state_probs=(0.5, 0.5),
            task=task,
            frame_skip=4,
            max_pool_frames=False,
            sticky_action_prob=0.0,
            observation_size=84,
            obs_crop=(32, 0, 0, 0),
            obs_resize_algorithm="area",
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
    config = env_config_from_model_metadata(source.model_path)
    if config is None:
        raise ValueError(
            f"{source.model_path} is missing canonical environment metadata"
        )
    return model, source.model_path, resolve_env_config(config)


def single_state_student_config(config: EnvConfig, state: str) -> EnvConfig:
    return with_task_termination(
        replace(
            config,
            state=state,
            states=(),
            state_probs=(),
        ),
        failure=[],
        success=[],
        max_episode_steps=0,
    )


def teacher_env_config(config: EnvConfig, state: str) -> EnvConfig:
    return with_task_termination(
        replace(config, state=state, states=(), state_probs=()),
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


def runtime_lane_info(env: Any, provider_info: Any, lane: int = 0) -> tuple[dict[str, Any], list[Any]]:
    records = drain_runtime_records(env)
    return {**dict(provider_info), **batch_metrics_for_lane(records, lane)}, episode_records(records)


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
            teacher_info, teacher_episode_records = runtime_lane_info(
                teacher_env, teacher_infos[0]
            )
            student_info, student_episode_records = runtime_lane_info(
                student_env, student_infos[0]
            )
            completed = bool(
                student_info.get("completion_event") or teacher_info.get("completion_event")
            )
            died = bool(student_info.get("died") or teacher_info.get("died"))
            timed_out = steps_this_episode >= max_steps_per_episode
            terminated = bool(
                student_dones[0]
                or teacher_dones[0]
                or student_episode_records
                or teacher_episode_records
            )
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


def empty_sequence_dataset() -> dict[str, list[np.ndarray] | list[int]]:
    return {"image": [], "task": [], "action": []}


def append_model_sample(
    samples: dict[str, list[np.ndarray] | list[int]],
    model_obs: dict[str, np.ndarray],
    action: int,
) -> None:
    images = samples["image"]
    tasks = samples["task"]
    actions = samples["action"]
    assert isinstance(images, list)
    assert isinstance(tasks, list)
    assert isinstance(actions, list)
    images.append(np.asarray(model_obs["image"][0], dtype=np.uint8).copy())
    tasks.append(np.asarray(model_obs["task"][0], dtype=np.float32).copy())
    actions.append(int(action))


def materialize_sequence_dataset(samples: dict[str, list[np.ndarray] | list[int]]) -> dict[str, np.ndarray]:
    images = samples["image"]
    tasks = samples["task"]
    actions = samples["action"]
    assert isinstance(images, list)
    assert isinstance(tasks, list)
    assert isinstance(actions, list)
    return {
        "image": np.stack(images, axis=0),
        "task": np.stack(tasks, axis=0),
        "action": np.asarray(actions, dtype=np.int64),
    }


def _copy_teacher_module(
    student_state: dict[str, torch.Tensor],
    teacher_state: dict[str, torch.Tensor],
    *,
    student_prefix: str,
    teacher_prefix: str,
) -> None:
    for suffix in (
        "cnn.0.weight",
        "cnn.0.bias",
        "cnn.2.weight",
        "cnn.2.bias",
        "cnn.4.weight",
        "cnn.4.bias",
        "linear.0.weight",
        "linear.0.bias",
    ):
        student_state[f"{student_prefix}.{suffix}"] = teacher_state[f"{teacher_prefix}.{suffix}"].clone()


def initialize_student_from_teacher(student: PPO, teacher: PPO) -> None:
    student_state = student.policy.state_dict()
    teacher_state = teacher.policy.state_dict()
    feature_dim = teacher_state["action_net.weight"].shape[1]
    for teacher_prefix, student_prefix in (
        ("features_extractor", "features_extractor.extractors.image"),
        ("pi_features_extractor", "pi_features_extractor.extractors.image"),
        ("vf_features_extractor", "vf_features_extractor.extractors.image"),
    ):
        _copy_teacher_module(
            student_state,
            teacher_state,
            student_prefix=student_prefix,
            teacher_prefix=teacher_prefix,
        )
    for branch in ("policy_net", "value_net"):
        for key in sorted(student_state):
            prefix = f"mlp_extractor.{branch}."
            if not key.startswith(prefix) or not key.endswith(".weight"):
                continue
            layer_idx = key[len(prefix) : -len(".weight")]
            if not layer_idx.isdigit():
                continue
            weight_key = key
            bias_key = f"mlp_extractor.{branch}.{layer_idx}.bias"
            weight = student_state[weight_key]
            bias = student_state[bias_key]
            if int(weight.shape[0]) != feature_dim:
                raise ValueError(
                    f"teacher initialization needs {branch} layer {layer_idx} output dim "
                    f"{feature_dim}, got {tuple(weight.shape)}",
                )
            weight.zero_()
            bias.zero_()
            input_dim = min(int(weight.shape[1]), feature_dim)
            weight[:input_dim, :input_dim] = torch.eye(
                input_dim,
                dtype=weight.dtype,
                device=weight.device,
            )
    student_state["action_net.weight"] = teacher_state["action_net.weight"].clone()
    student_state["action_net.bias"] = teacher_state["action_net.bias"].clone()
    student_state["value_net.weight"] = teacher_state["value_net.weight"].clone()
    student_state["value_net.bias"] = teacher_state["value_net.bias"].clone()
    student.policy.load_state_dict(student_state)


def fit_policy_supervised(
    *,
    model: PPO,
    dataset: dict[str, np.ndarray],
    env_config: EnvConfig,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    freeze_image_features: bool,
    action_logit_scale: float,
    run_description: str,
    recipe_slug: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if freeze_image_features:
        for name, parameter in model.policy.named_parameters():
            if "features_extractor" in name and ".extractors.image." in name:
                parameter.requires_grad = False
    trainable_parameters = [parameter for parameter in model.policy.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable policy parameters remain")
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, eps=1e-8)
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
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 0.5)
            optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(distribution.logits, dim=1)
                accuracies.append(float((pred == action_tensor).float().mean().item()))
                losses.append(float(loss.item()))
    if action_logit_scale <= 0.0:
        raise ValueError("--action-logit-scale must be > 0")
    if action_logit_scale != 1.0:
        with torch.no_grad():
            model.policy.action_net.weight.mul_(action_logit_scale)
            model.policy.action_net.bias.mul_(action_logit_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    metadata_args = argparse.Namespace(
        run_name=output_path.stem,
        run_description=run_description,
        goal_slug="Levels_1-1_1-2",
        recipe_slug=recipe_slug,
        recipe_path="",
        queue_train_job_id=0,
        runtime_image_ref="",
        run_target="local-macbook",
    )
    write_model_metadata(output_path, metadata_args, env_config, "final")
    return {
        "samples": sample_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": float(np.mean(losses[-10:])) if losses else None,
        "accuracy": float(np.mean(accuracies[-10:])) if accuracies else None,
        "action_logit_scale": action_logit_scale,
        "freeze_image_features": freeze_image_features,
        "model": str(output_path),
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
    init_teacher: PPO | None = None,
    freeze_image_features: bool = False,
    action_logit_scale: float = 1.0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
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
            policy_kwargs={
                "activation_fn": nn.ReLU,
                "features_extractor_kwargs": {"cnn_output_dim": 512},
                "net_arch": {"pi": list(net_arch), "vf": list(net_arch)},
            },
            device=resolve_sb3_device(device),
            seed=seed,
            verbose=0,
        )
        if init_teacher is not None:
            initialize_student_from_teacher(model, init_teacher)
        return fit_policy_supervised(
            model=model,
            dataset=dataset,
            env_config=env_config,
            output_path=output_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            freeze_image_features=freeze_image_features,
            action_logit_scale=action_logit_scale,
            run_description="Behavior-cloned Mario Level1-1/Level1-2 distilled policy",
            recipe_slug="distill-bc",
        )
    finally:
        env.close()


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
            observation_spaces = getattr(model.observation_space, "spaces", None)
            if isinstance(obs, dict) and isinstance(observation_spaces, dict):
                model_obs = obs
            else:
                image_obs = obs["image"] if isinstance(obs, dict) else obs
                model_obs = model_observation(
                    model,
                    np.asarray(image_obs),
                    config,
                    active_info_value=active_info_value,
                )
            action, _ = model.predict(model_obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            info, runtime_episodes = runtime_lane_info(env, infos[0])
            final_info = info
            reward_total += float(rewards[0])
            max_x = max(max_x, int(info.get("max_x_pos", 0)))
            next_info_value = (
                (int(info["level_hi"]), int(info["level_lo"]))
                if "level_hi" in info and "level_lo" in info
                else task_info_value_from_info(info, config)
            )
            if next_info_value is not None:
                active_info_value = next_info_value
            if bool(info.get("died")):
                died = True
            terminated = bool(dones[0]) or bool(runtime_episodes)
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


def collect_success_sequence_samples(
    *,
    model: PPO,
    config: EnvConfig,
    seed: int,
    successes: int,
    max_attempts: int,
    max_steps: int,
    deterministic: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    collected = empty_sequence_dataset()
    success_summaries: list[dict[str, Any]] = []
    failure_summaries: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        env = make_eval_vec_env(
            config=replace(config, state="Level1-1", states=(), state_probs=()),
            n_envs=1,
            seed=seed + attempt,
        )
        episode_samples = empty_sequence_dataset()
        try:
            obs = env.reset()
            active_info_value: tuple[int | str, ...] | None = (0, 0)
            reward_total = 0.0
            max_x = 0
            max_completed_level_count = 0
            final_info: dict[str, Any] = {}
            died = False
            terminated = False
            steps = 0
            for step in range(1, max_steps + 1):
                steps = step
                observation_spaces = getattr(model.observation_space, "spaces", None)
                if isinstance(obs, dict) and isinstance(observation_spaces, dict):
                    model_obs = obs
                else:
                    image_obs = obs["image"] if isinstance(obs, dict) else obs
                    model_obs = model_observation(
                        model,
                        np.asarray(image_obs),
                        config,
                        active_info_value=active_info_value,
                    )
                if not isinstance(model_obs, dict):
                    raise ValueError("self-imitation requires a task-conditioned dict-observation model")
                action, _ = model.predict(model_obs, deterministic=deterministic)
                action_value = single_env_action(action)
                append_model_sample(episode_samples, model_obs, action_value)
                obs, rewards, dones, infos = env.step(np.asarray([action_value]))
                info, runtime_episodes = runtime_lane_info(env, infos[0])
                final_info = info
                reward_total += float(rewards[0])
                max_x = max(max_x, int(info.get("max_x_pos", 0)))
                next_info_value = (
                    (int(info["level_hi"]), int(info["level_lo"]))
                    if "level_hi" in info and "level_lo" in info
                    else task_info_value_from_info(info, config)
                )
                if next_info_value is not None:
                    active_info_value = next_info_value
                if bool(info.get("died")):
                    died = True
                terminated = bool(dones[0]) or bool(runtime_episodes)
                completed_count = int(info.get("completed_level_count") or max_completed_level_count)
                max_completed_level_count = max(max_completed_level_count, completed_count)
                if max_completed_level_count >= 2 or died or terminated:
                    break
            episode_summary = {
                "attempt": attempt + 1,
                "seed": seed + attempt,
                "steps": steps,
                "reward": reward_total,
                "max_x_pos": max_x,
                "completed_level_count": max_completed_level_count,
                "success": max_completed_level_count >= 2 and not died,
                "died": died,
                "terminated": terminated,
                "final_info": final_info,
            }
            if episode_summary["success"]:
                for key in ("image", "task", "action"):
                    target = collected[key]
                    source = episode_samples[key]
                    assert isinstance(target, list)
                    assert isinstance(source, list)
                    target.extend(source)
                success_summaries.append(episode_summary)
                print(json.dumps({"kind": "self_success", **episode_summary}, sort_keys=True), flush=True)
                if len(success_summaries) >= successes:
                    break
            else:
                failure_summaries.append(episode_summary)
                if (attempt + 1) % 20 == 0:
                    recent = failure_summaries[-20:]
                    print(
                        json.dumps(
                            {
                                "kind": "self_collect_progress",
                                "attempts": attempt + 1,
                                "successes": len(success_summaries),
                                "recent_max_x_pos": max(item["max_x_pos"] for item in recent),
                                "recent_cleared_level1": sum(
                                    1 for item in recent if int(item["completed_level_count"]) >= 1
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            env.close()
    image_samples = collected["image"]
    assert isinstance(image_samples, list)
    if not success_summaries or not image_samples:
        raise RuntimeError(f"collected no successful sequence trajectories in {max_attempts} attempts")
    return (
        materialize_sequence_dataset(collected),
        {
            "attempts": len(success_summaries) + len(failure_summaries),
            "requested_successes": successes,
            "successes": len(success_summaries),
            "failures": len(failure_summaries),
            "success_episodes": success_summaries,
            "failure_summary_tail": failure_summaries[-10:],
        },
    )


def cmd_train(args: argparse.Namespace) -> int:
    teachers = [parse_teacher(value) for value in args.teacher]
    student_config = mario_student_config()
    datasets: list[dict[str, np.ndarray]] = []
    collection_summaries: list[dict[str, Any]] = []
    teacher_paths: dict[str, str] = {}
    teacher_models: dict[str, PPO] = {}
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
        teacher_models[state] = teacher
        print(json.dumps(summary, sort_keys=True), flush=True)
    init_teacher = None
    if args.init_from_teacher:
        init_teacher = teacher_models.get(args.init_from_teacher)
        if init_teacher is None:
            choices = ", ".join(sorted(teacher_models))
            raise SystemExit(f"--init-from-teacher must match one of: {choices}")
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
        init_teacher=init_teacher,
        freeze_image_features=args.freeze_image_features,
        action_logit_scale=args.action_logit_scale,
    )
    summary = {
        "kind": "distill_train",
        "elapsed_seconds": time.perf_counter() - started_at,
        "teachers": teacher_paths,
        "init_from_teacher": args.init_from_teacher or None,
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


def cmd_self_train(args: argparse.Namespace) -> int:
    config = mario_student_config()
    env = make_eval_vec_env(config=config, n_envs=1, seed=args.seed)
    try:
        model = PPO.load(args.model, env=env, device=resolve_sb3_device(args.device))
        started_at = time.perf_counter()
        dataset, collection_summary = collect_success_sequence_samples(
            model=model,
            config=config,
            seed=args.seed,
            successes=args.successes,
            max_attempts=args.max_attempts,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        if args.dataset_output:
            Path(args.dataset_output).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.dataset_output,
                image=dataset["image"],
                task=dataset["task"],
                action=dataset["action"],
            )
        train_summary = fit_policy_supervised(
            model=model,
            dataset=dataset,
            env_config=config,
            output_path=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            freeze_image_features=args.freeze_image_features,
            action_logit_scale=args.action_logit_scale,
            run_description="Self-imitation distilled Mario Level1-1 into Level1-2 sequence policy",
            recipe_slug="distill-self-imitation",
        )
    finally:
        env.close()
    summary = {
        "kind": "distill_self_train",
        "base_model": args.model,
        "elapsed_seconds": time.perf_counter() - started_at,
        "collection": collection_summary,
        "train": train_summary,
    }
    summary_path = Path(args.output).with_suffix(".self_distill.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    train.add_argument("--init-from-teacher", choices=DEFAULT_STATES)
    train.add_argument("--freeze-image-features", action="store_true")
    train.add_argument("--action-logit-scale", type=float, default=1.0)
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

    self_train = subparsers.add_parser(
        "self-train",
        help="collect successful stochastic sequence rollouts and clone them back into the policy",
    )
    self_train.add_argument("--model", required=True)
    self_train.add_argument("--output", default="runs/distill/levels_1_1_1_2_self/model.zip")
    self_train.add_argument("--dataset-output")
    self_train.add_argument("--successes", type=int, default=3)
    self_train.add_argument("--max-attempts", type=int, default=200)
    self_train.add_argument("--max-steps", type=int, default=9000)
    self_train.add_argument("--epochs", type=int, default=80)
    self_train.add_argument("--batch-size", type=int, default=256)
    self_train.add_argument("--learning-rate", type=float, default=3e-5)
    self_train.add_argument("--seed", type=int, default=10000)
    self_train.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    self_train.add_argument("--freeze-image-features", action="store_true")
    self_train.add_argument("--action-logit-scale", type=float, default=5.0)
    self_train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    self_train.set_defaults(func=cmd_self_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
