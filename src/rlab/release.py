from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlab.config_validation import load_goal_contract
from rlab.dotenv import load_env_file
from rlab.env import EnvConfig, assert_rom_imported, resolve_env_config
from rlab.eval_runner import evaluate_model_episodes
from rlab.json_utils import json_safe
from rlab.model_sources import download_artifact_ref_source
from rlab.seeds import DEFAULT_EVAL_SEED, validate_eval_seed
from rlab.wandb_leaders import (
    CHECKPOINT_PRIMARY_ORDER,
    checkpoint_leader,
    checkpoint_summary_filter,
    rank_checkpoint_leaders,
    wandb_runs,
)
from rlab.wandb_utils import DEFAULT_WANDB_PROJECT_PATH


HUGGINGFACE_OWNER_ENV_KEYS = (
    "RLAB_HUGGINGFACE_OWNER",
    "RELEASE_HUGGINGFACE_OWNER",
    "HUGGINGFACE_OWNER",
    "HF_OWNER",
)


@dataclass(frozen=True)
class HuggingFaceReleaseConfig:
    owner: str
    repo: str
    card_template: str
    checkpoint_filename: str
    preview_filename: str
    include_youtube_preview: bool

    @property
    def repo_id(self) -> str:
        return f"{self.owner}/{self.repo}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release evaluated checkpoint winners from goal contracts.",
    )
    subparsers = parser.add_subparsers(dest="command")

    hf = subparsers.add_parser(
        "huggingface",
        aliases=["hf"],
        help="Publish a goal's best evaluated checkpoint to Hugging Face.",
    )
    hf.add_argument("--goal", required=True, help="Goal id or path to _goal.yaml.")
    hf.add_argument("--project", default=DEFAULT_WANDB_PROJECT_PATH)
    hf.add_argument(
        "--hf-owner",
        help=(
            "Hugging Face owner override. Defaults to RLAB_HUGGINGFACE_OWNER "
            "from .env."
        ),
    )
    hf.add_argument("--repo-root", type=Path, default=Path("."))
    hf.add_argument("--stage-root", type=Path, default=Path("runs/hf_upload"))
    hf.add_argument("--artifact-root", type=Path, default=Path("runs/wandb_artifacts"))
    hf.add_argument("--episodes", type=int, default=None)
    hf.add_argument("--max-steps", type=int, default=4500)
    hf.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    hf.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    hf.add_argument("--video-fps", type=float, default=30.0)
    hf.add_argument("--video-scale", type=int, default=4)
    hf.add_argument("--dry-run", action="store_true")
    hf.add_argument("--no-hf-upload", action="store_true")
    hf.add_argument("--no-youtube", action="store_true")
    hf.add_argument("--private", action="store_true")
    hf.add_argument("--progress", action="store_true")
    hf.set_defaults(func=cmd_huggingface)
    return parser


def resolve_goal_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    goals_dir = repo_root / "experiments" / "goals"
    for filename in ("_goal.yaml", "goal.yaml"):
        for yaml_path in sorted(goals_dir.rglob(f"{value}/{filename}")):
            if ".deprecated" not in yaml_path.parts and yaml_path.is_file():
                return yaml_path
    return goals_dir / value / "_goal.yaml"


def huggingface_owner_from_env() -> str:
    load_env_file(".env", key_filter=lambda key: key in HUGGINGFACE_OWNER_ENV_KEYS)
    for key in HUGGINGFACE_OWNER_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def release_config_from_goal(
    goal: Mapping[str, Any],
    *,
    owner: str | None = None,
) -> HuggingFaceReleaseConfig:
    release = goal.get("release")
    if not isinstance(release, Mapping):
        raise SystemExit("goal has no release section")
    huggingface = release.get("huggingface")
    if not isinstance(huggingface, Mapping):
        raise SystemExit("goal has no release.huggingface section")
    resolved_owner = (owner or huggingface_owner_from_env()).strip()
    if not resolved_owner:
        raise SystemExit(
            "Hugging Face owner is required; set RLAB_HUGGINGFACE_OWNER in .env "
            "or pass --hf-owner."
        )
    return HuggingFaceReleaseConfig(
        owner=resolved_owner,
        repo=str(huggingface["repo"]),
        card_template=str(huggingface.get("card_template", "stable-retro-sb3")),
        checkpoint_filename=str(
            huggingface.get(
                "checkpoint_filename",
                "model.zip",
            )
        ),
        preview_filename=str(huggingface.get("preview_filename", "replay.mp4")),
        include_youtube_preview=bool(huggingface.get("include_youtube_preview", True)),
    )


def _section_env(section: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    environment = section.get("environment")
    if not isinstance(environment, Mapping):
        return None, {}
    env_config = environment.get("env_config")
    config = dict(env_config) if isinstance(env_config, Mapping) else {}
    provider = environment.get("env_provider")
    return str(provider) if provider else None, config


def env_config_from_goal(goal: Mapping[str, Any]) -> EnvConfig:
    train_provider, train_config = _section_env(goal.get("train", {}))
    eval_provider, eval_config = _section_env(goal.get("eval", {}))
    provider = eval_provider or train_provider
    merged = {**train_config, **eval_config}
    if provider:
        merged["env_provider"] = provider
    fields = set(EnvConfig.__dataclass_fields__)
    config_kwargs = {key: value for key, value in merged.items() if key in fields}
    return resolve_env_config(EnvConfig(**config_kwargs))


def goal_eval_episodes(goal: Mapping[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)
    _provider, eval_config = _section_env(goal.get("eval", {}))
    value = eval_config.get("max_episodes", 1)
    return int(value)


def best_checkpoint_for_goal(goal_id: str, project: str):
    leaders = [
        leader
        for leader in (
            checkpoint_leader(run)
            for run in wandb_runs(
                project=project,
                goal=goal_id,
                extra_filter=checkpoint_summary_filter(),
                order=CHECKPOINT_PRIMARY_ORDER,
            )
        )
        if leader is not None and leader.goal_slug == goal_id
    ]
    ranked = rank_checkpoint_leaders(leaders)
    if not ranked:
        raise SystemExit(f"no evaluated checkpoint leaders found for goal {goal_id!r}")
    return ranked[0]


def format_checkpoint_filename(template: str, checkpoint_step: int | None) -> str:
    if checkpoint_step is None:
        return template.replace("_{checkpoint_step}_steps", "").format(checkpoint_step="unknown")
    return template.format(checkpoint_step=checkpoint_step)


def _format_bool(value: Any) -> str:
    return "enabled" if bool(value) else "disabled"


def _format_completion(completion_rate: float, metrics: Mapping[str, Any]) -> str:
    completion_count = metrics.get("completion_count")
    episodes = metrics.get("episodes")
    if completion_count is not None and episodes:
        return f"`{int(completion_count)}/{int(episodes)}` episodes"
    return f"{100.0 * completion_rate:.1f}%"


def _format_seed_start(metrics: Mapping[str, Any]) -> str:
    for key in ("seed_start", "eval_seed"):
        value = metrics.get(key)
        if value is not None:
            return str(value)
    episode_results = metrics.get("episode_results")
    if isinstance(episode_results, Sequence):
        seeds = [
            int(episode["seed"])
            for episode in episode_results
            if isinstance(episode, Mapping) and episode.get("seed") is not None
        ]
        if seeds:
            return str(min(seeds))
    return ""


def _format_episodes(metrics: Mapping[str, Any]) -> str:
    value = metrics.get("episodes")
    if value is not None:
        return str(value)
    episode_results = metrics.get("episode_results")
    if isinstance(episode_results, Sequence):
        return str(len(episode_results))
    return ""


def _format_eval_profile(leader, metrics: Mapping[str, Any]) -> str:
    profile = metrics.get("profile") or leader.eval_source or "checkpoint promotion"
    deterministic = metrics.get("deterministic")
    if deterministic is None:
        return str(profile)
    mode = "deterministic policy" if deterministic else "stochastic policy sampling"
    return f"`{profile}`, {mode}"


def _format_done_on(env_config: Mapping[str, Any]) -> str:
    done_on = env_config.get("done_on") or env_config.get("done_on_events")
    if isinstance(done_on, Sequence) and not isinstance(done_on, str):
        return ", ".join(str(item).replace("_", " ") for item in done_on)
    if env_config.get("terminate_on_life_loss") and env_config.get("terminate_on_completion"):
        return "life loss and completion"
    return str(done_on or "goal-specific termination")


def _format_env_id(env_provider: str, game: str) -> str:
    if env_provider and env_provider not in {"Stable Retro", "stable-retro"}:
        return f"{env_provider}:{game}"
    return game


def _format_reward_shaping(env_config: Mapping[str, Any], reward_mode: str) -> str:
    fields = []
    for key in (
        "reward_scale",
        "terminal_reward",
        "death_penalty",
        "completion_reward",
        "progress_reward_scale",
        "time_penalty",
    ):
        if key in env_config:
            fields.append(f"{key}=`{env_config[key]}`")
    if fields:
        return ", ".join(fields)
    return f"reward_mode=`{reward_mode}`"


def write_model_card(
    *,
    path: Path,
    goal: Mapping[str, Any],
    release: HuggingFaceReleaseConfig,
    leader,
    metrics: Mapping[str, Any],
    checkpoint_filename: str,
) -> None:
    env_config = goal["train"]["environment"]["env_config"]
    game = str(env_config.get("game", "SuperMarioBros-Nes-v0"))
    level = str(env_config.get("state", goal["goal_id"]))
    reward = float(leader.reward_mean)
    max_x = float(leader.max_x_max)
    obs_size = int(env_config.get("observation_size", 84))
    frame_stack = int(env_config.get("frame_stack", 4))
    frame_skip = int(env_config.get("frame_skip", 4))
    hud_crop_top = int(env_config.get("hud_crop_top", 32))
    action_set = str(env_config.get("action_set", "simple"))
    reward_mode = str(env_config.get("reward_mode", "score"))
    max_pool_frames = bool(env_config.get("max_pool_frames", False))
    max_episode_steps = env_config.get("max_episode_steps", "")
    env_provider = str(env_config.get("env_provider", "Stable Retro"))
    env_id = _format_env_id(env_provider, game)
    done_on = _format_done_on(env_config)
    reward_shaping = _format_reward_shaping(env_config, reward_mode)
    completion_text = _format_completion(float(leader.completion_rate), metrics)
    episodes_text = _format_episodes(metrics)
    seed_start_text = _format_seed_start(metrics)
    eval_profile = _format_eval_profile(leader, metrics)
    checkpoint_step = leader.checkpoint_step or ""
    model_name = release.repo.removeprefix("SuperMarioBros-NES_").replace("Level", "Level ")
    content = f"""---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - stable-baselines3
  - ppo
  - stable-retro
  - rlab
  - super-mario-bros
  - nes
  - {game}
metrics:
  - completion-rate
---

# SuperMarioBros-NES {model_name}

PPO policy checkpoint for completing `{game}` `{level}` with Stable Retro, trained with [`rlab`](https://github.com/tsilva/rlab).

## Quick Start

Install `rlab` once, import the ROM, then play or evaluate this checkpoint directly from Hugging Face:

```bash
uv tool install --from git+https://github.com/tsilva/rlab rlab
rlab import-roms ~/roms --game {game}
rlab play hf://{release.repo_id}
rlab eval hf://{release.repo_id}
```

## Evaluation Results

| `eval_profile` | `episodes` | `seed_start` | `completion_rate` | `max_x_max` | `reward_mean` | `checkpoint_step` |
|---|---:|---:|---:|---:|---:|---:|
| {eval_profile} | {episodes_text} | {seed_start_text} | {completion_text} | {max_x:.0f} | {reward:.3f} | {checkpoint_step} |

This is a checkpoint promotion metric from the current [`rlab`](https://github.com/tsilva/rlab) release process.

## Environment Details

| Setting | Value |
|---|---|
| `env_provider` | `{env_provider}` |
| `env_id` | `{env_id}` |
| `game` | `{game}` |
| `state` | `{level}` |
| `preprocessing` | crop top `{hud_crop_top}` px, grayscale, resize to `{obs_size} x {obs_size}` |
| `frame_stack` | `{frame_stack}` |
| `frame_skip` | `{frame_skip}` |
| `max_pool_frames` | {_format_bool(max_pool_frames)} |
| `policy_observation_layout` | channel-first `({frame_stack}, {obs_size}, {obs_size})` |
| `action_set` | `{action_set}` |
| `reward_mode` | `{reward_mode}` |
| `reward_shaping` | {reward_shaping} |
| `max_episode_steps` | `{max_episode_steps}` |
| `done_on_events` | {done_on} |

## Training Recipe

| Setting | Value |
|---|---:|
| `goal_id` | `{goal["goal_id"]}` |
| `spec_id` | `{leader.spec_slug}` |
| `checkpoint_step` | {checkpoint_step} |
| `frame_skip` | {frame_skip} |
| `max_episode_steps` | {max_episode_steps} |
| `reward_mode` | `{reward_mode}` |
| `reward_shaping` | {reward_shaping} |
| `done_on_events` | {done_on} |

## Provenance

| Item | Value |
|---|---|
| `source_project` | [`rlab`](https://github.com/tsilva/rlab) |
| `goal_id` | `{goal["goal_id"]}` |
| `goal_title` | `{goal.get("title", "")}` |
| `wandb_run` | [`{leader.run_name}`]({leader.url}) |
| `wandb_artifact` | `{leader.artifact_ref}` |
| `eval_source` | `{leader.eval_source or ""}` |
"""
    path.write_text(content, encoding="utf-8")


def copy_release_files(
    *,
    stage_dir: Path,
    model_path: Path,
    checkpoint_filename: str,
) -> tuple[Path, Path | None]:
    checkpoint_path = stage_dir / checkpoint_filename
    shutil.copy2(model_path, checkpoint_path)
    metadata_source = model_path.with_suffix(".metadata.json")
    metadata_path = None
    if metadata_source.is_file():
        metadata_path = stage_dir / "model_metadata.json"
        shutil.copy2(metadata_source, metadata_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None
        if isinstance(metadata, dict):
            metadata["filename"] = checkpoint_filename
            metadata_path.write_text(
                json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return checkpoint_path, metadata_path


def run_command(command: Sequence[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def upload_to_huggingface(
    *,
    repo_id: str,
    stage_dir: Path,
    private: bool,
    dry_run: bool,
) -> None:
    if shutil.which("hf") is None:
        raise SystemExit("hf CLI is required for Hugging Face upload")
    create_command = ["hf", "repo", "create", repo_id, "--type", "model", "--exist-ok"]
    create_command.append("--private" if private else "--public")
    run_command(create_command, dry_run=dry_run)
    run_command(
        [
            "hf",
            "upload",
            repo_id,
            str(stage_dir),
            ".",
            "--type",
            "model",
            "--commit-message",
            "Release rlab checkpoint",
        ],
        dry_run=dry_run,
    )
    if not dry_run:
        run_command(["hf", "models", "card", repo_id, "--text"], dry_run=False)


def upload_to_youtube(
    *,
    args: argparse.Namespace,
    stage_dir: Path,
    release: HuggingFaceReleaseConfig,
    goal: Mapping[str, Any],
    leader,
    video_path: Path,
) -> None:
    script = args.repo_root / "scripts" / "upload_youtube_video.py"
    if not script.is_file():
        raise SystemExit(f"YouTube uploader not found: {script}")
    env_config = goal["train"]["environment"]["env_config"]
    game = str(env_config.get("game", "SuperMarioBros-Nes-v0"))
    level = str(env_config.get("state", goal["goal_id"]))
    win_rate = f"{100.0 * float(leader.completion_rate):.0f}%"
    run_command(
        [
            "python3",
            str(script),
            str(video_path),
            "--title",
            f"{game}, {level}, PPO, {win_rate} win rate",
            "--human-description",
            f"PPO policy checkpoint completing {game} {level}, trained with `rlab`.",
            "--model-page",
            f"Model: https://huggingface.co/{release.repo_id}\nrlab: https://github.com/tsilva/rlab",
            "--playlist-title",
            "rlab",
            "--privacy-status",
            "public",
            "--output",
            str(stage_dir / "youtube_upload_result.json"),
        ],
        dry_run=args.dry_run,
    )


def write_manifest(
    *,
    path: Path,
    goal_path: Path,
    goal: Mapping[str, Any],
    release: HuggingFaceReleaseConfig,
    leader,
    metrics: Mapping[str, Any],
    checkpoint_path: Path,
    video_path: Path | None,
) -> None:
    manifest = {
        "goal_path": str(goal_path),
        "goal_id": goal["goal_id"],
        "huggingface": asdict(release) | {"repo_id": release.repo_id},
        "leader": asdict(leader),
        "checkpoint_file": str(checkpoint_path),
        "preview_file": str(video_path) if video_path else None,
        "metrics": dict(metrics),
    }
    path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_huggingface(args: argparse.Namespace) -> int:
    args.repo_root = args.repo_root.resolve()
    goal_path = resolve_goal_path(args.goal, args.repo_root)
    if not goal_path.is_file():
        raise SystemExit(f"goal contract not found: {goal_path}")
    goal = load_goal_contract(goal_path, args.repo_root)
    release = release_config_from_goal(goal, owner=args.hf_owner)
    leader = best_checkpoint_for_goal(str(goal["goal_id"]), args.project)
    checkpoint_step = leader.checkpoint_step
    checkpoint_filename = format_checkpoint_filename(release.checkpoint_filename, checkpoint_step)
    stage_dir = args.stage_root / release.repo
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"goal={goal['goal_id']}", flush=True)
    print(f"leader={leader.run_name}", flush=True)
    print(f"artifact={leader.artifact_ref}", flush=True)
    print(f"huggingface={release.repo_id}", flush=True)
    print(f"stage_dir={stage_dir}", flush=True)

    source = download_artifact_ref_source(leader.artifact_ref, args.artifact_root)
    checkpoint_path, _metadata_path = copy_release_files(
        stage_dir=stage_dir,
        model_path=source.model_path,
        checkpoint_filename=checkpoint_filename,
    )

    config = env_config_from_goal(goal)
    assert_rom_imported(config.game)
    from stable_baselines3 import PPO

    model = PPO.load(source.model_path, device=args.device)
    episodes = goal_eval_episodes(goal, args.episodes)
    seed = validate_eval_seed(args.seed)
    video_path = stage_dir / release.preview_filename
    metrics, written_video = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=episodes,
        seed=seed,
        max_steps=int(args.max_steps),
        deterministic=True,
        n_envs=1,
        capture_best_video=True,
        video_path=video_path,
        video_fps=float(args.video_fps),
        video_scale=int(args.video_scale),
        progress=bool(args.progress),
        progress_description=f"release preview {goal['goal_id']}",
        extra={
            "checkpoint_step": checkpoint_step,
            "checkpoint_artifact": leader.artifact_ref,
            "eval_seed": seed,
            "release_goal": goal["goal_id"],
        },
    )

    write_model_card(
        path=stage_dir / "README.md",
        goal=goal,
        release=release,
        leader=leader,
        metrics=metrics,
        checkpoint_filename=checkpoint_filename,
    )
    write_manifest(
        path=stage_dir / "release_manifest.json",
        goal_path=goal_path,
        goal=goal,
        release=release,
        leader=leader,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        video_path=written_video,
    )

    if args.no_hf_upload:
        print("hf_upload=skipped", flush=True)
    else:
        upload_to_huggingface(
            repo_id=release.repo_id,
            stage_dir=stage_dir,
            private=bool(args.private),
            dry_run=bool(args.dry_run),
        )

    if args.no_youtube or not release.include_youtube_preview:
        print("youtube_upload=skipped", flush=True)
    elif written_video is None:
        raise SystemExit("preview video was not written; refusing YouTube upload")
    else:
        upload_to_youtube(
            args=args,
            stage_dir=stage_dir,
            release=release,
            goal=goal,
            leader=leader,
            video_path=written_video,
        )

    print(f"model_url=https://huggingface.co/{release.repo_id}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
