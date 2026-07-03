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
from rlab.env_config_aliases import normalize_provider_env_config_aliases
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
                "ppo_supermariobros-nes-v0_{checkpoint_step}_steps.zip",
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
    normalized = normalize_provider_env_config_aliases(
        merged,
        label="goal.release.eval.environment.env_config",
    )
    fields = set(EnvConfig.__dataclass_fields__)
    config_kwargs = {key: value for key, value in normalized.items() if key in fields}
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
    completion = 100.0 * float(leader.completion_rate)
    reward = float(leader.reward_mean)
    max_x = float(leader.max_x_max)
    eval_episodes = metrics.get("episodes", metrics.get("eval/episodes", ""))
    content = f"""---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - stable-baselines3
  - stable-retro
  - rlab
  - {game}
metrics:
  - reward
---

# {game} {level} PPO

PPO policy checkpoint for `{game}` `{level}`, trained with `rlab`.

## At a Glance

| Item | Value |
|---|---|
| Task | Complete `{game}` `{level}` |
| Model | Stable-Baselines3 PPO |
| Format | SB3 `.zip` checkpoint |
| Checkpoint | `{checkpoint_filename}` |
| Completion rate | {completion:.1f}% |
| Mean reward | {reward:.3f} |
| Max x-position | {max_x:.0f} |
| W&B run | [{leader.run_name}]({leader.url}) |
| W&B artifact | `{leader.artifact_ref}` |

## Quick Start

```bash
hf download {release.repo_id} {checkpoint_filename} --local-dir .
rlab play {checkpoint_filename}
```

For the original W&B artifact:

```bash
rlab play {leader.artifact_ref} --policy-env fast
```

## Validate

This release is selected from the goal checkpoint leaderboard:

```bash
rlab leaders checkpoints --goal {goal["goal_id"]} --limit 1 --json
```

Release staging re-evaluated the checkpoint for preview generation with `{eval_episodes}`
episode(s). See `release_manifest.json` for exact metrics and provenance.

## Results

| Metric | Value |
|---|---:|
| Completion rate (leaderboard) | {completion:.1f}% |
| Completion rate mean (leaderboard) | {100.0 * float(leader.completion_rate_mean):.1f}% |
| Mean reward (leaderboard) | {reward:.3f} |
| Max x-position (leaderboard) | {max_x:.0f} |
| Checkpoint step | {leader.checkpoint_step or ""} |

## Files

| File | Description |
|---|---|
| `{checkpoint_filename}` | SB3 PPO checkpoint |
| `{release.preview_filename}` | Representative preview episode |
| `model_metadata.json` | Downloaded W&B artifact metadata when available |
| `release_manifest.json` | Release provenance and verification inputs |

## Provenance

- Source project: `rlab`
- Goal: `{goal["goal_id"]}`
- Goal title: `{goal.get("title", "")}`
- W&B run: `{leader.run_name}`
- W&B artifact: `{leader.artifact_ref}`
- Eval source: `{leader.eval_source or ""}`

## Limitations

This is a single selected checkpoint for a specific Stable Retro task. Reported leaderboard
metrics come from the current `rlab` checkpoint promotion contract and should not be treated
as cross-environment benchmark results.
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
