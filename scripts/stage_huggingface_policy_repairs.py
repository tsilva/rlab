from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from rlab.config_validation import load_goal_contract
from rlab.env_identity import environment_hash, environment_identity_from_train_config
from rlab.policy_bundle import (
    build_model_document,
    build_recipe_document,
    canonical_json_sha256,
    load_policy_bundle,
    load_recipe_document,
    sha256_file,
    write_canonical_json,
)
from rlab.publication import (
    build_model_repo_id,
    publication_identity_from_model_metadata,
)
from rlab.training.sb3_ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from rlab.training.sb3_ppo import normalize_config as normalize_ppo_config
from rlab.training_backend import training_backend_config_hash
from rlab.wandb_artifacts import model_zip_from_download
from rlab.wandb_utils import load_wandb_env


PPO_CONFIG_KEYS = frozenset(PPO_DEFAULT_CONFIG)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _snapshot_root(cache_root: Path, repo_id: str, revision: str) -> Path:
    return (
        cache_root
        / ("models--" + repo_id.replace("/", "--"))
        / "snapshots"
        / revision
    )


def _runtime_source_commit(runtime_image_ref: str) -> str:
    image_ref = runtime_image_ref.removeprefix("docker:")
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image_ref,
            "--format",
            "{{json .Image}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    image = json.loads(result.stdout)
    commit = str(
        ((image.get("config") or {}).get("Labels") or {}).get(
            "org.opencontainers.image.revision"
        )
        or ""
    ).strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError(f"runtime image has no exact source revision: {runtime_image_ref}")
    return commit


def _task_contract(goal: Mapping[str, Any], done_on_events: list[str]) -> dict[str, Any]:
    train = goal.get("train")
    environment = train.get("environment") if isinstance(train, Mapping) else None
    task = environment.get("task") if isinstance(environment, Mapping) else None
    if not isinstance(task, Mapping):
        raise ValueError("goal has no canonical task contract")
    result = deepcopy(dict(task))
    result["events"] = {
        key: value
        for key, value in dict(result.get("events") or {}).items()
        if key in done_on_events
    }
    termination = dict(result.get("termination") or {})
    termination["success"] = [
        name for name in done_on_events if name == "level_change"
    ]
    termination["failure"] = [
        name for name in done_on_events if name != "level_change"
    ]
    result["termination"] = termination
    return result


def _portable_environment_config(
    legacy_env: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    env_threads = int(legacy_env.get("env_threads") or 1)
    result = {
        "env_provider": str(legacy_env["env_provider"]),
        "game": str(legacy_env["game"]),
        "env_args": {
            "frame_stack": 4,
            "info": "data",
            "info_filter": "all",
            "inttype": "stable",
            "noop_reset_max": 0,
            "num_threads": env_threads,
            "obs_copy": "safe_view",
            "obs_grayscale": True,
            "obs_layout": "chw",
            "obs_type": "image",
            "players": 1,
            "record": False,
            "render_mode": "rgb_array",
            "reward_clip": False,
            "rom_path": None,
            "scenario": "scenario",
            "use_restricted_actions": "filtered",
        },
        "task": deepcopy(dict(task)),
        "state": legacy_env.get("state"),
        "frame_skip": int(legacy_env.get("frame_skip") or 4),
        "sticky_action_prob": float(legacy_env.get("sticky_action_prob") or 0.0),
        "max_pool_frames": bool(legacy_env.get("max_pool_frames", False)),
        "observation_size": int(legacy_env.get("observation_size") or 84),
        "obs_crop": list(legacy_env.get("obs_crop") or [0, 0, 0, 0]),
        "obs_crop_mode": "remove",
        "obs_crop_fill": 0,
        "obs_resize_algorithm": str(legacy_env.get("obs_resize_algorithm") or "area"),
    }
    states = list(legacy_env.get("states") or [])
    if states:
        result["states"] = states
    state_probs = list(legacy_env.get("state_probs") or [])
    if state_probs:
        result["state_probs"] = state_probs
    return result


def _goal_contract(
    goal: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    episodes: int,
    checkpoint_freq: int,
) -> dict[str, Any]:
    result = deepcopy(dict(goal))
    # The checked-in goal tags are source templates. W&B already records their
    # rendered values, so the portable goal copy must not retain interpolation.
    result.pop("tags", None)
    train = dict(result.get("train") or {})
    train["environment"] = deepcopy(dict(environment))
    train["stop_on_acceptance"] = False
    train["checkpoint_eval_backend"] = "local"
    train["checkpoint_freq"] = checkpoint_freq
    result["train"] = train
    evaluation = dict(result.get("eval") or {})
    evaluation["environment"] = deepcopy(dict(environment))
    evaluation["episodes"] = episodes
    result["eval"] = evaluation
    return result


def _materialized_recipe(
    *,
    goal: Mapping[str, Any],
    config: Mapping[str, Any],
    legacy_metadata: Mapping[str, Any],
    source_commit: str,
    runtime_image_ref: str,
    recipe_path: str,
    evaluation_seed: int,
) -> dict[str, Any]:
    training = legacy_metadata["training_metadata"]
    legacy_env = training["env_config"]
    done_on_events = [str(item) for item in legacy_env.get("done_on_events") or []]
    task = _task_contract(goal, done_on_events)
    environment = _portable_environment_config(legacy_env, task=task)
    backend_raw = {key: config.get(key) for key in PPO_CONFIG_KEYS if key in config}
    backend = normalize_ppo_config(backend_raw, label="recovered sb3.ppo config")
    episodes = int(config.get("post_train_eval_episodes") or 100)
    eval_n_envs = int(
        config.get("post_train_eval_n_envs")
        or config.get("checkpoint_eval_n_envs")
        or 1
    )
    early_stop = deepcopy(config.get("early_stop"))
    if isinstance(early_stop, list):
        for rule in early_stop:
            if (
                isinstance(rule, dict)
                and rule.get("metric")
                == "train/info/level_complete/rate/min/last"
            ):
                # The metric registry renamed this same rolling completion signal.
                rule["metric"] = "train/outcome/success/window_100/rate/min"
    train_config = {
        "timesteps": int(config["timesteps"]),
        "training_backend": {"id": "sb3.ppo", "config": backend},
        "n_envs": int(config["n_envs"]),
        "seed": int(config["seed"]),
        **deepcopy(environment),
        "checkpoint_freq": int(config.get("checkpoint_freq") or 0),
        "post_train_eval_episodes": episodes,
        "checkpoint_eval_environment": deepcopy(environment),
        "checkpoint_eval_n_envs": eval_n_envs,
        "stop_on_acceptance": False,
        "checkpoint_eval_backend": "local",
        "checkpoint_eval_seed_protocol": "vector-lane-v1",
        "checkpoint_eval_seed": evaluation_seed,
        "post_train_eval_max_steps": int(config.get("post_train_eval_max_steps") or 0),
        "post_train_eval_stochastic": True,
        "early_stop": early_stop,
    }
    normalized_environment = environment_identity_from_train_config(train_config)
    materialized = {
        "schema_version": 2,
        "goal": _goal_contract(
            goal,
            environment=environment,
            episodes=episodes,
            checkpoint_freq=int(config.get("checkpoint_freq") or 0),
        ),
        "recipe_id": str(config.get("recipe_slug") or legacy_metadata.get("recipe_slug") or "base"),
        "description": str(config.get("run_description") or legacy_metadata.get("run_description") or "Recovered legacy policy recipe"),
        "tags": [str(item) for item in getattr(config.get("wandb_tags"), "split", lambda *_: [])(",") if str(item)],
        "train": {"backend": {"id": "sb3.ppo", "config": backend}},
        "train_config": train_config,
        "environment": normalized_environment,
        "environment_hash": environment_hash(normalized_environment),
        "_composition": {"source_files": []},
    }
    document = build_recipe_document(
        materialized,
        repo_root=Path.cwd(),
        source_commit=source_commit,
        run_description=str(materialized["description"]),
        seed=int(config["seed"]),
        runtime_image_ref=runtime_image_ref,
    )
    document["provenance"]["runtime"]["packages"] = {
        key.replace("_", "-"): str(value)
        for key, value in dict(training.get("versions") or {}).items()
        if value
    }
    return document


def _publication_metadata(
    *,
    legacy_metadata: Mapping[str, Any],
    artifact_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    recipe_document: Mapping[str, Any],
    source_commit: str,
    runtime_image_ref: str,
    run: Any,
) -> dict[str, Any]:
    result = deepcopy(dict(legacy_metadata))
    result.update(
        {
            "algorithm_id": "ppo",
            "model_class": "stable_baselines3.ppo.ppo.PPO",
            "training_backend_id": "sb3.ppo",
            "training_backend_config_hash": training_backend_config_hash(
                recipe_document["recipe"]["train_config"]
            ),
            "seed": int(config["seed"]),
            "repo_git_commit": source_commit,
            "runtime_image_ref": runtime_image_ref,
            "wandb_run_id": str(run.id),
            "wandb_project": str(run.project),
            "wandb_run_path": "/".join(str(item) for item in run.path),
        }
    )
    for key in ("training_metadata_hash", "queue_train_job_id", "kind"):
        value = artifact_metadata.get(key)
        if value not in (None, ""):
            result[key] = value
    training = dict(result.get("training_metadata") or {})
    environment = deepcopy(recipe_document["recipe"]["environment"])
    training["environment"] = environment
    training["environment_hash"] = recipe_document["recipe"]["environment_hash"]
    training["preprocessing"] = deepcopy(environment["preprocessing"])
    result["training_metadata"] = training
    result["training_metadata_hash"] = canonical_json_sha256(training)
    return result


def stage_foundations(
    *,
    audit_path: Path,
    cache_root: Path,
    output_root: Path,
    wandb_root: Path,
) -> list[dict[str, Any]]:
    load_wandb_env()
    import wandb

    audit = _load_object(audit_path)
    wandb_api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for row in audit["repositories"]:
        repo_id = str(row["repo_id"])
        revision = str(row["remote_parent_commit"])
        snapshot = _snapshot_root(cache_root, repo_id, revision)
        legacy_metadata = _load_object(snapshot / "model_metadata.json")
        legacy_manifest = _load_object(snapshot / "release_manifest.json")
        artifact_ref = str(legacy_manifest["leader"]["artifact_ref"])
        artifact = wandb_api.artifact(artifact_ref)
        run = artifact.logged_by()
        if run is None:
            raise ValueError(f"checkpoint artifact has no logging run: {artifact_ref}")
        config = dict(run.config)
        artifact_metadata = dict(artifact.metadata or {})
        runtime_image_ref = str(
            config.get("runtime_image_ref")
            or artifact_metadata.get("runtime_image_ref")
            or legacy_metadata.get("runtime_image_ref")
            or ""
        )
        source_commit = _runtime_source_commit(runtime_image_ref)
        goal_id = str(legacy_manifest["goal_id"])
        goal = load_goal_contract(
            Path("experiments/goals/SuperMarioBros-Nes-v0") / goal_id / "_goal.yaml"
        )
        evaluation_seed = int(legacy_manifest.get("metrics", {}).get("eval_seed") or 10_000)
        recipe_document = _materialized_recipe(
            goal=goal,
            config=config,
            legacy_metadata=legacy_metadata,
            source_commit=source_commit,
            runtime_image_ref=runtime_image_ref,
            recipe_path=str(config.get("recipe_path") or artifact_metadata.get("recipe_path") or ""),
            evaluation_seed=evaluation_seed,
        )
        publication_metadata = _publication_metadata(
            legacy_metadata=legacy_metadata,
            artifact_metadata=artifact_metadata,
            config=config,
            recipe_document=recipe_document,
            source_commit=source_commit,
            runtime_image_ref=runtime_image_ref,
            run=run,
        )
        identity = publication_identity_from_model_metadata(goal_id, publication_metadata)
        if build_model_repo_id(identity) != repo_id:
            raise ValueError(f"recovered identity disagrees with repository: {repo_id}")

        artifact_root = wandb_root / repo_id.replace("/", "__")
        downloaded = Path(artifact.download(root=str(artifact_root)))
        artifact_model = model_zip_from_download(
            downloaded,
            expected_filename=str(artifact_metadata.get("filename") or "") or None,
        )
        checkpoint = snapshot / "model.zip"
        if sha256_file(artifact_model) != sha256_file(checkpoint):
            raise ValueError(f"W&B checkpoint bytes disagree with Hugging Face: {repo_id}")

        root = output_root / repo_id.replace("/", "__")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        shutil.copy2(checkpoint, root / "model.zip")
        write_canonical_json(root / "recipe.json", recipe_document)
        load_recipe_document(root / "recipe.json")
        write_canonical_json(
            root / "model.json",
            build_model_document(
                root / "model.zip", root / "recipe.json", publication_metadata
            ),
        )
        load_policy_bundle(root, source=repo_id, revision=revision)
        write_canonical_json(root / "publication_metadata.json", publication_metadata)
        rows.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "goal_id": goal_id,
                "artifact_ref": artifact_ref,
                "wandb_run_path": "/".join(str(item) for item in run.path),
                "runtime_image_ref": runtime_image_ref,
                "source_commit": source_commit,
                "checkpoint_sha256": sha256_file(root / "model.zip"),
                "recipe_sha256": sha256_file(root / "recipe.json"),
                "foundation": str(root),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("runs/hf_repair/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/hf_repair/foundations"))
    parser.add_argument("--wandb-root", type=Path, default=Path("runs/hf_repair/wandb"))
    parser.add_argument("--inventory", type=Path, default=Path("runs/hf_repair/foundations.json"))
    args = parser.parse_args()
    rows = stage_foundations(
        audit_path=args.audit,
        cache_root=args.cache_root,
        output_root=args.output_root,
        wandb_root=args.wandb_root,
    )
    payload = {
        "repositories": rows,
        "digest": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    write_canonical_json(args.inventory, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
