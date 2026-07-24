from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rlab.dstack_backend import (
    TERMINAL_DSTACK_STATUSES,
    ComputeRequest,
    DstackBackend,
    TaskRequest,
)
from rlab.env_registry import resolve_env_provider
from rlab.file_utils import file_sha256
from rlab.json_utils import json_safe
from rlab.modal_eval_config import load_modal_eval_config
from rlab.r2_store import RunStorageConfig
from rlab.policy_bundle import build_recipe_document, canonical_json_sha256
from rlab.recipe_documents import (
    compose_train_document,
    prepare_checkpoint_eval_mode,
)
from rlab.rom_assets import (
    ROM_ASSET_IDENTITY_ALGORITHM,
    ROM_ASSET_PREFIX,
    ROM_ASSET_SCHEMA_VERSION,
    discover_rom_path,
    provider_rom_identity,
    validate_rom_asset_manifest,
)
from rlab.run_authority import RunAuthority
from rlab.run_contracts import (
    RUN_ID_PATTERN,
    RunManifest,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_WORKFLOW,
    DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    clean_git_source_sha,
    current_git_branch,
    runtime_release_from_args,
)
from rlab.wandb_utils import (
    canonical_wandb_environment,
    wandb_entity_from_env,
)


DEFAULT_MAX_DURATION_SECONDS = 48 * 60 * 60
DEFAULT_ROM_MOUNT = "/var/lib/rlab/rom-cache:/rom-cache"
QUIESCENCE_SECONDS = 30.0
COMMON_SECRET_ENV = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "RLAB_CONTROL_R2_URI",
    "RLAB_CONTROL_R2_ENDPOINT_URL",
    "RLAB_CONTROL_R2_REGION",
    "RLAB_CONTROL_R2_ACCESS_KEY_ID",
    "RLAB_CONTROL_R2_SECRET_ACCESS_KEY",
    "RLAB_EVAL_R2_URI",
    "RLAB_EVAL_R2_ENDPOINT_URL",
    "RLAB_EVAL_R2_REGION",
    "RLAB_EVAL_R2_ACCESS_KEY_ID",
    "RLAB_EVAL_R2_SECRET_ACCESS_KEY",
    "RLAB_MODELS_R2_URI",
    "RLAB_MODELS_R2_ENDPOINT_URL",
    "RLAB_MODELS_R2_REGION",
    "RLAB_MODELS_R2_ACCESS_KEY_ID",
    "RLAB_MODELS_R2_SECRET_ACCESS_KEY",
    "RLAB_MODELS_R2_PUBLIC_BASE_URL",
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def repository_root() -> Path:
    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def _load_environment(root: Path) -> None:
    from rlab.dotenv import load_env_file

    load_env_file(root / ".env")


def _tracked_committed_path(root: Path, path: Path, *, label: str) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if tracked.returncode:
        raise ValueError(f"{label} must be checked in: {relative}")
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=root,
        check=False,
    )
    if changed.returncode:
        raise ValueError(f"{label} has uncommitted changes: {relative}")
    return resolved


def _parse_duration(value: str | int | float) -> float:
    if isinstance(value, int | float):
        result = float(value)
    else:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value)
        if match is None:
            raise argparse.ArgumentTypeError("duration must look like 30s, 10m, 2h, or 1d")
        scale = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        result = float(match.group(1)) * scale
    if result <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return result


def _require_run_id(value: str) -> str:
    text = str(value).strip()
    if RUN_ID_PATTERN.fullmatch(text) is None:
        raise argparse.ArgumentTypeError("run id must match rlab-<32 lowercase hex>")
    return text


def _storage(root: Path) -> tuple[RunStorageConfig, RunAuthority]:
    _load_environment(root)
    storage = RunStorageConfig.from_env()
    return storage, RunAuthority(storage)


def _stage_rom(
    authority: RunAuthority,
    *,
    env_provider: str,
    game: str,
    rom_path: Path | None,
) -> dict[str, Any] | None:
    if not resolve_env_provider(env_provider).requires_external_rom_asset:
        if rom_path is not None:
            raise ValueError(
                f"--rom-path is invalid for ROM-free provider {env_provider!r}"
            )
        return None
    source = discover_rom_path(game, rom_path=rom_path)
    digest = file_sha256(source)
    key = f"{ROM_ASSET_PREFIX}/objects/sha256/{digest}/{source.name}"
    manifest = validate_rom_asset_manifest(
        {
            "schema_version": ROM_ASSET_SCHEMA_VERSION,
            "game": game,
            "filename": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": digest,
            "object_uri": authority.evaluation.uri(key),
            "provider_rom_identity": provider_rom_identity(source),
            "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
        },
        expected_game=game,
    )
    authority.evaluation.put_file(
        key,
        source,
        sha256=digest,
        content_type="application/octet-stream",
    )
    authority.evaluation.put_json(
        f"{ROM_ASSET_PREFIX}/manifests/{game}/{digest}.json",
        manifest,
        create_only=True,
    )
    return manifest


def _compute(args: argparse.Namespace) -> ComputeRequest:
    request = ComputeRequest(
        kind=args.compute,
        target=args.target,
        max_price=args.max_price,
        max_cost_usd=args.max_cost_usd,
        allow_on_demand=bool(args.allow_on_demand),
        max_duration_seconds=int(args.max_duration),
    )
    request.validate()
    return request


def _task_name(run_id: str, attempt_id: str, *, initial: bool) -> str:
    return run_id if initial else f"{run_id}-a{attempt_id.removeprefix('attempt-')}"


def _task_request(manifest: RunManifest, *, manifest_uri: str) -> TaskRequest:
    compute = ComputeRequest(
        **dict(manifest.compute.get("selected") or manifest.compute["request"])
    )
    local = compute.kind == "local"
    return TaskRequest(
        run_id=manifest.run_id,
        task_name=str(manifest.compute["dstack_task"]),
        image=manifest.image_digest,
        manifest_uri=manifest_uri,
        compute=compute,
        secret_env=(
            *COMMON_SECRET_ENV,
            *(
                (
                    "MODAL_TOKEN_ID",
                    "MODAL_TOKEN_SECRET",
                    f"MODAL_ENVIRONMENT={manifest.modal['environment_name']}",
                )
                if bool(manifest.modal["enabled"])
                else ()
            ),
        ),
        rom_mount=(
            DEFAULT_ROM_MOUNT
            if local and isinstance(manifest.modal.get("rom_asset_manifest"), dict)
            else None
        ),
    )


def _wandb_identity(document: dict[str, Any], run_id: str) -> dict[str, Any]:
    config = dict(document["train_config"])
    project, family = canonical_wandb_environment(
        config.get("env_provider"),
        config.get("game"),
    )
    entity = wandb_entity_from_env()
    return {
        "run_id": run_id,
        "entity": entity,
        "project": project,
        "group": run_id,
        "game_family": family,
        "url": (
            f"https://wandb.ai/{quote(entity, safe='')}/"
            f"{quote(project, safe='')}/runs/{quote(run_id, safe='')}"
        ),
    }


def cmd_launch(args: argparse.Namespace) -> int:
    root = repository_root()
    _load_environment(root)
    source_sha = clean_git_source_sha(root)
    branch = current_git_branch(root)
    goal_path = _tracked_committed_path(root, args.goal_file, label="goal")
    recipe_path = _tracked_committed_path(root, args.recipe_file, label="recipe")
    recipe_overrides = tuple(str(value) for value in args.recipe_overrides)
    checkpoint_eval_backend = str(args.checkpoint_eval_backend)
    document = compose_train_document(
        goal_path,
        recipe_path,
        recipe_overrides=recipe_overrides,
        prepare_materialized=partial(
            prepare_checkpoint_eval_mode,
            checkpoint_eval_backend=checkpoint_eval_backend,
        ),
    )
    release = runtime_release_from_args(
        args,
        repo_root=root,
        checkpoint_eval_backend=checkpoint_eval_backend,
        wait_for_modal=checkpoint_eval_backend == "modal",
    )
    if release.source_sha != source_sha:
        raise RuntimeError("runtime release source does not match committed HEAD")
    compute = _compute(args)
    dstack_backend = DstackBackend()
    selected_compute, selected_offer = dstack_backend.select_compute(compute)
    storage, authority = _storage(root)
    config = dict(document["train_config"])
    asset = _stage_rom(
        authority,
        env_provider=str(config["env_provider"]),
        game=str(config["game"]),
        rom_path=args.rom_path,
    )
    run_id = new_run_id()
    attempt_id = new_attempt_id()
    dstack_task = _task_name(run_id, attempt_id, initial=True)
    wandb = _wandb_identity(document, run_id)
    modal_app = str(release.modal_app_name or "").strip()
    if checkpoint_eval_backend == "modal" and not modal_app:
        raise RuntimeError("exact-source runtime has no immutable Modal deployment receipt")
    modal_config = load_modal_eval_config(root / "experiments" / "modal_eval.yaml")
    contract_document = dict(document)
    contract_config = dict(contract_document["train_config"])
    contract_config["rom_asset_manifest"] = asset
    contract_config["checkpoint_eval_backend"] = checkpoint_eval_backend
    contract_document["train_config"] = contract_config
    portable_recipe = build_recipe_document(
        contract_document,
        repo_root=root,
        source_commit=source_sha,
        run_description=str(args.run_description),
        seed=int(args.seed),
        runtime_image_ref=release.runtime_image_ref,
    )
    manifest = RunManifest(
        run_id=run_id,
        attempt_id=attempt_id,
        created_at=utc_now(),
        source_sha=source_sha,
        image_digest=release.runtime_image_ref,
        goal_slug=goal_path.parent.relative_to(root / "experiments" / "goals").as_posix(),
        goal_sha256=str(
            document["train_config"]["effective_goal_contract_sha256"]
        ),
        recipe_slug=recipe_path.stem,
        recipe_sha256=canonical_json_sha256(portable_recipe),
        recipe_overrides=recipe_overrides,
        environment_sha256=str(document["environment_hash"]).removeprefix("sha256:"),
        seed=int(args.seed),
        run_description=str(args.run_description),
        compute={
            "request": compute.as_manifest(),
            "selected": selected_compute.as_manifest(),
            "selected_offer": selected_offer,
            "dstack_task": dstack_task,
            "source_branch": branch,
            "runtime_workflow_run_id": release.workflow_run_id,
            "runtime_input_sha256": release.runtime_input_sha256,
            "submission_key": str(args.submission_key or ""),
        },
        wandb=wandb,
        modal={
            "enabled": checkpoint_eval_backend == "modal",
            "environment_name": modal_config.environment_name,
            "app_name": modal_app,
            "function_name": modal_config.function_name,
            "deployment_source_sha": source_sha,
            "rom_asset_manifest": asset,
        },
        storage=storage.manifest_locations(),
    )
    authority.create_manifest(manifest)
    manifest_uri = authority.control.uri(f"runs/{run_id}/manifest.json")
    task = dstack_backend.submit(_task_request(manifest, manifest_uri=manifest_uri))
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "dstack": {"project": task.project, "task": task.name, "status": task.status},
        "compute": {
            "request": compute.as_manifest(),
            "selected": selected_compute.as_manifest(),
            "offer": selected_offer,
        },
        "source_sha": source_sha,
        "image_digest": release.runtime_image_ref,
        "runtime_input_sha256": release.runtime_input_sha256,
        "runtime_build_source_sha": release.source_sha,
        "goal_file": goal_path.relative_to(root).as_posix(),
        "recipe_file": recipe_path.relative_to(root).as_posix(),
        "goal_sha256": manifest.goal_sha256,
        "recipe_sha256": manifest.recipe_sha256,
        "recipe_overrides": list(recipe_overrides),
        "seed": int(args.seed),
        "run_description": str(args.run_description),
        "submission_key": str(args.submission_key or ""),
        "checkpoint_eval_backend": checkpoint_eval_backend,
        "wandb_url": wandb["url"],
        "public_run_index_url": authority.models.public_url(f"runs/{run_id}/index.json"),
    }
    print(
        json.dumps(json_safe(output), sort_keys=True)
        if args.json
        else (
            f"run={run_id} task={task.name} compute={selected_compute.kind} "
            f"image={release.runtime_image_ref} wandb={wandb['url']} "
            f"index={output['public_run_index_url']}"
        )
    )
    return 0


def _latest_attempt(state: dict[str, Any]) -> dict[str, Any]:
    attempts = list(state.get("attempts") or [])
    if attempts:
        return dict(attempts[-1])
    manifest = state.get("manifest")
    if isinstance(manifest, dict):
        return dict(manifest)
    raise KeyError(f"run not found: {state['run_id']}")


def _latest_attempt_terminal(state: dict[str, Any]) -> dict[str, Any] | None:
    terminals = list(state.get("attempt_terminals") or [])
    return dict(terminals[-1]) if terminals else None


def _status(root: Path, run_id: str) -> dict[str, Any]:
    _storage_config, authority = _storage(root)
    semantic = authority.semantic_state(run_id)
    attempt = _latest_attempt(semantic)
    task_name = str(attempt["compute"]["dstack_task"])
    try:
        dstack = DstackBackend().status(task_name)
        dstack_value: dict[str, Any] = {
            "project": dstack.project,
            "task": dstack.name,
            "status": dstack.status,
            "terminal": dstack.terminal,
            "raw": dstack.raw,
        }
    except KeyError:
        dstack_value = {
            "project": DstackBackend().project,
            "task": task_name,
            "status": "not-found",
            "terminal": False,
        }
    attempt_terminal = _latest_attempt_terminal(semantic)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt["attempt_id"],
        "dstack": dstack_value,
        "semantic": semantic,
        "attempt_terminal": attempt_terminal,
        "completed": attempt_terminal is not None,
        "scientific_success": semantic.get("terminal") is not None,
    }


def cmd_status(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            json_safe(_status(repository_root(), args.run_id)),
            indent=None if args.json else 2,
            sort_keys=True,
        )
    )
    return 0


def cmd_follow(args: argparse.Namespace) -> int:
    root = repository_root()
    deadline = time.monotonic() + float(args.timeout)
    previous = ""
    while True:
        value = _status(root, args.run_id)
        encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
        if encoded != previous:
            print(encoded, flush=True)
            previous = encoded
        if value["completed"]:
            return 0
        if time.monotonic() >= deadline:
            return 1
        time.sleep(float(args.poll_seconds))


def cmd_wait(args: argparse.Namespace) -> int:
    root = repository_root()
    deadline = time.monotonic() + float(args.timeout)
    while True:
        value = _status(root, args.run_id)
        reached = (
            value["completed"]
            if args.until == "terminal"
            else str(value["dstack"]["status"]).lower() in {"running", "pulling"}
        )
        if reached:
            print(json.dumps(json_safe(value), sort_keys=True))
            return 0
        if time.monotonic() >= deadline:
            print(json.dumps(json_safe(value), sort_keys=True))
            return 1
        time.sleep(2.0)


def cmd_cancel(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, authority = _storage(root)
    state = authority.semantic_state(args.run_id)
    attempt = _latest_attempt(state)
    task_name = str(attempt["compute"]["dstack_task"])
    DstackBackend().cancel(task_name, abort=bool(args.abort))
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "attempt_id": attempt["attempt_id"],
                "dstack_task": task_name,
                "cancel_requested": True,
                "abort": bool(args.abort),
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    root = repository_root()
    _storage_config, authority = _storage(root)
    attempt = _latest_attempt(authority.semantic_state(args.run_id))
    text = DstackBackend().logs(str(attempt["compute"]["dstack_task"]), since=args.since)
    lines = text.splitlines()
    if args.tail > 0:
        lines = lines[-args.tail :]
    print("\n".join(lines))
    return 0


def _lease_expiry(authority: RunAuthority, run_id: str) -> datetime | None:
    value = authority.control.get_json_optional(f"runs/{run_id}/writer-lease.json")
    if value is None:
        return None
    return datetime.fromisoformat(str(value["expires_at"]).replace("Z", "+00:00")).astimezone(UTC)


def cmd_retry(args: argparse.Namespace) -> int:
    root = repository_root()
    storage, authority = _storage(root)
    state = authority.semantic_state(args.run_id)
    if state.get("terminal") is not None:
        raise RuntimeError("a scientifically successful run must not be retried")
    attempt_terminal = _latest_attempt_terminal(state)
    if (
        attempt_terminal is not None
        and str(attempt_terminal.get("state") or "") == "succeeded"
    ):
        raise RuntimeError("a successfully drained training-only run must not be retried")
    previous = _latest_attempt(state)
    dstack_backend = DstackBackend()
    previous_task = dstack_backend.status(str(previous["compute"]["dstack_task"]))
    if previous_task.status.lower().replace("_", "-") not in TERMINAL_DSTACK_STATUSES:
        raise RuntimeError("the previous dstack attempt must be terminal before retry")
    expiry = _lease_expiry(authority, args.run_id)
    if expiry is not None and expiry > datetime.now(UTC):
        raise RuntimeError(f"the previous writer lease has not expired: {expiry.isoformat()}")
    time.sleep(QUIESCENCE_SECONDS)
    attempt_id = new_attempt_id()
    task_name = _task_name(args.run_id, attempt_id, initial=False)
    compute = dict(previous["compute"])
    compute["dstack_task"] = task_name
    selected_compute, selected_offer = dstack_backend.select_compute(
        ComputeRequest(**dict(compute["request"]))
    )
    compute["selected"] = selected_compute.as_manifest()
    compute["selected_offer"] = selected_offer
    public_checkpoints = list((state.get("public_index") or {}).get("checkpoints") or [])
    learner_finished = any(
        str(row.get("purpose") or "") == "final"
        for row in public_checkpoints
        if isinstance(row, dict)
    )
    if learner_finished or authority.has_accepted_eval(args.run_id):
        compute["recovery_mode"] = "drain-only"
    else:
        compute["recovery_mode"] = "resume-training"
    manifest = replace(
        RunManifest(**previous),
        attempt_id=attempt_id,
        created_at=utc_now(),
        compute=compute,
    )
    manifest.validate()
    authority.create_attempt_manifest(manifest)
    manifest_key = f"runs/{args.run_id}/attempts/{attempt_id}/manifest.json"
    manifest_uri = authority.control.uri(manifest_key)
    task = dstack_backend.submit(_task_request(manifest, manifest_uri=manifest_uri))
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "attempt_id": attempt_id,
                "retried_from_attempt_id": previous["attempt_id"],
                "dstack_task": task.name,
                "recovery_mode": compute.get("recovery_mode", "resume-training"),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab experiment",
        description="Launch and observe dstack-backed training experiments.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser(
        "launch",
        help="Launch a checked-in goal and recipe with an exact-source runtime image.",
        description=(
            "Launch one checked-in goal and recipe through dstack. The command requires "
            "a clean committed source revision and resolves its exact-source immutable "
            "runtime image before scheduling compute; it never falls back to an older image."
        ),
    )
    launch.add_argument("--goal-file", type=Path, required=True)
    launch.add_argument("--recipe-file", type=Path, required=True)
    launch.add_argument("--seed", type=int, required=True)
    launch.add_argument("--run-description", required=True)
    launch.add_argument(
        "--set",
        dest="recipe_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hash-bound Hydra recipe override; repeat for independent keys.",
    )
    launch.add_argument(
        "--checkpoint-eval-backend",
        choices=("modal", "none"),
        default="modal",
        help=(
            "Use Modal acceptance evaluation, or publish training-only evidence "
            "without promotion or goal acceptance."
        ),
    )
    launch.add_argument(
        "--submission-key",
        help="Optional research-wave identity recorded in launch output.",
    )
    launch.add_argument(
        "--compute",
        choices=("auto", "local", "spot", "on-demand"),
        default="auto",
    )
    launch.add_argument("--target")
    launch.add_argument("--max-price", type=float)
    launch.add_argument("--max-cost-usd", type=float)
    launch.add_argument("--allow-on-demand", action="store_true")
    launch.add_argument(
        "--max-duration",
        type=_parse_duration,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    launch.add_argument("--rom-path", type=Path)
    launch.add_argument("--runtime-image-ref-file", type=Path)
    launch.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    launch.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    launch.add_argument("--image-branch")
    launch.add_argument("--existing-runtime-only", action="store_true")
    launch.add_argument(
        "--runtime-readiness-timeout",
        type=_parse_duration,
        default=DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    )
    launch.add_argument("--json", action="store_true")
    launch.set_defaults(func=cmd_launch)

    status = commands.add_parser("status", help="Inspect dstack and R2 run state.")
    status.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    follow = commands.add_parser("follow", help="Stream changes in semantic run state.")
    follow.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    follow.add_argument("--poll-seconds", type=float, default=2.0)
    follow.add_argument("--timeout", type=_parse_duration, default=12 * 60 * 60)
    follow.set_defaults(func=cmd_follow)

    wait = commands.add_parser("wait", help="Wait for one run state.")
    wait.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    wait.add_argument("--until", choices=("running", "terminal"), required=True)
    wait.add_argument("--timeout", type=_parse_duration, default=12 * 60 * 60)
    wait.set_defaults(func=cmd_wait)

    cancel = commands.add_parser("cancel", help="Cancel the current dstack attempt.")
    cancel.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    cancel.add_argument("--abort", action="store_true")
    cancel.set_defaults(func=cmd_cancel)

    retry = commands.add_parser("retry", help="Retry a terminal failed attempt.")
    retry.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    retry.set_defaults(func=cmd_retry)

    logs = commands.add_parser("logs", help="Read dstack logs for the current attempt.")
    logs.add_argument("--run", dest="run_id", type=_require_run_id, required=True)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--since")
    logs.set_defaults(func=cmd_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
