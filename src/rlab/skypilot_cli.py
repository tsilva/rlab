from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from rlab.campaign import connect, database_url, enqueue_train_job, goal_id_from_slug, load_json_arg
from rlab.compute_targets import canonical_target_name
from rlab.runtime_refs import normalize_runtime_image_ref, runtime_image_digest_slug
from rlab.skypilot_launch import (
    build_launch_command,
    build_runner_launch_command,
    cleanup_command,
    collect_results,
    ensure_skypilot_api,
    execute_launch,
    fetch_wandb_run_config,
    format_results_table,
    instance_defaults,
    instance_label,
    launch_infra,
    LaunchSummary,
    launch_summary,
    load_instance_config,
    load_manifest,
    load_runner_profile,
    manifest_from_wandb_config,
    merged_env,
    preflight_checks,
    preflight_runner_profile,
    render_runner_task_yaml,
    render_task_yaml,
    runner_cluster_name,
    sanitize_slug,
    shell_join,
    target_name,
    write_launch_report,
    write_rendered_task,
    write_rendered_runner_task,
)


def repo_root_from_args(args: argparse.Namespace) -> Path:
    return Path(args.repo_root).expanduser().resolve()


def add_common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path, help="Experiment manifest JSON file")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative paths; defaults to the current directory.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        help="Machine-readable instance config; defaults to experiments/instances.json.",
    )
    parser.add_argument(
        "--target",
        help="Named compute target from experiments/instances.json; overrides manifest target.",
    )


def add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile", type=Path, help="Runner profile JSON file")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative paths; defaults to the current directory.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        help="Machine-readable instance config; defaults to experiments/instances.json.",
    )
    parser.add_argument(
        "--target",
        help="Named compute target from experiments/instances.json; overrides profile target.",
    )


def cmd_render(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    manifest = load_manifest(args.manifest)
    instance_config = load_instance_config(repo_root, args.instances)
    if args.output:
        path = write_rendered_task(
            manifest,
            instance_config,
            repo_root,
            args.output,
            target_override=args.target,
        )
        print(path)
    else:
        print(render_task_yaml(manifest, instance_config, repo_root, target_override=args.target), end="")
    return 0


def cmd_render_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    if args.output:
        path = write_rendered_runner_task(
            profile,
            instance_config,
            repo_root,
            args.output,
            target_override=args.target,
        )
        print(path)
    else:
        print(
            render_runner_task_yaml(profile, instance_config, repo_root, target_override=args.target),
            end="",
        )
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    manifest = load_manifest(args.manifest)
    instance_config = load_instance_config(repo_root, args.instances)
    checks = preflight_checks(
        manifest,
        instance_config,
        repo_root,
        env=merged_env(repo_root / ".env"),
        target_override=args.target,
    )
    for check in checks:
        print(f"{check.level}: {check.message}")
    return 1 if any(check.level == "error" for check in checks) else 0


def cmd_preflight_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    checks = preflight_runner_profile(
        profile,
        instance_config,
        repo_root,
        env=merged_env(repo_root / ".env"),
        target_override=args.target,
    )
    for check in checks:
        print(f"{check.level}: {check.message}")
    return 1 if any(check.level == "error" for check in checks) else 0


def cmd_launch(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    summary = launch_summary(
        args.manifest,
        args.output,
        repo_root,
        args.instances,
        target_override=args.target,
        detach_run=args.detach_run,
    )
    print(f"task: {summary.task_path}")
    print(f"cluster: {summary.cluster}")
    print(f"wandb_group_prefix: {summary.wandb_group_prefix}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def cmd_launch_runner(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    target = target_name(profile, args.target)
    instance = instance_defaults(instance_config, target)
    canonical_target = canonical_target_name(instance_config, target)
    task_path = write_rendered_runner_task(
        profile,
        instance_config,
        repo_root,
        args.output,
        target_override=target,
    )
    image_id = str(profile.get("image_id", instance.get("image_id", ""))).strip()
    prebuilt_image = bool(profile.get("prebuilt_image", instance.get("prebuilt_image", False)))
    runtime_image_ref = normalize_runtime_image_ref(
        str(profile.get("runtime_image_ref", image_id if prebuilt_image else "")).strip()
    )
    cluster = str(
        args.cluster
        or runner_cluster_name(
            profile_id=str(profile["profile_id"]),
            target=canonical_target,
            runtime_image_ref=runtime_image_ref,
        )
    )
    env = merged_env(repo_root / ".env")
    command = build_runner_launch_command(
        cluster,
        task_path,
        env=env,
        infra=launch_infra(instance),
        detach_run=args.detach_run,
    )
    summary = LaunchSummary(
        command=command,
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(profile["profile_id"]),
    )
    print(f"task: {summary.task_path}")
    print(f"cluster: {summary.cluster}")
    print(f"profile: {profile['profile_id']}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def default_runner_task_path(canonical_target: str, runtime_image_ref: str) -> Path:
    return Path(
        f"sky_train_runner_{sanitize_slug(canonical_target)}_"
        f"{runtime_image_digest_slug(runtime_image_ref)}.yaml"
    )


def _run_sky_inspect_command(
    command: list[str],
    *,
    repo_root: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    process_env.update(env)
    return subprocess.run(
        command,
        cwd=repo_root,
        env=process_env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def sky_runner_active(cluster: str, *, repo_root: Path, env: dict[str, str]) -> bool:
    status_result = _run_sky_inspect_command(
        ["sky", "status", "--refresh"],
        repo_root=repo_root,
        env=env,
    )
    if status_result.returncode != 0:
        print("warning: sky status --refresh failed; assuming runner is absent")
        if status_result.stdout.strip():
            print(status_result.stdout.strip())
        return False
    if cluster not in status_result.stdout:
        return False

    queue_result = _run_sky_inspect_command(
        ["sky", "queue", cluster],
        repo_root=repo_root,
        env=env,
    )
    if queue_result.returncode != 0:
        print("warning: sky queue failed; assuming runner is absent")
        if queue_result.stdout.strip():
            print(queue_result.stdout.strip())
        return False
    active_statuses = ("RUNNING", "PENDING", "INIT", "SUBMITTED")
    if any(status in queue_result.stdout.upper() for status in active_statuses):
        return True
    print(f"runner_cluster_without_active_job: {cluster}")
    return False


def cmd_queue_train(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    profile = load_runner_profile(args.profile)
    instance_config = load_instance_config(repo_root, args.instances)
    target = target_name(profile, args.target)
    canonical_target = canonical_target_name(instance_config, target)
    instance = instance_defaults(instance_config, target)
    runtime_image_ref = normalize_runtime_image_ref(args.runtime_image_ref)

    runner_profile = dict(profile)
    runner_profile["image_id"] = runtime_image_ref
    runner_profile["runtime_image_ref"] = runtime_image_ref
    runner_profile["prebuilt_image"] = True

    output = args.output or default_runner_task_path(canonical_target, runtime_image_ref)
    task_path = write_rendered_runner_task(
        runner_profile,
        instance_config,
        repo_root,
        output,
        target_override=target,
    )
    cluster = str(
        args.cluster
        or runner_cluster_name(
            profile_id=str(profile["profile_id"]),
            target=canonical_target,
            runtime_image_ref=runtime_image_ref,
        )
    )
    env = merged_env(repo_root / ".env")
    launch_command = build_runner_launch_command(
        cluster,
        task_path,
        env=env,
        infra=launch_infra(instance),
        detach_run=args.detach_run,
    )
    run_target = canonical_target if args.target else None

    print(f"task: {task_path}")
    print(f"cluster: {cluster}")
    print(f"profile: {profile['profile_id']}")
    print(f"runtime_image_ref: {runtime_image_ref}")
    print(f"job_target: {run_target or 'any'}")
    print(shell_join(launch_command))
    if not args.execute:
        print("dry_run: pass --execute to enqueue the train job")
        if args.ensure_runner:
            print("dry_run: --ensure-runner would check sky status and launch the runner if absent")
        return 0

    conn = connect(database_url(args.direct))
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = enqueue_train_job(
            conn,
            goal_id=goal_id,
            experiment_spec_id=args.spec_id,
            profile_id=str(profile["profile_id"]),
            runtime_image_ref=runtime_image_ref,
            run_target=run_target,
            train_config=load_json_arg(args.train_config_json, default={}),
            priority=args.priority,
            max_attempts=args.max_attempts,
            run_name=args.run_name,
            run_description=args.run_description,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tag,
            origin_decision_id=args.origin_decision_id,
        )
    finally:
        conn.close()
    print(f"train_job_id={row['id']} status={row['status']}")

    if not args.ensure_runner:
        return 0
    if sky_runner_active(cluster, repo_root=repo_root, env=env):
        print(f"runner_active: {cluster}")
        return 0
    summary = LaunchSummary(
        command=launch_command,
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(profile["profile_id"]),
    )
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=False,
    )


def cmd_command(args: argparse.Namespace) -> int:
    cmd = build_launch_command(args.cluster, args.task, infra=args.infra)
    print(shell_join(cmd))
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    instances = instance_config.get("instances", {})
    if not isinstance(instances, dict):
        print("error: instances config must contain an instances object")
        return 1
    headers = [
        "target",
        "label",
        "available",
        "kind",
        "infra",
        "accelerator",
        "workers",
        "env_threads",
    ]
    rows = []
    for name, raw in sorted(instances.items()):
        if not isinstance(raw, dict):
            continue
        instance = instance_defaults(instance_config, str(name))
        rows.append(
            {
                "target": str(name),
                "label": instance_label(instance),
                "available": "no" if instance.get("available") is False else "yes",
                "kind": str(instance.get("kind", "skypilot")),
                "infra": str(instance.get("infra", "")),
                "accelerator": str(instance.get("accelerator", "")),
                "workers": str(instance.get("children", "")),
                "env_threads": str(instance.get("env_threads", "")),
            }
        )
    if not rows:
        print("No compute targets configured.")
        return 0
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print(" | ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row.get(header, "").ljust(widths[header]) for header in headers))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    rows = collect_results(args.log_dir, args.runs_dir)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_results_table(rows))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    output = write_launch_report(args.log_path, args.output)
    print(output)
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    cmd = cleanup_command(args.cluster)
    print(shell_join(cmd))
    print(
        "known_cancel_workaround: if SkyPilot cancel/down fails with PermissionError, "
        "use sky exec to find the training process group, kill -TERM the group, "
        "verify no trainer remains, then rerun sky down."
    )
    if not args.execute:
        print("dry_run: pass --execute to run sky down")
        return 0
    return subprocess.run(cmd, cwd=repo_root_from_args(args), check=False).returncode


def cmd_doctor_api(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    checks, command = ensure_skypilot_api(
        instance_config,
        repo_root=repo_root,
        instance_name=args.instance,
        execute=args.execute,
    )
    for check in checks:
        level = "ok" if check.ok else "error"
        print(f"{level}: {check.endpoint} {check.message}")
    if command is None:
        print("error: no healthy SkyPilot API endpoint found")
        return 1
    print(shell_join(command))
    if not args.execute:
        print("dry_run: pass --execute to run sky api login")
    return 0


def cmd_repro_from_wandb(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_args(args)
    instance_config = load_instance_config(repo_root, args.instances)
    config = fetch_wandb_run_config(args.run_ref)
    manifest = manifest_from_wandb_config(
        args.run_ref,
        config,
        str(args.rom_source),
        name=args.name,
        cluster=args.cluster,
        artifact_storage_uri=args.artifact_storage_uri,
    )
    if args.target:
        manifest["target"] = args.target
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"manifest: {args.manifest_output}")

    checks = preflight_checks(
        manifest,
        instance_config,
        repo_root,
        env=merged_env(repo_root / ".env"),
        target_override=args.target,
    )
    for check in checks:
        print(f"{check.level}: {check.message}")
    if any(check.level == "error" for check in checks):
        return 1

    if args.ensure_api:
        api_checks, command = ensure_skypilot_api(
            instance_config,
            repo_root=repo_root,
            instance_name=target_name(manifest, args.target),
            execute=args.execute,
        )
        for check in api_checks:
            level = "ok" if check.ok else "error"
            print(f"{level}: {check.endpoint} {check.message}")
        if command is None:
            return 1
        print(shell_join(command))

    target = target_name(manifest, args.target)
    instance = instance_defaults(instance_config, target)
    task_path = write_rendered_task(
        manifest,
        instance_config,
        repo_root,
        args.output,
        target_override=target,
    )
    cluster = str(manifest.get("cluster", manifest["name"]))
    summary = LaunchSummary(
        command=build_launch_command(
            cluster,
            task_path,
            infra=launch_infra(instance),
            detach_run=args.detach_run,
        ),
        task_path=task_path,
        cluster=cluster,
        wandb_group_prefix=str(manifest.get("wandb_group_prefix", manifest["name"])),
    )
    print(f"task: {task_path}")
    print(f"cluster: {cluster}")
    print(shell_join(summary.command))
    if not args.execute:
        print("dry_run: pass --execute to run sky launch")
        return 0
    return execute_launch(
        summary,
        repo_root,
        repo_root / ".env",
        sparse=args.sparse,
        log_path=args.log_output,
        down_on_complete=args.down_on_complete,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render, launch, and summarize rlab SkyPilot RTX4090 batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render a manifest to SkyPilot YAML")
    add_common_manifest_args(render)
    render.add_argument("--output", type=Path, help="Write rendered task YAML to this path")
    render.set_defaults(func=cmd_render)

    render_runner = subparsers.add_parser(
        "render-runner",
        help="Render a long-lived train-runner profile to SkyPilot YAML",
    )
    add_common_runner_args(render_runner)
    render_runner.add_argument("--output", type=Path, help="Write rendered task YAML to this path")
    render_runner.set_defaults(func=cmd_render_runner)

    preflight = subparsers.add_parser("preflight", help="Check manifest/env launch readiness")
    add_common_manifest_args(preflight)
    preflight.set_defaults(func=cmd_preflight)

    preflight_runner = subparsers.add_parser(
        "preflight-runner",
        help="Check runner profile/env launch readiness",
    )
    add_common_runner_args(preflight_runner)
    preflight_runner.set_defaults(func=cmd_preflight_runner)

    launch = subparsers.add_parser("launch", help="Render a task and optionally run sky launch")
    add_common_manifest_args(launch)
    launch.add_argument(
        "--output",
        type=Path,
        default=Path("sky_rlab_generated_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    launch.add_argument("--execute", action="store_true", help="Actually run sky launch")
    launch.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    launch.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    launch.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    launch.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    launch.set_defaults(func=cmd_launch)

    launch_runner = subparsers.add_parser(
        "launch-runner",
        help="Render a train-runner task and optionally run sky launch",
    )
    add_common_runner_args(launch_runner)
    launch_runner.add_argument(
        "--output",
        type=Path,
        default=Path("sky_train_runner_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    launch_runner.add_argument("--cluster", help="SkyPilot cluster name; overrides profile.cluster")
    launch_runner.add_argument("--execute", action="store_true", help="Actually run sky launch")
    launch_runner.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    launch_runner.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    launch_runner.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    launch_runner.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    launch_runner.set_defaults(func=cmd_launch_runner)

    queue_train = subparsers.add_parser(
        "queue-train",
        help="Enqueue a digest-pinned train job and optionally ensure its runner exists",
    )
    add_common_runner_args(queue_train)
    queue_train.add_argument("--goal", required=True, help="Research goal slug")
    queue_train.add_argument("--spec-id", type=int, required=True)
    queue_train.add_argument("--train-config-json", required=True)
    queue_train.add_argument("--runtime-image-ref", required=True)
    queue_train.add_argument("--priority", type=int, default=0)
    queue_train.add_argument("--max-attempts", type=int, default=1)
    queue_train.add_argument("--run-name")
    queue_train.add_argument("--run-description")
    queue_train.add_argument("--wandb-group")
    queue_train.add_argument("--wandb-tag", action="append", default=[])
    queue_train.add_argument("--origin-decision-id", type=int)
    queue_train.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    queue_train.add_argument("--ensure-runner", action="store_true")
    queue_train.add_argument("--cluster", help="SkyPilot cluster name; defaults to target/profile/digest")
    queue_train.add_argument("--output", type=Path, help="Rendered SkyPilot YAML path.")
    queue_train.add_argument("--execute", action="store_true", help="Actually enqueue and launch if needed")
    queue_train.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    queue_train.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    queue_train.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    queue_train.set_defaults(func=cmd_queue_train)

    command = subparsers.add_parser("command", help="Print the standard sky launch command")
    command.add_argument("cluster")
    command.add_argument("task", type=Path)
    command.add_argument("--infra", help="Optional SkyPilot infra target to include in command")
    command.set_defaults(func=cmd_command)

    targets = subparsers.add_parser("targets", help="List configured compute targets")
    targets.add_argument("--repo-root", default=".")
    targets.add_argument("--instances", type=Path)
    targets.set_defaults(func=cmd_targets)

    collect = subparsers.add_parser("collect", help="Summarize child logs and run markers")
    collect.add_argument("log_dir", type=Path)
    collect.add_argument("--runs-dir", type=Path, default=Path("runs"))
    collect.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    collect.set_defaults(func=cmd_collect)

    report = subparsers.add_parser("report", help="Write a JSON report from a full SkyPilot launch log")
    report.add_argument("log_path", type=Path)
    report.add_argument("--output", type=Path, default=Path("reports/skypilot_launch_report.json"))
    report.set_defaults(func=cmd_report)

    repro = subparsers.add_parser(
        "repro-from-wandb",
        help="Clone a W&B run config into a ROM-agnostic RTX4090 SkyPilot launch.",
    )
    repro.add_argument("run_ref", help="W&B run ref in entity/project/run_id form")
    repro.add_argument("--rom-source", required=True, type=Path, help="Local ROM file to mount")
    repro.add_argument("--repo-root", default=".")
    repro.add_argument("--instances", type=Path)
    repro.add_argument("--name", help="Experiment name; defaults to repro-<run-id>-4090")
    repro.add_argument("--cluster", help="SkyPilot cluster name")
    repro.add_argument(
        "--artifact-storage-uri",
        default="${CHECKPOINT_BUCKET_URI}",
        help="s3:// bucket/prefix or ${CHECKPOINT_BUCKET_URI}; training appends the game id.",
    )
    repro.add_argument(
        "--output",
        type=Path,
        default=Path("sky_repro_wandb_4090.yaml"),
        help="Rendered SkyPilot YAML path.",
    )
    repro.add_argument("--manifest-output", type=Path, help="Optional generated manifest JSON path")
    repro.add_argument("--ensure-api", action="store_true", help="Select and login to a healthy API endpoint")
    repro.add_argument(
        "--target",
        help="Named compute target from experiments/instances.json; overrides generated manifest target.",
    )
    repro.add_argument("--execute", action="store_true", help="Actually run sky launch")
    repro.add_argument(
        "--detach-run",
        action="store_true",
        help="Submit the SkyPilot job and return without streaming remote logs.",
    )
    repro.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a full local launch log but print only milestone lines while SkyPilot runs.",
    )
    repro.add_argument("--log-output", type=Path, help="Full SkyPilot launch log path for --sparse")
    repro.add_argument(
        "--down-on-complete",
        action="store_true",
        help="Run the standard sky down cleanup after the launch command exits.",
    )
    repro.set_defaults(func=cmd_repro_from_wandb)

    cleanup = subparsers.add_parser("cleanup", help="Print or run standard cleanup")
    cleanup.add_argument("cluster")
    cleanup.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used as cwd if --execute is passed.",
    )
    cleanup.add_argument("--execute", action="store_true", help="Actually run sky down")
    cleanup.set_defaults(func=cmd_cleanup)

    doctor = subparsers.add_parser("doctor-api", help="Find and optionally select a healthy SkyPilot API endpoint")
    doctor.add_argument("--repo-root", default=".")
    doctor.add_argument("--instances", type=Path)
    doctor.add_argument("--instance", default="rtx4090", help="Target name from experiments/instances.json")
    doctor.add_argument("--execute", action="store_true", help="Actually run sky api login")
    doctor.set_defaults(func=cmd_doctor_api)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
