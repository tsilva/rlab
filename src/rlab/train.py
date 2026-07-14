from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from rlab.artifacts import init_wandb, write_run_description, write_wandb_url
from rlab.cli_args import explicit_arg_dests, parse_json_value
from rlab.env import (
    EnvConfig,
    assert_provider_runtime_available,
    default_run_dir,
    resolve_env_config,
    resolve_mixed_state_config,
)
from rlab.env_config import env_config_from_args, parse_obs_crop
from rlab.metric_store import MetricStore, metric_store_path
from rlab.provider_config import provider_num_envs
from rlab.seeds import validate_training_seed
from rlab.train_config import (
    TRAIN_CONFIG_FIELDS,
    add_train_config_args,
    apply_training_backend_arg_view,
    load_materialized_train_config,
    materialized_train_args,
    validate_and_normalize_train_config,
)
from rlab.training_backend import (
    BackendContext,
    GracefulStopFlag,
    load_training_backend,
    training_backend_config,
    training_backend_config_hash,
    training_backend_id,
    training_backend_runtime_metadata,
)


GRACEFUL_STOP_SIGNAL = getattr(signal, "SIGUSR1", None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a configured backend on a registered provider environment"
    )
    parser.add_argument(
        "--train-config-json",
        type=Path,
        help="Authoritative materialized train configuration JSON.",
    )
    add_train_config_args(
        parser,
        env_defaults=EnvConfig(),
        parse_json_value=parse_json_value,
        parse_obs_crop=parse_obs_crop,
    )
    return parser


def explicit_n_envs(args: argparse.Namespace) -> int | None:
    explicit_fields = set(getattr(args, "_explicit_train_arg_dests", set())) | set(
        getattr(args, "_train_config_json_fields", set())
    )
    return int(args.n_envs) if "n_envs" in explicit_fields else None


def effective_n_envs(args: argparse.Namespace) -> int:
    return provider_num_envs(args, explicit_n_envs=explicit_n_envs(args))


def _direct_train_config(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in TRAIN_CONFIG_FIELDS:
        value = getattr(args, field.dest, None)
        if value is None or value == "" or (field.sequence_items and value in ((), [])):
            continue
        if field.sequence_items and isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            value = [float(item) for item in items] if field.sequence_items == "number" else items
        payload[field.dest] = value
    return validate_and_normalize_train_config(
        payload,
        label="train arguments",
        required_keys=("training_backend",),
    )


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    explicit_dests = explicit_arg_dests(parser, argv_list)
    parsed = parser.parse_args(argv_list)
    if parsed.train_config_json is None:
        parsed._train_config_json_fields = set()
        parsed._explicit_train_arg_dests = set(explicit_dests)
        parsed._materialized_train_config = _direct_train_config(parsed)
        apply_training_backend_arg_view(parsed, parsed._materialized_train_config)
        args = parsed
    else:
        path = Path(parsed.train_config_json)
        payload = load_materialized_train_config(path)
        args = materialized_train_args(path)
        for key in explicit_dests:
            if key != "train_config_json":
                value = getattr(parsed, key)
                setattr(args, key, value)
                if key in payload:
                    payload[key] = value
        args._explicit_train_arg_dests = set(explicit_dests)
        args._materialized_train_config = payload
    validate_training_seed(args.seed, label="--seed", seed_span=effective_n_envs(args))
    return args


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal-{signum}"


def install_graceful_stop_handler(stop_flag: GracefulStopFlag) -> int | None:
    if GRACEFUL_STOP_SIGNAL is None:
        return None

    def handle_graceful_stop(signum, _frame) -> None:
        stop_flag.request(signal_name(signum))

    signal.signal(GRACEFUL_STOP_SIGNAL, handle_graceful_stop)
    return int(GRACEFUL_STOP_SIGNAL)


def main(argv: list[str] | None = None) -> int:
    args = parse_train_args(argv)
    train_config = dict(args._materialized_train_config)
    backend_id = training_backend_id(train_config)
    backend_config = training_backend_config(train_config)
    backend = load_training_backend(backend_id)
    backend.validate(train_config, backend_config)
    for key, value in training_backend_runtime_metadata(backend_id, backend_config).items():
        setattr(args, key, value)
    args.training_backend_config_hash = training_backend_config_hash(train_config)

    environment = resolve_env_config(env_config_from_args(args, include_states=True))
    n_envs = effective_n_envs(args)
    environment = resolve_mixed_state_config(environment, n_envs=n_envs)
    assert_provider_runtime_available(environment)
    args.resolved_n_envs = n_envs

    run_dir = Path(default_run_dir(args.run_name, args.runs_dir))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "learner_ready.json").unlink(missing_ok=True)
    store = MetricStore(metric_store_path(run_dir))
    store.init()
    write_run_description(args, str(run_dir))
    if args.run_description.strip():
        print(f"run description: {args.run_description.strip()}", flush=True)
    else:
        print("warning: --run-description is empty", flush=True)

    external_publisher = os.environ.get("RLAB_EXTERNAL_WANDB_PUBLISHER") == "1"
    wandb_run = None if external_publisher else init_wandb(args, str(run_dir), environment)
    write_wandb_url(wandb_run, str(run_dir))
    stop_flag = GracefulStopFlag()
    graceful_stop_signal = install_graceful_stop_handler(stop_flag)
    if graceful_stop_signal is not None:
        print(f"graceful stop signal: {signal_name(graceful_stop_signal)}", flush=True)

    context = BackendContext(
        train_config=train_config,
        args=args,
        environment=environment,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        metric_store=store,
        wandb_run=wandb_run,
        stop_flag=stop_flag,
        external_wandb_publisher=external_publisher,
    )
    try:
        backend.run(context)
        write_wandb_url(wandb_run, str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
