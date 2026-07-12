from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


CommandMain = Callable[[list[str] | None], object]


def _run(command: CommandMain, argv: Sequence[str], *, prog: str) -> int:
    del prog
    result = command(list(argv))
    return int(result) if isinstance(result, int) else 0


def _train(argv: Sequence[str]) -> int:
    from rlab.job_queue import build_train_enqueue_parser, cmd_enqueue_train

    return int(cmd_enqueue_train(build_train_enqueue_parser().parse_args(list(argv))))


def _eval(argv: Sequence[str]) -> int:
    from rlab.eval import main as eval_main

    return _run(eval_main, argv, prog="rlab eval")


def _jobs(argv: Sequence[str]) -> int:
    from rlab.job_queue import main as queue_main

    return _run(queue_main, argv, prog="rlab jobs")


def _fleet(argv: Sequence[str]) -> int:
    from rlab.fleet import main as fleet_main

    return _run(fleet_main, argv, prog="rlab fleet")


def _leaders(argv: Sequence[str]) -> int:
    from rlab.wandb_leaders import main as leaders_main

    return _run(leaders_main, argv, prog="rlab leaders")


def _run_job(argv: Sequence[str]) -> int:
    from rlab.run_job import main as run_job_main

    return _run(run_job_main, argv, prog="rlab run-job")


def _play(argv: Sequence[str]) -> int:
    from rlab.play import main as play_main

    return _run(play_main, argv, prog="rlab play")


def _distill(argv: Sequence[str]) -> int:
    from rlab.distill import main as distill_main

    return _run(distill_main, argv, prog="rlab distill")


def _import_roms(argv: Sequence[str]) -> int:
    from rlab.import_roms import main as import_roms_main

    return _run(import_roms_main, argv, prog="rlab import-roms")


def _benchmark(argv: Sequence[str]) -> int:
    from rlab.benchmark import main as benchmark_main

    return _run(benchmark_main, argv, prog="rlab benchmark")


def _validate(argv: Sequence[str]) -> int:
    from rlab.config_validation import main as validate_main

    return _run(validate_main, argv, prog="rlab validate")


COMMANDS: dict[str, tuple[str, Callable[[Sequence[str]], int]]] = {
    "train": ("enqueue queue-backed train jobs from checked-in recipes", _train),
    "eval": ("run local evals", _eval),
    "play": ("render a local, W&B, or Hugging Face model in a GUI window", _play),
    "import-roms": ("import ROMs into the installed rlab runtime", _import_roms),
    "benchmark": ("run named smoke, throughput, fleet, and eval-contract profiles", _benchmark),
    "validate": ("validate checked-in YAML experiments, recipes, benchmarks, and ops configs", _validate),
    "jobs": ("manage queue schema, status, and cancellation", _jobs),
    "leaders": ("query W&B run and checkpoint leaderboards", _leaders),
    "fleet": ("manage one-job Docker containers from queue state", _fleet),
    "run-job": ("run one claimed job payload inside a container", _run_job),
    "distill": ("distill and verify Mario policies from teacher checkpoints", _distill),
}
INTERNAL_COMMANDS = frozenset({"run-job"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab",
        description="Unified command surface for rlab training, eval, playback, and ops.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name in (name for name in COMMANDS if name not in INTERNAL_COMMANDS):
        help_text, _handler = COMMANDS[name]
        subparser = subparsers.add_parser(name, help=help_text, add_help=False)
        subparser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if argv_list else 2
    command = argv_list[0]
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    _help, handler = COMMANDS[command]
    return handler(argv_list[1:])


if __name__ == "__main__":
    raise SystemExit(main())
