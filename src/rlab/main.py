from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


CommandMain = Callable[[list[str] | None], object]


def _run(command: CommandMain, argv: Sequence[str]) -> int:
    result = command(list(argv))
    return int(result) if isinstance(result, int) else 0


def _experiment(argv: Sequence[str]) -> int:
    from rlab.experiment_cli import main as experiment_main

    return _run(experiment_main, argv)


def _eval(argv: Sequence[str]) -> int:
    if argv and argv[0] == "run":
        from rlab.eval import main as eval_main

        return _run(eval_main, argv[1:])
    parser = argparse.ArgumentParser(
        prog="rlab eval",
        description="Run an ad-hoc evaluation locally.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    commands.add_parser("run", help="Evaluate one model against an environment contract.")
    if argv and argv[0] in {"-h", "--help"}:
        parser.parse_args(["--help"])
    if not argv:
        parser.print_help()
        return 2
    parser.error(f"unknown eval command: {argv[0]}")
    return 2


def _leaders(argv: Sequence[str]) -> int:
    from rlab.wandb_leaders import main as leaders_main

    return _run(leaders_main, argv)


def _reports(argv: Sequence[str]) -> int:
    from rlab.wandb_reports import main as reports_main

    return _run(reports_main, argv)


def _play(argv: Sequence[str]) -> int:
    from rlab.play import main as play_main

    return _run(play_main, argv)


def _import_roms(argv: Sequence[str]) -> int:
    from rlab.import_roms import main as import_roms_main

    return _run(import_roms_main, argv)


def _benchmark(argv: Sequence[str]) -> int:
    from rlab.benchmark import main as benchmark_main

    return _run(benchmark_main, argv)


def _validate(argv: Sequence[str]) -> int:
    from rlab.config_validation import main as validate_main

    return _run(validate_main, argv)


def _env(argv: Sequence[str]) -> int:
    from rlab.env_cli import main as env_main

    return _run(env_main, argv)


def _rom(argv: Sequence[str]) -> int:
    from rlab.rom_cli import main as rom_main

    return _run(rom_main, argv)


def _dataset(argv: Sequence[str]) -> int:
    from rlab.dataset_cli import main as dataset_main

    return _run(dataset_main, argv)


COMMANDS: dict[str, tuple[str, Callable[[Sequence[str]], int]]] = {
    "experiment": ("launch and observe dstack training experiments", _experiment),
    "eval": ("run a direct local evaluation", _eval),
    "play": ("render a local, public-run, or Hugging Face model in a GUI window", _play),
    "import-roms": ("import ROMs into the installed rlab runtime", _import_roms),
    "benchmark": ("run gated local-smoke and throughput profiles", _benchmark),
    "validate": (
        "validate checked-in YAML experiments, recipes, benchmarks, and ops configs",
        _validate,
    ),
    "env": ("list, inspect, and preflight environment providers", _env),
    "rom": ("provision, verify, and warm immutable ROM assets", _rom),
    "dataset": ("record, inspect, verify, migrate, and publish gameplay datasets", _dataset),
    "leaders": ("query accepted runs and promoted checkpoints", _leaders),
    "reports": ("plan, synchronize, and verify declarative W&B reports", _reports),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab",
        description="Unified command surface for rlab training, eval, playback, and ops.",
        epilog=(
            "Research: experiment, eval, play, validate.  Environments: env, rom, import-roms, "
            "benchmark.  Datasets: dataset.  Results: leaders, reports."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name in COMMANDS:
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
