from __future__ import annotations

import argparse
from collections.abc import Sequence


def explicit_arg_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    option_dests = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    return {
        option_dests[argument.split("=", 1)[0]]
        for argument in argv
        if argument.split("=", 1)[0] in option_dests
    }


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Preview planned changes without applying them.",
    )


def add_direct_database_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
