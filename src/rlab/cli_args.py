from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any


def parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {exc}") from exc


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
