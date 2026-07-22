from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


Command = Callable[[argparse.Namespace], int]


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a number > 0")
    return parsed


def _dataset_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (default: ~/.rlab/datasets).",
    )


def _record(args: argparse.Namespace) -> int:
    from rlab.dataset_record import record_command

    return record_command(args)


def _list(args: argparse.Namespace) -> int:
    from rlab.dataset_store import list_command

    return list_command(args)


def _play(args: argparse.Namespace) -> int:
    from rlab.dataset_store import play_command

    return play_command(args)


def _verify(args: argparse.Namespace) -> int:
    from rlab.dataset_store import verify_command

    return verify_command(args)


def _video(args: argparse.Namespace) -> int:
    from rlab.dataset_store import video_command

    return video_command(args)


def _upload(args: argparse.Namespace) -> int:
    from rlab.dataset_hub import upload_command

    return upload_command(args)


def _adopt(args: argparse.Namespace) -> int:
    from rlab.dataset_store import adopt_command

    return adopt_command(args)


def _export_minari(args: argparse.Namespace) -> int:
    from rlab.dataset_minari import export_minari_command

    return export_minari_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab dataset",
        description=(
            "Record and inspect Gymrec v3-compatible gameplay datasets. Dataset results are "
            "never checkpoint-promotion or research-goal evidence."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    record = subparsers.add_parser("record", help="Record gameplay into a local collection.")
    record.add_argument("reference", help="Stable local collection reference.")
    record.add_argument("--env-id", required=True, help="Provider-native environment identifier.")
    record.add_argument("--provider", default="stable-retro-turbo")
    record.add_argument("--env-config", type=_json_object, default={})
    record.add_argument("--agent", choices=("human", "random", "ppo"), default="human")
    record.add_argument("--model", default=None, help="Approved PPO checkpoint source.")
    record.add_argument("--episodes", type=_positive_int, default=None)
    record.add_argument("--seed", type=int, default=0)
    record.add_argument("--headless", action="store_true")
    record.add_argument("--fps", type=_positive_float, default=None)
    record.add_argument("--upload-live", action="store_true")
    sampling = record.add_mutually_exclusive_group()
    sampling.add_argument("--deterministic", action="store_true", default=None)
    sampling.add_argument("--stochastic", action="store_false", dest="deterministic")
    _dataset_root_arg(record)
    record.set_defaults(handler=_record)

    listing = subparsers.add_parser("list", help="List local collections or episodes.")
    listing.add_argument("reference", nargs="?", default=None)
    _dataset_root_arg(listing)
    listing.set_defaults(handler=_list)

    play = subparsers.add_parser("play", help="Play recorded observations without a provider.")
    play.add_argument("source")
    play.add_argument("--episode", type=_positive_int, default=1)
    play.add_argument("--fps", type=float, default=None)
    play.add_argument("--scale", type=_positive_int, default=3)
    _dataset_root_arg(play)
    play.set_defaults(handler=_play)

    verify = subparsers.add_parser("verify", help="Verify v3 structure and media integrity.")
    verify.add_argument("source")
    verify.add_argument("--reexecute", action="store_true")
    _dataset_root_arg(verify)
    verify.set_defaults(handler=_verify)

    video = subparsers.add_parser("video", help="Export recorded observations to browser MP4.")
    video.add_argument("source")
    video.add_argument("output", type=Path)
    video.add_argument("--first", type=_positive_int, default=1)
    video.add_argument("--last", type=_positive_int, default=None)
    video.add_argument("--fps", type=float, default=None)
    video.add_argument("--scale", type=_positive_int, default=1)
    _dataset_root_arg(video)
    video.set_defaults(handler=_video)

    upload = subparsers.add_parser("upload", help="Append a verified local collection to the Hub.")
    upload.add_argument("source")
    upload.add_argument("target", help="Hugging Face dataset repository ID.")
    upload.add_argument("--revision", default="main")
    _dataset_root_arg(upload)
    upload.set_defaults(handler=_upload)

    adopt = subparsers.add_parser("adopt", help="Adopt one explicit compatible v3 tree.")
    adopt.add_argument("source", type=Path)
    adopt.add_argument("reference")
    _dataset_root_arg(adopt)
    adopt.set_defaults(handler=_adopt)

    minari = subparsers.add_parser(
        "export-minari", help="Export provider-bounded episodes to a new Minari dataset."
    )
    minari.add_argument("source")
    minari.add_argument("name")
    minari.add_argument("--author", default=None)
    _dataset_root_arg(minari)
    minari.set_defaults(handler=_export_minari)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    handler: Command = args.handler
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
