from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from rlab.config_validation import load_goal_contract
from rlab.publication import (
    GITATTRIBUTES_TEXT,
    MIT_LICENSE_TEXT,
    build_model_repo_id,
    build_release_manifest,
    publication_identity_from_model_metadata,
    publication_model_metadata,
    release_artifact_records,
    validate_release_bundle,
)


def _load_object(path: Path, *, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _verify_replay(path: Path) -> dict[str, object]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to validate replay.mp4")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(completed.stdout)
    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("replay video does not contain a video stream")
    stream = streams[0]
    expected = {"codec_name": "h264", "codec_tag_string": "avc1", "pix_fmt": "yuv420p"}
    for key, value in expected.items():
        if stream.get(key) != value:
            raise ValueError(f"replay video {key} must be {value!r}, got {stream.get(key)!r}")
    duration = float(probe.get("format", {}).get("duration") or 0.0)
    frames = int(stream.get("nb_read_frames") or 0)
    if duration <= 0 or frames <= 0:
        raise ValueError("replay video must have a positive duration and frame count")
    data = path.read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("replay video must use faststart with moov before mdat")
    return {"duration_seconds": duration, "frames": frames, **expected}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate one deterministic rlab Hugging Face release bundle."
    )
    parser.add_argument("--goal-file", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--release-version")
    parser.add_argument("--published-at")
    parser.add_argument("--youtube-url")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    goal = load_goal_contract(args.goal_file)
    metadata = _load_object(args.model_metadata, label="model metadata")
    identity = publication_identity_from_model_metadata(goal.get("goal_id"), metadata)
    repo_id = build_model_repo_id(identity)
    summary = {"repo_id": repo_id, **identity.__dict__}
    if args.identity_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    required = {
        "--model": args.model,
        "--replay": args.replay,
        "--readme": args.readme,
        "--evaluation-json": args.evaluation_json,
        "--source-json": args.source_json,
        "--release-version": args.release_version,
        "--youtube-url": args.youtube_url,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("full bundle preparation requires " + ", ".join(missing))
    assert args.model is not None
    assert args.replay is not None
    assert args.readme is not None
    assert args.evaluation_json is not None
    assert args.source_json is not None
    assert args.release_version is not None
    assert args.youtube_url is not None
    assert args.output_dir is not None
    for path in (
        args.model,
        args.replay,
        args.readme,
        args.evaluation_json,
        args.source_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"release output directory is not empty: {args.output_dir}")

    replay = _verify_replay(args.replay)
    evaluation = _load_object(args.evaluation_json, label="evaluation")
    if evaluation.get("action_sampling") != "stochastic":
        raise ValueError("release evaluation must declare action_sampling='stochastic'")
    source = _load_object(args.source_json, label="source")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model, args.output_dir / "model.zip")
    shutil.copy2(args.replay, args.output_dir / "replay.mp4")
    shutil.copy2(args.readme, args.output_dir / "README.md")
    (args.output_dir / ".gitattributes").write_text(GITATTRIBUTES_TEXT, encoding="utf-8")
    (args.output_dir / "LICENSE").write_text(MIT_LICENSE_TEXT, encoding="utf-8")
    publication_metadata = publication_model_metadata(metadata, identity)
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(publication_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_records = release_artifact_records(args.output_dir)
    evaluation.setdefault("replay", replay)
    manifest = build_release_manifest(
        identity,
        publication_metadata,
        release_version=args.release_version,
        published_at=args.published_at
        or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        source=source,
        evaluation=evaluation,
        artifacts=artifact_records,
        youtube_url=args.youtube_url,
    )
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_release_bundle(args.output_dir)
    print(json.dumps({**summary, "bundle": str(args.output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
