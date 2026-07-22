from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import contextlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from rlab.dataset_contract import (
    CANONICAL_VIDEO_SUFFIX,
    STORAGE_FORMAT_IMAGES,
    STORAGE_FORMAT_LOSSLESS_VIDEO,
    observation_to_rgb,
)


MAX_IMAGE_PIXELS = 100_000_000
MAX_DECODED_VIDEO_BYTES = 32 * 1024**3
FFMPEG_TIMEOUT_SECONDS = 30


def _safe_relative(value: Any) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or "\x00" in text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"unsafe dataset media path {value!r}")
    return path


def resolve_media_path(root: Path, value: Any) -> Path:
    relative = _safe_relative(value)
    candidate = root.joinpath(*relative.parts)
    root_resolved = root.resolve()
    try:
        candidate.resolve(strict=True).relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"dataset media escapes or is missing from its source: {value!r}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"dataset media is not a regular file: {value!r}")
    return candidate


def _image_from_feature(value: Any, root: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required; install Rlab with --extra dataset") from exc
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        path = value.get("path")
        if raw is not None:
            import io

            source: Any = io.BytesIO(bytes(raw))
        elif path:
            source = resolve_media_path(root, path)
        else:
            raise ValueError("image observation has neither bytes nor path")
    elif isinstance(value, (str, Path)):
        source = resolve_media_path(root, value)
    else:
        return observation_to_rgb(value)
    with Image.open(source) as image:
        image.load()
        return observation_to_rgb(image)


def _video_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for lossless dataset video")
    return ffmpeg, ffprobe


def probe_lossless_video(path: Path, *, width: int, height: int) -> Mapping[str, Any]:
    _ffmpeg, ffprobe = _video_tools()
    command = [
        ffprobe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "matroska",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=FFMPEG_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode:
        raise ValueError(
            f"ffprobe rejected {path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    if len(completed.stdout) > 1 << 20:
        raise ValueError(f"ffprobe returned excessive metadata for {path}")
    try:
        document = json.loads(completed.stdout)
        stream = document["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"ffprobe returned invalid metadata for {path}") from exc
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        raise ValueError(f"video dimensions do not match rows for {path}")
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") not in {"gbrp", "rgb24"}:
        raise ValueError(f"video is not canonical lossless RGB H.264: {path}")
    return stream


def iter_lossless_video_frames(
    path: Path,
    *,
    width: int,
    height: int,
    expected_frames: int,
) -> Iterator[np.ndarray]:
    if width < 1 or height < 1 or expected_frames < 1:
        raise ValueError("video dimensions and expected frame count must be positive")
    frame_bytes = width * height * 3
    total_bytes = frame_bytes * expected_frames
    if total_bytes > MAX_DECODED_VIDEO_BYTES:
        raise ValueError(f"decoded video exceeds {MAX_DECODED_VIDEO_BYTES} bytes")
    ffmpeg, _ffprobe = _video_tools()
    probe_lossless_video(path, width=width, height=height)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "matroska",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        for frame_index in range(expected_frames):
            payload = bytearray()
            while len(payload) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) != frame_bytes:
                raise ValueError(f"video ended before frame {frame_index}: {path}")
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
        if process.stdout.read(1):
            raise ValueError(f"video contains more than {expected_frames} frames: {path}")
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
        if return_code:
            raise ValueError(
                f"ffmpeg rejected {path}: " + stderr.decode("utf-8", errors="replace")[-1000:]
            )
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def iter_recoverable_video_prefix(
    path: Path,
    *,
    width: int,
    height: int,
    expected_frames: int,
) -> Iterator[np.ndarray]:
    """Decode an exact prefix from an interrupted Matroska file without trusting its tail."""
    if width < 1 or height < 1 or expected_frames < 1:
        raise ValueError("recovery dimensions and frame count must be positive")
    frame_bytes = width * height * 3
    if frame_bytes * expected_frames > MAX_DECODED_VIDEO_BYTES:
        raise ValueError(f"decoded video exceeds {MAX_DECODED_VIDEO_BYTES} bytes")
    ffmpeg, _ffprobe = _video_tools()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "matroska",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        for frame_index in range(expected_frames):
            payload = bytearray()
            while len(payload) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) != frame_bytes:
                raise ValueError(f"interrupted video lacks durable frame {frame_index}: {path}")
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stderr is not None:
            process.stderr.close()


def iter_episode_frames(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> Iterator[np.ndarray]:
    if not rows:
        return
    storage = str(rows[0].get("storage_format"))
    if storage == STORAGE_FORMAT_IMAGES:
        for row in rows:
            yield _image_from_feature(row.get("observations"), root)
        return
    if storage != STORAGE_FORMAT_LOSSLESS_VIDEO:
        raise ValueError(f"unsupported storage format {storage!r}")
    paths = {str(row.get("video_path") or "") for row in rows}
    widths = {int(row.get("frame_width") or 0) for row in rows}
    heights = {int(row.get("frame_height") or 0) for row in rows}
    if len(paths) != 1 or len(widths) != 1 or len(heights) != 1:
        raise ValueError("video episode rows disagree on path or dimensions")
    relative = paths.pop()
    if not relative.endswith(CANONICAL_VIDEO_SUFFIX):
        raise ValueError(f"video path is not canonical lossless media: {relative}")
    path = resolve_media_path(root, relative)
    for row, frame in zip(
        rows,
        iter_lossless_video_frames(
            path,
            width=widths.pop(),
            height=heights.pop(),
            expected_frames=len(rows),
        ),
        strict=True,
    ):
        expected = str(row.get("frame_sha256") or "")
        actual = hashlib.sha256(frame.tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"decoded frame hash mismatch for {relative} step {row.get('step_index')}"
            )
        yield frame


class LosslessVideoWriter:
    def __init__(self, output: Path, *, fps: float) -> None:
        self.output = output
        self.fps = float(fps)
        self._process: subprocess.Popen | None = None
        self._shape: tuple[int, int, int] | None = None

    def _start(self, frame: np.ndarray) -> None:
        ffmpeg, _ffprobe = _video_tools()
        height, width = frame.shape[:2]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "rgb24",
                "-f",
                "matroska",
                str(self.output),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._shape = tuple(frame.shape)

    def write(self, value: Any) -> np.ndarray:
        frame = observation_to_rgb(value)
        if self._process is None:
            self._start(frame)
        elif tuple(frame.shape) != self._shape:
            raise ValueError(f"recording frame shape changed from {self._shape} to {frame.shape}")
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(frame.tobytes())
        return frame

    def close(self) -> None:
        if self._process is None or self._process.stdin is None:
            raise ValueError("cannot close an empty lossless video")
        self._process.stdin.close()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        return_code = self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code:
            self.output.unlink(missing_ok=True)
            raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])
        _fsync_file(self.output)
        _fsync_directory(self.output.parent)

    def abort(self, *, preserve: bool = False) -> None:
        if self._process is not None:
            if self._process.stdin is not None and not self._process.stdin.closed:
                with contextlib.suppress(BrokenPipeError):
                    self._process.stdin.close()
            if self._process.poll() is None:
                if preserve:
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                else:
                    self._process.kill()
            self._process.wait()
        if preserve and self.output.is_file():
            _fsync_file(self.output)
            _fsync_directory(self.output.parent)
        else:
            self.output.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_episode_media(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(frame.tobytes()).hexdigest()
        for frame in iter_episode_frames(rows, root=root)
    )


def iter_selected_frames(
    episodes: Iterable[Sequence[Mapping[str, Any]]],
    *,
    root: Path,
) -> Iterator[np.ndarray]:
    for rows in episodes:
        yield from iter_episode_frames(rows, root=root)
