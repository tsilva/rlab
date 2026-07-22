from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import shutil
import subprocess
from itertools import chain
import cv2
import numpy as np


def _uint8_image(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype == np.uint8:
        return np.ascontiguousarray(source)
    if np.issubdtype(source.dtype, np.floating) and source.size:
        finite = source[np.isfinite(source)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            source = source * 255.0
    return np.ascontiguousarray(np.clip(source, 0, 255), dtype=np.uint8)


def policy_observation_mosaic(
    observation: object,
    *,
    max_lanes: int = 4,
) -> tuple[np.ndarray, int] | None:
    """Return a compact grayscale mosaic of the policy-facing vector observation."""

    if isinstance(observation, Mapping):
        observation = observation.get("image")
    if observation is None:
        return None
    source = np.asarray(observation)
    grayscale: np.ndarray
    if source.ndim == 3 and source.shape[1] > 4 and source.shape[2] > 4:
        grayscale = source
    elif source.ndim == 4 and source.shape[1] in {1, 4}:
        # Policy frame stacks are oldest-to-newest along the channel axis.
        grayscale = source[:, -1]
    elif source.ndim == 4 and source.shape[-1] in {1, 4}:
        grayscale = source[..., -1]
    elif source.ndim == 4 and source.shape[1] == 3:
        rgb = np.moveaxis(source, 1, -1).astype(np.float32, copy=False)
        grayscale = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    elif source.ndim == 4 and source.shape[-1] == 3:
        rgb = source.astype(np.float32, copy=False)
        grayscale = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    else:
        return None
    if grayscale.shape[0] < 1:
        return None
    lane_count = min(int(grayscale.shape[0]), max(1, int(max_lanes)), 4)
    lanes = _uint8_image(grayscale[:lane_count])
    if lane_count == 1:
        return lanes[0].copy(), lane_count
    if lane_count == 2:
        return np.ascontiguousarray(np.concatenate((lanes[0], lanes[1]), axis=1)), lane_count
    height, width = lanes.shape[1:]
    mosaic = np.zeros((height * 2, width * 2), dtype=np.uint8)
    for index, lane in enumerate(lanes):
        row, column = divmod(index, 2)
        mosaic[row * height : (row + 1) * height, column * width : (column + 1) * width] = lane
    return mosaic, lane_count


class PolicyObservationPreview:
    """Bounded in-memory capture of actual policy observations for diagnostics."""

    def __init__(self, *, max_frames: int = 450, max_lanes: int = 4) -> None:
        self.max_frames = max(1, int(max_frames))
        self.max_lanes = max(1, int(max_lanes))
        self.frames: list[np.ndarray] = []
        self.lane_count = 0
        self.error: str | None = None
        self._disabled = False

    def capture(self, observation: object) -> bool:
        if self._disabled or len(self.frames) >= self.max_frames:
            return False
        try:
            converted = policy_observation_mosaic(observation, max_lanes=self.max_lanes)
            if converted is None:
                self.error = "policy observation is not a supported image batch"
                self._disabled = True
                return False
            frame, lane_count = converted
            if self.frames and frame.shape != self.frames[0].shape:
                self.error = "policy observation image shape changed during evaluation"
                self._disabled = True
                return False
            self.frames.append(frame.copy())
            self.lane_count = lane_count
            return True
        except Exception as exc:
            self.error = f"preview capture failed: {type(exc).__name__}"
            self._disabled = True
            return False


def write_preview_video(
    frames: list[np.ndarray],
    output: Path,
    *,
    fps: int,
    scale: int,
    timeout_seconds: int,
    max_bytes: int,
) -> dict[str, int | float]:
    """Encode a tiny grayscale H.264 preview within a hard wall-time and size bound."""

    if not frames:
        raise ValueError("No preview frames to write")
    first = np.asarray(frames[0])
    if first.ndim != 2:
        raise ValueError("preview frames must be grayscale")
    height, width = first.shape
    if any(np.asarray(frame).shape != first.shape for frame in frames):
        raise ValueError("preview frame shapes must be constant")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write browser-compatible MP4 video")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        f"scale=iw*{int(scale)}:ih*{int(scale)}:flags=neighbor",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "34",
        "-maxrate",
        "250k",
        "-bufsize",
        "500k",
        "-threads",
        "1",
        "-profile:v",
        "main",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    raw_video = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    try:
        completed = subprocess.run(
            command,
            input=raw_video,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError("preview encoding timed out") from exc
    if completed.returncode:
        output.unlink(missing_ok=True)
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to write preview: {message[-1000:]}")
    size = output.stat().st_size
    if size < 1:
        output.unlink(missing_ok=True)
        raise RuntimeError("preview encoder produced an empty file")
    if size > int(max_bytes):
        output.unlink(missing_ok=True)
        raise RuntimeError(f"preview exceeds size limit: {size} > {int(max_bytes)}")
    return {
        "frames": len(frames),
        "fps": int(fps),
        "width": width * int(scale),
        "height": height * int(scale),
        "duration_seconds": len(frames) / float(fps),
        "size_bytes": size,
    }


def write_video(frames: Iterable[np.ndarray], output: Path, fps: float, scale: int) -> None:
    frame_iterator = iter(frames)
    try:
        first_frame = np.asarray(next(frame_iterator))
    except StopIteration as exc:
        raise ValueError("No frames to write") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    out_size = (width * scale, height * scale)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write browser-compatible MP4 video")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_size[0]}x{out_size[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in chain((first_frame,), frame_iterator):
            frame = np.asarray(frame)
            if frame.shape[:2] != (height, width) or frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(
                    f"video frames must have a constant HxWx3 shape; got {frame.shape}"
                )
            if scale != 1:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_NEAREST)
            process.stdin.write(np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        if not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()
    if return_code != 0:
        output.unlink(missing_ok=True)
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to write {output}: {message}")
