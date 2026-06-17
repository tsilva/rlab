from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_video(frames: list[np.ndarray], output: Path, fps: float, scale: int) -> None:
    if not frames:
        raise ValueError("No frames to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    out_size = (width * scale, height * scale)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame in frames:
            if scale != 1:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def replay_actions_for_video(env, actions: list[Any], seed: int) -> list[np.ndarray]:
    env.reset(seed=seed)
    frames = [env.render()]
    for action in actions:
        _obs, _reward, terminated, truncated, _info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    return frames
