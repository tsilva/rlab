from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.logger import HumanOutputFormat

from rlab.callbacks import CallbackHelper


SB3_HUMAN_OUTPUT_MAX_LENGTH = 512


def disable_sb3_human_output_truncation(
    model, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH
) -> None:
    logger = getattr(model, "_logger", None)
    logger_attr = getattr(type(model), "logger", None)
    if logger is None and not isinstance(logger_attr, property):
        logger = getattr(model, "logger", None)
    if logger is None:
        return
    for output_format in getattr(logger, "output_formats", ()):
        if isinstance(output_format, HumanOutputFormat):
            output_format.max_length = max_length


class Sb3HumanOutputFormatHelper(CallbackHelper):
    def __init__(self, *, max_length: int = SB3_HUMAN_OUTPUT_MAX_LENGTH) -> None:
        super().__init__()
        self.max_length = max_length

    def _on_training_start(self) -> None:
        disable_sb3_human_output_truncation(self.model, max_length=self.max_length)

    def _on_step(self) -> bool:
        return True


class GracefulStopHelper(CallbackHelper):
    def __init__(self, stop_flag: Any, *, marker_path: Path | None = None) -> None:
        super().__init__()
        self.stop_flag = stop_flag
        self.marker_path = marker_path
        self.logged = False

    def _on_step(self) -> bool:
        if not self.stop_flag.requested:
            return True
        if not self.logged:
            reason = self.stop_flag.reason or "graceful stop"
            print(
                f"graceful stop requested by {reason}; stopping at "
                f"num_timesteps={self.num_timesteps}",
                flush=True,
            )
            if self.marker_path is not None:
                self.marker_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.marker_path.with_suffix(self.marker_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "observed_at": datetime.now(UTC).isoformat(),
                            "reason": reason,
                            "num_timesteps": int(self.num_timesteps),
                            "pid": os.getpid(),
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(self.marker_path)
            self.logged = True
        return False
