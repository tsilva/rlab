from __future__ import annotations

import re


def metric_path_segment(value: object) -> str:
    segment = str(value).strip()
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment)
    return segment.strip("_") or "unknown"
