from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


LEVEL_STATE_RE = re.compile(r"^Level\d+-\d+$")


def normalize_wandb_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = value.split(",")
    elif isinstance(value, list | tuple):
        raw_tags = value
    else:
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def append_unique_wandb_tag(tags: list[str], tag: str) -> list[str]:
    tag = tag.strip()
    if tag and tag not in set(tags):
        tags.append(tag)
    return tags


def normalize_level_states(config: Mapping[str, Any]) -> list[str]:
    raw_states = config.get("states") or []
    if isinstance(raw_states, str):
        candidates = [raw_states]
    elif isinstance(raw_states, list | tuple):
        candidates = [str(state) for state in raw_states]
    else:
        candidates = []

    state = str(config.get("state") or "").strip()
    if state:
        candidates.append(state)

    levels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        level = candidate.strip()
        if not LEVEL_STATE_RE.match(level) or level in seen:
            continue
        seen.add(level)
        levels.append(level)
    return levels
