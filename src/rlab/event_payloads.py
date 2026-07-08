from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def event_payloads(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(name): payload for name, payload in value.items() if str(name)}
    if isinstance(value, (list, tuple, set)):
        return {str(name): {} for name in value if str(name)}
    if isinstance(value, str) and value:
        return {value: {}}
    return {}


def info_event_payloads(info: Mapping[str, Any]) -> dict[str, Any]:
    info_events = info.get("info_events")
    if isinstance(info_events, Mapping):
        return event_payloads(info_events)
    return event_payloads(info.get("done_on_info"))
