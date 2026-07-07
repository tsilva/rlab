from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def label_path(label: str, key: str) -> str:
    return f"{label}.{key}" if label else key


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def require_int(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    minimum: int | None = None,
    require_present: bool = True,
) -> int:
    value = require_key(document, key, label=label) if require_present else document.get(key)
    if not is_int(value):
        raise ValueError(f"{label_path(label, key)} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label_path(label, key)} must be >= {minimum}")
    return value


def require_key(document: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in document:
        raise ValueError(f"{label_path(label, key)} is required")
    return document[key]


def require_non_empty_string(
    document: Mapping[str, Any],
    key: str,
    *,
    label: str,
    require_present: bool = True,
    strip: bool = True,
) -> str:
    value = require_key(document, key, label=label) if require_present else document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label_path(label, key)} must be a non-empty string")
    return value.strip() if strip else value


def require_schema_version(
    document: Mapping[str, Any],
    expected: int,
    *,
    label: str,
    require_present: bool = True,
) -> None:
    schema_version = require_int(
        document,
        "schema_version",
        label=label,
        minimum=1,
        require_present=require_present,
    )
    if schema_version != expected:
        raise ValueError(f"{label_path(label, 'schema_version')} must be {expected}, got {schema_version}")


def string_list(value: Any, *, label: str, allow_empty: bool = False, strip: bool = True) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip() if strip else item)
    return result


def int_list(value: Any, *, label: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    result: list[int] = []
    for index, item in enumerate(value):
        if not is_int(item):
            raise ValueError(f"{label}[{index}] must be an integer")
        result.append(item)
    return result
