from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def wrapper_spec_id(spec: Mapping[str, Any], *, label: str, id_keys: Sequence[str]) -> str:
    for key in id_keys:
        value = spec.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
            return value.strip()
    raise ValueError(f"{label} must include one of: {', '.join(id_keys)}")


def normalize_wrapper_spec_sequence(
    value: Any,
    *,
    label: str,
    id_keys: Sequence[str],
    item_kind: str,
) -> tuple[dict[str, Any], ...]:
    if value in (None, "", ()):
        return ()
    if isinstance(value, str):
        value = [{"id": value}]
    elif isinstance(value, Mapping):
        value = [value]
    elif not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list of {item_kind} specs")

    specs: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if isinstance(item, str):
            spec = {"id": item}
        elif isinstance(item, Mapping):
            spec = deepcopy(dict(item))
        else:
            raise ValueError(f"{item_label} must be an object or {item_kind} id string")
        spec["id"] = wrapper_spec_id(spec, label=item_label, id_keys=id_keys)
        specs.append(spec)
    return tuple(specs)
