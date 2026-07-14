from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml


YAML_EXTENSIONS = {".yaml", ".yml"}
TEMPLATE_VARS_KEY = "template_vars"
QUEUE_TEMPLATE_VALUES: dict[str, Any] = {
    "batch_id": "bx0123456789abcdef",
    "campaign_id": "b-test",
    "seed": 123,
    "recipe_id": "candidate",
    "timestamp": "20260626T120000Z",
    "utc": "20260626T120000Z",
}
QUEUE_TEMPLATE_FIELDS = frozenset(QUEUE_TEMPLATE_VALUES)

_LEVEL_ID_RE = re.compile(r"^Level(?P<world>\d+)-(?P<level>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ComposedDocument:
    document: dict[str, Any]
    sources: tuple[Path, ...]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    cfg = OmegaConf.merge(OmegaConf.create(dict(base)), OmegaConf.create(dict(override)))
    return _plain_dict(cfg)


def dotlist_to_mapping(overrides: Sequence[str], *, label: str = "overrides") -> dict[str, Any]:
    cleaned = [str(item).strip() for item in overrides if str(item).strip()]
    if not cleaned:
        return {}
    try:
        cfg = OmegaConf.from_dotlist(cleaned)
    except Exception as exc:
        raise ValueError(f"failed to parse {label}: {exc}") from exc
    return _plain_dict(cfg)


def apply_dotlist_overrides(
    document: Mapping[str, Any],
    overrides: Sequence[str],
    *,
    label: str = "overrides",
) -> dict[str, Any]:
    override_mapping = dotlist_to_mapping(overrides, label=label)
    if not override_mapping:
        return dict(document)
    return deep_merge(document, override_mapping)


def slugify_template_value(value: Any) -> str:
    chars: list[str] = []
    for char in str(value or "").strip().lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")


def _concrete_template_source(value: Any) -> str:
    text = str(value or "").strip()
    return "" if "{" in text or "}" in text else text


def _environment_mapping_from_document(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for candidate in (document, document.get("goal")):
        if not isinstance(candidate, Mapping):
            continue
        train = candidate.get("train")
        if isinstance(train, Mapping) and isinstance(train.get("environment"), Mapping):
            return train["environment"]
        if isinstance(candidate.get("environment"), Mapping):
            return candidate["environment"]
    return None


def _environment_template_context_from_document(document: Mapping[str, Any]) -> dict[str, str]:
    environment = _environment_mapping_from_document(document)
    if not isinstance(environment, Mapping):
        return {}
    env_provider = _concrete_template_source(environment.get("env_provider"))
    env_id = _concrete_template_source(environment.get("env_id"))
    if env_id:
        if ":" in env_id:
            provider, provider_env_id = env_id.split(":", 1)
            return {"env_provider": provider, "env_id": provider_env_id}
        context = {"env_id": env_id}
        if env_provider:
            context["env_provider"] = env_provider
        return context
    env_config = environment.get("env_config")
    game = (
        _concrete_template_source(env_config.get("game"))
        if isinstance(env_config, Mapping)
        else ""
    )
    if not game and isinstance(env_config, Mapping):
        env_args = env_config.get("env_args")
        if isinstance(env_args, Mapping):
            game = _concrete_template_source(env_args.get("game"))
    context = {}
    if env_provider:
        context["env_provider"] = env_provider
    if game:
        context["env_id"] = game
    return context


def template_context_from_path(
    path: Path, document: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Build stable template variables from a goal/recipe path and optional document."""

    resolved = path.resolve()
    goal_id = ""
    game = ""
    recipe_slug = ""
    if resolved.parent.name == "recipes":
        recipe_slug = resolved.stem
        goal_id = resolved.parent.parent.name
        game = (
            resolved.parent.parent.parent.name
            if resolved.parent.parent.parent.name != "goals"
            else ""
        )
    else:
        goal_id = resolved.parent.name
        game = resolved.parent.parent.name if resolved.parent.parent.name != "goals" else ""

    if isinstance(document, Mapping):
        environment_context = _environment_template_context_from_document(document)
        raw_goal = document.get("goal")
        if isinstance(raw_goal, Mapping):
            goal_id = (
                _concrete_template_source(raw_goal.get("goal_id"))
                or _concrete_template_source(raw_goal.get("goal"))
                or goal_id
            )
        elif isinstance(raw_goal, str) and raw_goal.strip():
            goal_id = _concrete_template_source(raw_goal) or goal_id
        goal_id = (
            _concrete_template_source(document.get("goal_id"))
            or _concrete_template_source(document.get("goal_slug"))
            or goal_id
        )
        recipe_slug = (
            _concrete_template_source(document.get("recipe_id"))
            or recipe_slug
        )

    game_slug = slugify_template_value(game)
    goal_slug = slugify_template_value(goal_id)
    level_match = _LEVEL_ID_RE.match(goal_id)
    level_short = (
        f"l{level_match.group('world')}{level_match.group('level')}" if level_match else goal_slug
    )
    return {
        key: value
        for key, value in {
            "env_id": environment_context.get("env_id", "") if isinstance(document, Mapping) else "",
            "env_provider": environment_context.get("env_provider", "")
            if isinstance(document, Mapping)
            else "",
            "game": game,
            "game_slug": game_slug,
            "goal_id": goal_id,
            "goal_slug": goal_id,
            "level_short": level_short,
            "level_tag": goal_slug,
            "slug": recipe_slug,
            "recipe_id": recipe_slug,
            "recipe_slug": recipe_slug,
        }.items()
        if value
    }


def _template_field_root(field_name: str) -> str:
    return field_name.split(".", 1)[0].split("[", 1)[0]


def _referenced_template_fields(value: Any) -> set[str]:
    if isinstance(value, str):
        fields: set[str] = set()
        for _, field_name, format_spec, _ in Formatter().parse(value):
            if field_name is not None:
                fields.add(_template_field_root(field_name))
                fields.update(_referenced_template_fields(format_spec))
        return fields
    if isinstance(value, Mapping):
        return set().union(
            *(
                _referenced_template_fields(nested)
                for key, nested in value.items()
                if key != TEMPLATE_VARS_KEY
            ),
            set(),
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return set().union(*map(_referenced_template_fields, value), set())
    return set()


def _validate_template_var_usage(document: Mapping[str, Any], *, label: str) -> None:
    raw_vars = document.get(TEMPLATE_VARS_KEY)
    if not isinstance(raw_vars, Mapping):
        return
    declared = {key for key in raw_vars if isinstance(key, str)}
    used = declared & _referenced_template_fields(document)
    pending = list(used)
    while pending:
        key = pending.pop()
        dependencies = declared & _referenced_template_fields(raw_vars[key])
        new_dependencies = dependencies - used
        used.update(new_dependencies)
        pending.extend(new_dependencies)
    unused = sorted(declared - used)
    if unused:
        raise ValueError(
            f"{label}.{TEMPLATE_VARS_KEY} declares unused fields: {', '.join(unused)}"
        )


def _template_vars_from_document(
    document: Mapping[str, Any],
    *,
    base_context: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    raw_vars = document.get(TEMPLATE_VARS_KEY)
    if raw_vars is None:
        return {}
    if not isinstance(raw_vars, Mapping):
        raise ValueError(f"{label}.{TEMPLATE_VARS_KEY} must be an object")
    rendered: dict[str, str] = {}
    for key, value in raw_vars.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label}.{TEMPLATE_VARS_KEY} keys must be non-empty strings")
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"{label}.{TEMPLATE_VARS_KEY}.{key} must be a scalar")
        text = str(value)
        if isinstance(value, str):
            text = _render_template_string(
                value,
                context={**base_context, **rendered},
                deferred_fields=frozenset(),
                label=f"{label}.{TEMPLATE_VARS_KEY}.{key}",
            )
        rendered[key] = text
    return rendered


def _format_deferred_field(field_name: str, conversion: str | None, format_spec: str) -> str:
    text = "{" + field_name
    if conversion:
        text += f"!{conversion}"
    if format_spec:
        text += f":{format_spec}"
    return text + "}"


def _apply_conversion(value: Any, conversion: str | None) -> Any:
    if conversion == "s":
        return str(value)
    if conversion == "r":
        return repr(value)
    if conversion == "a":
        return ascii(value)
    if conversion:
        raise ValueError(f"unsupported template conversion: !{conversion}")
    return value


def _render_template_string(
    value: str,
    *,
    context: Mapping[str, Any],
    deferred_fields: frozenset[str],
    label: str,
) -> str:
    chunks: list[str] = []
    try:
        parsed = list(Formatter().parse(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid format template: {exc}") from exc
    for literal_text, field_name, format_spec, conversion in parsed:
        chunks.append(literal_text)
        if field_name is None:
            continue
        root_name = _template_field_root(field_name)
        if root_name in deferred_fields:
            chunks.append(_format_deferred_field(field_name, conversion, format_spec))
        elif root_name in context:
            rendered_format_spec = (
                _render_template_string(
                    format_spec,
                    context=context,
                    deferred_fields=deferred_fields,
                    label=f"{label} format spec",
                )
                if format_spec
                else ""
            )
            chunks.append(
                format(_apply_conversion(context[root_name], conversion), rendered_format_spec)
            )
        else:
            allowed = ", ".join(sorted({*context, *deferred_fields}))
            raise ValueError(
                f"{label} uses unknown template field {root_name!r}; allowed: {allowed}"
            )
    return "".join(chunks)


def validate_template_string(
    value: str,
    *,
    allowed_values: Mapping[str, Any],
    required_fields: frozenset[str] = frozenset(),
    label: str,
) -> frozenset[str]:
    try:
        parsed = list(Formatter().parse(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid format template: {exc}") from exc
    fields = frozenset(
        _template_field_root(field_name)
        for _literal, field_name, _format_spec, _conversion in parsed
        if field_name
    )
    unknown = sorted(fields - set(allowed_values))
    if unknown:
        raise ValueError(f"{label} uses unsupported template field(s): {', '.join(unknown)}")
    missing = sorted(required_fields - fields)
    if missing:
        raise ValueError(f"{label} must include template field(s): {', '.join(missing)}")
    _render_template_string(
        value,
        context=allowed_values,
        deferred_fields=frozenset(),
        label=label,
    )
    return fields


def _render_template_value(
    value: Any,
    *,
    context: Mapping[str, Any],
    deferred_fields_by_path: Mapping[tuple[str, ...], frozenset[str]],
    path: tuple[str, ...],
    label: str,
) -> Any:
    if isinstance(value, str):
        deferred_fields = deferred_fields_by_path.get(path, frozenset())
        return _render_template_string(
            value,
            context=context,
            deferred_fields=deferred_fields,
            label=label,
        )
    if isinstance(value, Mapping):
        return {
            key: _render_template_value(
                nested,
                context=context,
                deferred_fields_by_path=deferred_fields_by_path,
                path=(*path, str(key)),
                label=f"{label}.{key}",
            )
            for key, nested in value.items()
            if key != TEMPLATE_VARS_KEY
        }
    if isinstance(value, list):
        return [
            _render_template_value(
                item,
                context=context,
                deferred_fields_by_path=deferred_fields_by_path,
                path=(*path, str(index)),
                label=f"{label}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value


def render_template_vars(
    document: Mapping[str, Any],
    *,
    path: Path,
    label: str,
    extra_context: Mapping[str, Any] | None = None,
    deferred_fields_by_path: Mapping[tuple[str, ...], frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Render checked-in `{}` template variables and remove `template_vars`.

    This is intentionally stricter than OmegaConf interpolation: unknown fields fail
    unless the caller marks a specific document path as a deferred runtime template.
    """

    base_context = {
        **template_context_from_path(path, document),
        **{key: str(value) for key, value in (extra_context or {}).items()},
    }
    _validate_template_var_usage(document, label=label)
    template_vars = _template_vars_from_document(document, base_context=base_context, label=label)
    context = {**base_context, **template_vars}
    return _render_template_value(
        dict(document),
        context=context,
        deferred_fields_by_path=deferred_fields_by_path or {},
        path=(),
        label=label,
    )


def load_config_document(path: Path, *, default: Any = None) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in YAML_EXTENSIONS:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return default if loaded is None else loaded


def load_mapping_document(path: Path, *, label: str | None = None) -> dict[str, Any]:
    payload = load_config_document(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label or path} must contain a JSON/YAML object")
    return dict(payload)


def _plain_dict(value: Any) -> dict[str, Any]:
    payload = OmegaConf.to_container(value, resolve=False)
    if not isinstance(payload, Mapping):
        raise ValueError("composed config must contain a JSON/YAML object")
    return dict(payload)


def _default_entry_to_path(entry: Any) -> str | None:
    if isinstance(entry, str):
        if entry == "_self_" or entry.startswith("override ") or entry.startswith("optional "):
            return None
        return entry.split("@", 1)[0]
    if isinstance(entry, Mapping) and len(entry) == 1:
        key, value = next(iter(entry.items()))
        if key is None or key == "_self_":
            return None
        key = str(key)
        if key.startswith("override ") or key.startswith("optional "):
            return None
        if value in (None, "null"):
            return None
        if isinstance(value, str):
            return f"{key.split('@', 1)[0]}/{value.split('@', 1)[0]}"
    return None


def _resolve_default_path(default_path: str, *, base_dir: Path, config_root: Path) -> Path:
    is_absolute_default = default_path.startswith("/")
    path = default_path.lstrip("/")
    if path.endswith(".yaml") or path.endswith(".yml") or path.endswith(".json"):
        candidate = Path(path)
    else:
        candidate = Path(f"{path}.yaml")
    if not candidate.is_absolute():
        candidate = (config_root if is_absolute_default else base_dir) / candidate
    return candidate.resolve()


def _collect_hydra_sources(
    path: Path,
    *,
    config_root: Path,
    stack: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"cyclic Hydra defaults chain: {chain}")

    document = load_mapping_document(resolved_path, label=str(path))
    sources: list[Path] = []
    for entry in document.get("defaults", []) or []:
        default_path = _default_entry_to_path(entry)
        if default_path is None:
            continue
        source = _resolve_default_path(
            default_path,
            base_dir=resolved_path.parent,
            config_root=config_root,
        )
        if source.is_file():
            sources.extend(
                _collect_hydra_sources(
                    source,
                    config_root=config_root,
                    stack=(*stack, resolved_path),
                )
            )

    sources.append(resolved_path)
    return tuple(sources)


def load_composed_mapping(
    path: Path,
    *,
    stack: tuple[Path, ...] = (),
    cycle_label: str = "config",
    overrides: Sequence[str] = (),
) -> ComposedDocument:
    resolved_path = path.resolve()
    if stack:
        raise ValueError("load_composed_mapping no longer accepts recursive stack callers")
    if resolved_path.suffix.lower() not in YAML_EXTENSIONS:
        document = load_mapping_document(resolved_path, label=str(path))
        return ComposedDocument(
            document=apply_dotlist_overrides(
                document,
                overrides,
                label=f"{cycle_label} overrides for {path}",
            ),
            sources=(resolved_path,),
        )
    try:
        sources = _collect_hydra_sources(resolved_path, config_root=resolved_path.parent)
        with initialize_config_dir(version_base=None, config_dir=str(resolved_path.parent)):
            cfg = compose(config_name=resolved_path.stem)
    except Exception as exc:
        raise ValueError(f"failed to compose {cycle_label} config {path}: {exc}") from exc
    document = apply_dotlist_overrides(
        _plain_dict(cfg),
        overrides,
        label=f"{cycle_label} overrides for {path}",
    )
    return ComposedDocument(document=document, sources=sources)
