from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rlab.config_loader import load_mapping_document


DEFAULT_MODAL_EVAL_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{label} must be at least 1")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True)
class ModalEvalConfig:
    enabled: bool
    app_name_prefix: str
    function_name: str
    hard_max_active: int
    initial_effective_capacity: int
    per_run_budget_usd: float
    rolling_24h_budget_usd: float
    cpu: float
    memory_mib: int
    min_containers: int
    buffer_containers: int
    max_containers: int
    single_use_containers: bool
    scaledown_window_seconds: int
    startup_timeout_seconds: int
    screen_timeout_seconds: int
    confirm_timeout_seconds: int
    promotion_timeout_seconds: int
    child_margin_seconds: int
    expiry_margin_seconds: int
    estimated_hourly_usd: float
    schema_version: int
    seed_protocol: str
    max_attempts: int
    preview_enabled: bool
    preview_max_frames: int
    preview_fps: int
    preview_max_lanes: int
    preview_scale: int
    preview_max_bytes: int
    preview_encode_timeout_seconds: int
    preview_upload_timeout_seconds: int

    def timeout_for(self, purpose: str, stage_index: int) -> int:
        if purpose == "promotion":
            return self.promotion_timeout_seconds
        if stage_index > 0:
            return self.confirm_timeout_seconds
        return self.screen_timeout_seconds

    def reserved_cost(self, timeout_seconds: int) -> float:
        return self.estimated_hourly_usd * float(timeout_seconds) / 3600.0


def load_modal_eval_config(path: Path = DEFAULT_MODAL_EVAL_CONFIG) -> ModalEvalConfig:
    document = load_mapping_document(path, label=str(path))
    allowed = {
        "enabled",
        "deployment",
        "limits",
        "resources",
        "timeouts",
        "cost",
        "protocol",
        "preview",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(unknown)}")
    deployment = _mapping(document.get("deployment"), label="deployment")
    limits = _mapping(document.get("limits"), label="limits")
    resources = _mapping(document.get("resources"), label="resources")
    timeouts = _mapping(document.get("timeouts"), label="timeouts")
    cost = _mapping(document.get("cost"), label="cost")
    protocol = _mapping(document.get("protocol"), label="protocol")
    preview = _mapping(document.get("preview"), label="preview")
    sections = {
        "deployment": (deployment, {"app_name_prefix", "function_name"}),
        "limits": (
            limits,
            {
                "hard_max_active",
                "initial_effective_capacity",
                "per_run_budget_usd",
                "rolling_24h_budget_usd",
            },
        ),
        "resources": (
            resources,
            {
                "cpu",
                "memory_mib",
                "min_containers",
                "buffer_containers",
                "max_containers",
                "single_use_containers",
                "scaledown_window_seconds",
                "startup_timeout_seconds",
            },
        ),
        "timeouts": (
            timeouts,
            {
                "screen_seconds",
                "confirm_seconds",
                "promotion_seconds",
                "child_margin_seconds",
                "expiry_margin_seconds",
            },
        ),
        "cost": (cost, {"estimated_hourly_usd"}),
        "protocol": (protocol, {"schema_version", "seed_protocol", "max_attempts"}),
        "preview": (
            preview,
            {
                "enabled",
                "max_frames",
                "fps",
                "max_lanes",
                "scale",
                "max_bytes",
                "encode_timeout_seconds",
                "upload_timeout_seconds",
            },
        ),
    }
    for section_name, (section, section_allowed) in sections.items():
        section_unknown = sorted(set(section) - section_allowed)
        if section_unknown:
            raise ValueError(
                f"{path} {section_name} has unknown field(s): {', '.join(section_unknown)}"
            )
    hard_cap = _positive_int(limits.get("hard_max_active"), label="limits.hard_max_active")
    effective = _positive_int(
        limits.get("initial_effective_capacity"), label="limits.initial_effective_capacity"
    )
    modal_cap = _positive_int(resources.get("max_containers"), label="resources.max_containers")
    if hard_cap != modal_cap:
        raise ValueError("limits.hard_max_active must equal resources.max_containers")
    if effective > hard_cap:
        raise ValueError("limits.initial_effective_capacity exceeds the hard cap")
    max_attempts = _positive_int(protocol.get("max_attempts"), label="protocol.max_attempts")
    if max_attempts > 2:
        raise ValueError("protocol.max_attempts must not exceed 2")
    prefix = str(deployment.get("app_name_prefix") or "").strip()
    function_name = str(deployment.get("function_name") or "").strip()
    seed_protocol = str(protocol.get("seed_protocol") or "").strip()
    if not prefix or not function_name or not seed_protocol:
        raise ValueError("deployment names and protocol.seed_protocol must be non-empty")
    result = ModalEvalConfig(
        enabled=_bool(document.get("enabled", False), label="enabled"),
        app_name_prefix=prefix,
        function_name=function_name,
        hard_max_active=hard_cap,
        initial_effective_capacity=effective,
        per_run_budget_usd=_positive_float(
            limits.get("per_run_budget_usd"), label="limits.per_run_budget_usd"
        ),
        rolling_24h_budget_usd=_positive_float(
            limits.get("rolling_24h_budget_usd"), label="limits.rolling_24h_budget_usd"
        ),
        cpu=_positive_float(resources.get("cpu"), label="resources.cpu"),
        memory_mib=_positive_int(resources.get("memory_mib"), label="resources.memory_mib"),
        min_containers=_nonnegative_int(
            resources.get("min_containers"), label="resources.min_containers"
        ),
        buffer_containers=_nonnegative_int(
            resources.get("buffer_containers"), label="resources.buffer_containers"
        ),
        max_containers=modal_cap,
        single_use_containers=_bool(
            resources.get("single_use_containers"),
            label="resources.single_use_containers",
        ),
        scaledown_window_seconds=_positive_int(
            resources.get("scaledown_window_seconds"),
            label="resources.scaledown_window_seconds",
        ),
        startup_timeout_seconds=_positive_int(
            resources.get("startup_timeout_seconds"), label="resources.startup_timeout_seconds"
        ),
        screen_timeout_seconds=_positive_int(
            timeouts.get("screen_seconds"), label="timeouts.screen_seconds"
        ),
        confirm_timeout_seconds=_positive_int(
            timeouts.get("confirm_seconds"), label="timeouts.confirm_seconds"
        ),
        promotion_timeout_seconds=_positive_int(
            timeouts.get("promotion_seconds"), label="timeouts.promotion_seconds"
        ),
        child_margin_seconds=_positive_int(
            timeouts.get("child_margin_seconds"), label="timeouts.child_margin_seconds"
        ),
        expiry_margin_seconds=_positive_int(
            timeouts.get("expiry_margin_seconds"), label="timeouts.expiry_margin_seconds"
        ),
        estimated_hourly_usd=_positive_float(
            cost.get("estimated_hourly_usd"), label="cost.estimated_hourly_usd"
        ),
        schema_version=_positive_int(protocol.get("schema_version"), label="protocol.schema_version"),
        seed_protocol=seed_protocol,
        max_attempts=max_attempts,
        preview_enabled=_bool(preview.get("enabled", False), label="preview.enabled"),
        preview_max_frames=_positive_int(
            preview.get("max_frames"), label="preview.max_frames"
        ),
        preview_fps=_positive_int(preview.get("fps"), label="preview.fps"),
        preview_max_lanes=_positive_int(
            preview.get("max_lanes"), label="preview.max_lanes"
        ),
        preview_scale=_positive_int(preview.get("scale"), label="preview.scale"),
        preview_max_bytes=_positive_int(
            preview.get("max_bytes"), label="preview.max_bytes"
        ),
        preview_encode_timeout_seconds=_positive_int(
            preview.get("encode_timeout_seconds"),
            label="preview.encode_timeout_seconds",
        ),
        preview_upload_timeout_seconds=_positive_int(
            preview.get("upload_timeout_seconds"),
            label="preview.upload_timeout_seconds",
        ),
    )
    if result.child_margin_seconds >= min(
        result.screen_timeout_seconds,
        result.confirm_timeout_seconds,
        result.promotion_timeout_seconds,
    ):
        raise ValueError("timeouts.child_margin_seconds must be smaller than every eval timeout")
    if result.preview_max_frames > 450:
        raise ValueError("preview.max_frames must not exceed 450")
    if result.preview_fps > 15:
        raise ValueError("preview.fps must not exceed 15")
    if result.preview_max_lanes > 4:
        raise ValueError("preview.max_lanes must not exceed 4")
    if result.preview_scale > 2:
        raise ValueError("preview.scale must not exceed 2")
    if result.preview_max_bytes > 2 * 1024 * 1024:
        raise ValueError("preview.max_bytes must not exceed 2 MiB")
    if result.preview_encode_timeout_seconds + result.preview_upload_timeout_seconds > 5:
        raise ValueError("preview encode and upload timeouts must total no more than 5 seconds")
    return result


def modal_app_name(prefix: str, runtime_image_ref: str) -> str:
    digest = str(runtime_image_ref).rsplit("@sha256:", 1)[-1]
    if digest == runtime_image_ref or len(digest) < 12:
        raise ValueError("Modal eval runtime must be an immutable sha256 image ref")
    return f"{prefix}-{digest[:12]}"
