from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from huggingface_hub.utils import validate_repo_id

from rlab.env_registry import game_family_for_environment
from rlab.targets import target_for_game


HUGGINGFACE_NAMESPACE = "rlab-research"
REPO_NAMING_SCHEMA_VERSION = 1
RELEASE_MANIFEST_VERSION = 1
HUGGINGFACE_RELEASE_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "model.zip",
        "model_metadata.json",
        "release_manifest.json",
        "replay.mp4",
    }
)
HASHED_RELEASE_FILES = HUGGINGFACE_RELEASE_FILES - {"release_manifest.json"}
GITATTRIBUTES_TEXT = """*.zip filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
"""
MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 Tiago Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

ALGORITHM_MODEL_CLASSES: dict[str, frozenset[str]] = {
    "ppo": frozenset(
        {
            "stable_baselines3.ppo.ppo.PPO",
            "rlab.task_advantage.PerTaskAdvantagePPO",
        }
    ),
    "a2c": frozenset({"stable_baselines3.a2c.a2c.A2C"}),
    "dqn": frozenset({"stable_baselines3.dqn.dqn.DQN"}),
    "recurrent-ppo": frozenset(
        {"sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO"}
    ),
}


@dataclass(frozen=True)
class PublicationIdentity:
    game_family: str
    goal: str
    policy_variant: str
    algorithm: str

    @property
    def repo_name(self) -> str:
        return "_".join(asdict(self).values())


def normalize_publication_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise ValueError(f"{label} does not contain a valid repository-name component")
    if "_" in normalized:
        raise AssertionError("publication component normalization retained an underscore")
    return normalized


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _provider_and_environment(qualified_env_id: object) -> tuple[str, str]:
    value = str(qualified_env_id or "").strip()
    if ":" not in value:
        raise ValueError("model metadata environment.env_id must be provider-qualified")
    provider, environment = value.split(":", 1)
    if not provider or not environment:
        raise ValueError("model metadata environment.env_id must be provider-qualified")
    return provider, environment


def normalize_algorithm_id(value: object) -> str:
    algorithm = normalize_publication_component(value, label="algorithm").lower()
    if algorithm not in ALGORITHM_MODEL_CLASSES:
        known = ", ".join(sorted(ALGORITHM_MODEL_CLASSES))
        raise ValueError(f"unsupported publication algorithm {algorithm!r}; known: {known}")
    return algorithm


def validate_algorithm_model_class(algorithm: str, model_class: object) -> str:
    class_name = str(model_class or "").strip()
    if not class_name:
        raise ValueError("model metadata model_class is required for publication")
    allowed = ALGORITHM_MODEL_CLASSES[algorithm]
    if class_name not in allowed:
        raise ValueError(
            f"model class {class_name!r} is incompatible with algorithm {algorithm!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return class_name


def _observation_shape(preprocessing: Mapping[str, Any]) -> tuple[int, int]:
    resize = preprocessing.get("obs_resize")
    if not isinstance(resize, Sequence) or isinstance(resize, str | bytes) or len(resize) != 2:
        raise ValueError("publication preprocessing.obs_resize must contain height and width")
    height, width = (int(resize[0]), int(resize[1]))
    if height <= 0 or width <= 0:
        raise ValueError("publication observation dimensions must be positive")
    return height, width


def _view_component(
    preprocessing: Mapping[str, Any],
    *,
    game: str,
) -> str:
    raw_crop = preprocessing.get("obs_crop")
    if raw_crop is None:
        return "full"
    if (
        not isinstance(raw_crop, Sequence)
        or isinstance(raw_crop, str | bytes)
        or len(raw_crop) != 4
    ):
        raise ValueError("publication preprocessing.obs_crop must contain top,right,bottom,left")
    crop = tuple(int(value) for value in raw_crop)
    if any(value < 0 for value in crop):
        raise ValueError("publication crop values must be non-negative")
    if not any(crop):
        return "full"
    mode = str(preprocessing.get("obs_crop_mode") or "remove")
    if mode not in {"mask", "remove"}:
        raise ValueError(f"unsupported publication crop mode {mode!r}")
    default_hud_top = int(target_for_game(game).default_hud_crop_top)
    if default_hud_top > 0 and crop == (default_hud_top, 0, 0, 0):
        return "hudmask" if mode == "mask" else "hudcrop"
    prefix = "mask" if mode == "mask" else "crop"
    top, right, bottom, left = crop
    return f"{prefix}-t{top}-r{right}-b{bottom}-l{left}"


def policy_variant_from_contract(
    preprocessing: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    game: str,
) -> str:
    height, width = _observation_shape(preprocessing)
    grayscale = preprocessing.get("obs_grayscale")
    if not isinstance(grayscale, bool):
        raise ValueError("publication preprocessing.obs_grayscale must be boolean")
    color = "gray" if grayscale else "rgb"
    dimensions = str(height) if height == width else f"{height}x{width}"
    components = [f"{color}{dimensions}", _view_component(preprocessing, game=game)]

    frame_stack = int(preprocessing.get("frame_stack") or 0)
    if frame_stack <= 0:
        raise ValueError("publication preprocessing.frame_stack must be positive")
    components.append(f"stack{frame_stack}")

    layout = str(preprocessing.get("policy_observation_layout") or "")
    if layout == "dict_image_task":
        components.append("taskdict")
    elif layout != "channel_first":
        raise ValueError(f"unsupported policy observation layout {layout!r}")

    action = _require_mapping(task.get("action"), label="publication task.action")
    action_set = str(action.get("set") or "").strip()
    target_for_game(game).action_names_for_set(action_set)
    components.append(normalize_publication_component(action_set, label="publication action set"))
    return "-".join(components)


def publication_identity_from_model_metadata(
    goal_id: object,
    model_metadata: Mapping[str, Any],
) -> PublicationIdentity:
    training = _require_mapping(
        model_metadata.get("training_metadata"), label="model metadata training_metadata"
    )
    environment = _require_mapping(
        training.get("environment"), label="model metadata training_metadata.environment"
    )
    provider, game = _provider_and_environment(environment.get("env_id"))
    family = game_family_for_environment(provider, game, strict=True)
    preprocessing = _require_mapping(
        training.get("preprocessing"),
        label="model metadata training_metadata.preprocessing",
    )
    task = _require_mapping(environment.get("task"), label="model metadata environment.task")
    algorithm = normalize_algorithm_id(model_metadata.get("algorithm_id"))
    validate_algorithm_model_class(algorithm, model_metadata.get("model_class"))
    return PublicationIdentity(
        game_family=normalize_publication_component(family, label="game family"),
        goal=normalize_publication_component(goal_id, label="goal id"),
        policy_variant=policy_variant_from_contract(preprocessing, task, game=game),
        algorithm=algorithm,
    )


def upgrade_legacy_model_metadata_for_publication(
    model_metadata: Mapping[str, Any],
    *,
    algorithm_id: str,
    model_class: str,
    crop_mode: str,
) -> dict[str, Any]:
    """Make an older rlab metadata document explicit enough to derive its identity.

    Legacy metadata did not own algorithm identity, nested its action contract under
    ``environment.action``, and omitted whether ``obs_crop`` removed or masked pixels.
    Callers must supply those missing facts from inspected checkpoint/repository and
    source-history evidence; this helper deliberately does not guess them.
    """

    result = deepcopy(dict(model_metadata))
    algorithm = normalize_algorithm_id(algorithm_id)
    class_name = validate_algorithm_model_class(algorithm, model_class)
    if crop_mode not in {"mask", "remove"}:
        raise ValueError("legacy crop_mode must be 'mask' or 'remove'")

    training = _require_mapping(
        result.get("training_metadata"), label="legacy model metadata training_metadata"
    )
    training = deepcopy(dict(training))
    environment = _require_mapping(
        training.get("environment"), label="legacy model metadata environment"
    )
    environment = deepcopy(dict(environment))
    preprocessing = _require_mapping(
        training.get("preprocessing"), label="legacy model metadata preprocessing"
    )
    preprocessing = deepcopy(dict(preprocessing))

    if preprocessing.get("obs_crop_mode") not in {None, crop_mode}:
        raise ValueError(
            "legacy metadata crop mode conflicts with the explicitly supplied crop mode"
        )
    preprocessing["obs_crop_mode"] = crop_mode

    task = environment.get("task")
    if task is None:
        legacy_action = environment.get("action")
        action_set = legacy_action.get("action_set") if isinstance(legacy_action, Mapping) else None
        if not action_set:
            env_config = training.get("env_config")
            action_set = env_config.get("action_set") if isinstance(env_config, Mapping) else None
        if not action_set:
            raise ValueError("legacy metadata does not contain an action-set contract")
        environment["task"] = {"action": {"set": action_set}}
    else:
        _require_mapping(task, label="legacy model metadata environment.task")

    training["environment"] = environment
    training["preprocessing"] = preprocessing
    result["training_metadata"] = training
    result["algorithm_id"] = algorithm
    result["model_class"] = class_name
    return result


def build_model_repo_id(identity: PublicationIdentity) -> str:
    for field, value in asdict(identity).items():
        normalized = normalize_publication_component(value, label=field)
        if normalized != value:
            raise ValueError(
                f"publication identity {field} must already be canonical: "
                f"expected {normalized!r}, got {value!r}"
            )
    repo_id = f"{HUGGINGFACE_NAMESPACE}/{identity.repo_name}"
    validate_repo_id(repo_id)
    if len(identity.repo_name) > 96:
        raise ValueError("Hugging Face repository name exceeds 96 characters")
    return repo_id


def assert_unique_repo_ids(identities: Sequence[PublicationIdentity]) -> None:
    seen: dict[str, PublicationIdentity] = {}
    for identity in identities:
        repo_id = build_model_repo_id(identity)
        previous = seen.get(repo_id)
        if previous is not None and previous != identity:
            raise ValueError(
                f"publication identities normalize to the same repository {repo_id!r}: "
                f"{previous!r} and {identity!r}"
            )
        seen[repo_id] = identity


def assert_unique_goal_slugs(goal_ids: Sequence[object]) -> None:
    seen: dict[str, str] = {}
    for value in goal_ids:
        raw = str(value or "").strip()
        slug = normalize_publication_component(raw, label="goal id")
        previous = seen.get(slug)
        if previous is not None and previous != raw:
            raise ValueError(
                f"goal ids {previous!r} and {raw!r} normalize to the same publication "
                f"component {slug!r}"
            )
        seen[slug] = raw


def publication_model_metadata(
    model_metadata: Mapping[str, Any],
    identity: PublicationIdentity,
) -> dict[str, Any]:
    result = deepcopy(dict(model_metadata))
    result["publication"] = {
        "repo_naming_schema": REPO_NAMING_SCHEMA_VERSION,
        "repo_id": build_model_repo_id(identity),
        **asdict(identity),
    }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_artifact_records(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        filename: {
            "sha256": sha256_file(root / filename),
            "size_bytes": (root / filename).stat().st_size,
        }
        for filename in sorted(HASHED_RELEASE_FILES)
    }


def _assert_no_absolute_paths(value: object, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_no_absolute_paths(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _assert_no_absolute_paths(nested, path=f"{path}[{index}]")
        return
    if not isinstance(value, str) or not value:
        return
    if value.startswith(("http://", "https://", "hf://", "s3://", "r2://")):
        return
    if value.startswith("file://") or Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{path} contains an absolute local path")


def build_release_manifest(
    identity: PublicationIdentity,
    model_metadata: Mapping[str, Any],
    *,
    release_version: str,
    published_at: str,
    source: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    youtube_url: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"v[1-9][0-9]*", release_version):
        raise ValueError("release_version must be a sequential tag such as v1 or v2")
    expected_identity = publication_identity_from_model_metadata(identity.goal, model_metadata)
    if expected_identity != identity:
        raise ValueError("release identity does not match model metadata")
    training = _require_mapping(model_metadata.get("training_metadata"), label="training_metadata")
    environment = _require_mapping(training.get("environment"), label="training environment")
    manifest: dict[str, Any] = {
        "manifest_version": RELEASE_MANIFEST_VERSION,
        "repo_naming_schema": REPO_NAMING_SCHEMA_VERSION,
        "repository": {"repo_id": build_model_repo_id(identity), **asdict(identity)},
        "release": {"version": release_version, "published_at": published_at},
        "model": {
            "algorithm_id": model_metadata["algorithm_id"],
            "model_class": model_metadata["model_class"],
            "qualified_env_id": environment.get("env_id"),
            "environment_hash": training.get("environment_hash"),
            "preprocessing": training.get("preprocessing"),
            "action": _require_mapping(environment.get("task"), label="environment task").get(
                "action"
            ),
        },
        "source": dict(source),
        "evaluation": dict(evaluation),
        "artifacts": dict(artifacts),
    }
    if youtube_url:
        manifest["release"]["youtube_url"] = youtube_url
    _assert_no_absolute_paths(manifest)
    return manifest


def validate_release_bundle(root: Path) -> dict[str, Any]:
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != HUGGINGFACE_RELEASE_FILES:
        missing = sorted(HUGGINGFACE_RELEASE_FILES - actual_entries)
        extra = sorted(actual_entries - HUGGINGFACE_RELEASE_FILES)
        raise ValueError(f"release file set mismatch; missing={missing}, extra={extra}")
    non_files = sorted(path.name for path in root.iterdir() if not path.is_file())
    if non_files:
        raise ValueError(f"release entries must all be regular files: {non_files}")
    model_metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    _assert_no_absolute_paths(model_metadata, path="model_metadata")
    _assert_no_absolute_paths(manifest)
    repository = _require_mapping(manifest.get("repository"), label="manifest repository")
    identity = publication_identity_from_model_metadata(repository.get("goal"), model_metadata)
    if repository.get("repo_id") != build_model_repo_id(identity):
        raise ValueError("release manifest repository id does not match model metadata")
    if int(manifest.get("repo_naming_schema") or 0) != REPO_NAMING_SCHEMA_VERSION:
        raise ValueError("release manifest has an unsupported repository naming schema")
    expected_records = release_artifact_records(root)
    if manifest.get("artifacts") != expected_records:
        raise ValueError("release manifest artifact hashes or sizes do not match the bundle")
    return manifest
