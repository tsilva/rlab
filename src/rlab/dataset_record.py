from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlab import __version__
from rlab.dataset_contract import (
    COLLECTOR_ARTIFACT_DIR,
    COLLECTION_DOCUMENT_TYPE,
    COLLECTION_FORMAT_VERSION,
    DATASET_FORMAT_VERSION,
    ENVIRONMENT_ARTIFACT_DIR,
    ENVIRONMENT_DOCUMENT_FILENAME,
    PRODUCER_PREFIX,
    STORAGE_FORMAT_LOSSLESS_VIDEO,
    VIDEO_ARTIFACT_DIR,
    canonical_column_order,
    canonical_json_bytes,
    dataset_fields,
    observation_to_rgb,
)
from rlab.dataset_media import LosslessVideoWriter, iter_recoverable_video_prefix
from rlab.dataset_providers import (
    EnvironmentArtifact,
    ProviderSession,
    build_environment_artifact,
    create_provider_session,
    space_contract,
    validate_provider_request,
)
from rlab.dataset_store import (
    adopt_tree,
    combine_trees,
    dataset_root,
    preflight_recording_reference,
    validate_tree,
)
from rlab.model_sources import (
    download_artifact_ref_source,
    download_huggingface_model_source,
    is_huggingface_model_ref,
    is_wandb_run_ref,
    model_ref_from_run_path,
)
from rlab.policy_models import load_policy_model
from rlab.trusted_inputs import ApprovedModelInput, stage_and_approve_model


JOURNAL_VERSION = 1
MANIFEST_VERSION = 1
MIN_WORKING_MARGIN_BYTES = 256 * 1024**2
MAX_JOURNAL_LINE_BYTES = 1 << 20
MAX_EPISODE_SEED = 2**32 - 1
ROW_CONTEXT_COLUMNS = (
    "episode_id",
    "seed",
    "session_id",
    "dataset_format_version",
    "collector",
    "gymrec_version",
    "storage_format",
    "provider_id",
    "env_id",
    "environment_contract_id",
    "collector_contract_id",
    "policy_mode",
    "policy_seed",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_value(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CollectorArtifact:
    contract_id: str
    document: Mapping[str, Any]
    files: Mapping[str, bytes]


class EpisodeJournal:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / "journal.jsonl"
        self._stream = self.path.open("xb")
        self._candidates: list[Path] = []

    def write_candidate(self, *, step: int, frame: np.ndarray) -> tuple[Path, str]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required; install Rlab with --extra dataset") from exc
        candidate = self.directory / f"candidate-{step:012d}.png"
        temporary = candidate.with_name(f".{candidate.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as output:
            Image.fromarray(frame, mode="RGB").save(output, format="PNG", compress_level=1)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, candidate)
        _fsync_directory(self.directory)
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        self._candidates.append(candidate)
        while len(self._candidates) > 2:
            self._candidates.pop(0).unlink(missing_ok=True)
            _fsync_directory(self.directory)
        return candidate, digest

    def append(self, value: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(value) + b"\n"
        if len(payload) > MAX_JOURNAL_LINE_BYTES:
            raise ValueError(f"journal record exceeds {MAX_JOURNAL_LINE_BYTES} bytes")
        self._stream.write(payload)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def finalize(self) -> None:
        self._stream.close()
        for candidate in self._candidates:
            candidate.unlink(missing_ok=True)
        _fsync_directory(self.directory)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


class HumanController:
    KEY_LABELS = {
        "K_UP": "UP",
        "K_DOWN": "DOWN",
        "K_LEFT": "LEFT",
        "K_RIGHT": "RIGHT",
        "K_z": "B",
        "K_x": "A",
        "K_RETURN": "START",
        "K_RSHIFT": "SELECT",
    }

    def __init__(self, session: ProviderSession, frame: np.ndarray, *, headless: bool) -> None:
        if headless:
            raise ValueError("human recording cannot be headless")
        from rlab.play import PygameViewer

        self.session = session
        self.viewer = PygameViewer(frame.shape, 3)
        self.fps = max(int(round(session.fps)), 1)
        self._started = False
        self._help_visible = True
        self._special_keys: set[int] = set()
        self._last_tick = time.monotonic()

    def _overlay(self) -> list[str] | None:
        if not self._help_visible:
            return None
        prefix = [] if self._started else ["Press SPACE to start"]
        return [
            *prefix,
            "Arrows move · Z=B · X=A · Enter=START · Right Shift=SELECT",
            "Tab help · +/- speed · Esc/Q stop",
            f"{self.fps} FPS",
        ]

    def _handle_special_keys(self, pressed: Any) -> None:
        pygame = self.viewer.pygame
        current = {
            key
            for key in (
                pygame.K_TAB,
                pygame.K_EQUALS,
                getattr(pygame, "K_PLUS", pygame.K_EQUALS),
                getattr(pygame, "K_KP_PLUS", pygame.K_EQUALS),
                pygame.K_MINUS,
                getattr(pygame, "K_KP_MINUS", pygame.K_MINUS),
            )
            if pressed[key]
        }
        newly_pressed = current - self._special_keys
        if pygame.K_TAB in newly_pressed:
            self._help_visible = not self._help_visible
        plus = {
            pygame.K_EQUALS,
            getattr(pygame, "K_PLUS", pygame.K_EQUALS),
            getattr(pygame, "K_KP_PLUS", pygame.K_EQUALS),
        }
        minus = {pygame.K_MINUS, getattr(pygame, "K_KP_MINUS", pygame.K_MINUS)}
        if newly_pressed.intersection(plus):
            self.fps += 5
        if newly_pressed.intersection(minus):
            self.fps = max(1, self.fps - 5)
        self._special_keys = current

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        while True:
            remaining = (1.0 / self.fps) - (time.monotonic() - self._last_tick)
            if remaining > 0:
                time.sleep(remaining)
            self._last_tick = time.monotonic()
            if not self.viewer.show(frame, overlay=self._overlay()):
                return None, False
            pressed = self.viewer.pygame.key.get_pressed()
            self._handle_special_keys(pressed)
            if self._started or pressed[self.viewer.pygame.K_SPACE]:
                self._started = True
                break
        labels = {
            label
            for key_name, label in self.KEY_LABELS.items()
            if pressed[getattr(self.viewer.pygame, key_name)]
        }
        return self.session.action_from_labels(labels), True

    def close(self) -> None:
        self.viewer.close()


def _current_tree_size(root: Path, reference: str) -> int:
    from rlab.dataset_store import collection_paths

    current = collection_paths(reference, root).current
    if not current.exists():
        return 0
    return sum(path.stat().st_size for path in current.rglob("*") if path.is_file())


def _has_headroom(work: Path, *, frame_bytes: int, projected_rebuild: int) -> bool:
    required = MIN_WORKING_MARGIN_BYTES + frame_bytes * 4 + projected_rebuild
    return shutil.disk_usage(work).free >= required


def _model_support_files(approved: ApprovedModelInput) -> dict[str, bytes]:
    available = {entry.path: approved.root / entry.path for entry in approved.staged.manifest}
    model = available.get("model.json") or next(
        (path for name, path in available.items() if name.endswith(".model.json")), None
    )
    recipe = available.get("recipe.json") or next(
        (path for name, path in available.items() if name.endswith(".recipe.json")), None
    )
    model_payload = (
        model.read_bytes()
        if model is not None
        else canonical_json_bytes(
            {
                "document_type": "rlab.dataset.local_model",
                "format_version": 1,
                "approval_manifest": approved.manifest_hash,
            }
        )
    )
    recipe_payload = (
        recipe.read_bytes()
        if recipe is not None
        else canonical_json_bytes(
            {
                "document_type": "rlab.dataset.explicit_environment",
                "format_version": 1,
            }
        )
    )
    files = {"model.json": model_payload, "recipe.json": recipe_payload}
    release = available.get("release_manifest.json")
    if release is not None:
        files["release_manifest.json"] = release.read_bytes()
    return files


def _collector_artifact(
    approved: ApprovedModelInput,
    *,
    environment_contract_id: str,
    deterministic: bool,
    session: ProviderSession,
) -> CollectorArtifact:
    files = _model_support_files(approved)
    source: dict[str, Any] = {
        "identity": approved.staged.source_identity or str(approved.staged.source),
        "approval_manifest": approved.manifest_hash,
        "model": {
            "filename": "model.json",
            "sha256": hashlib.sha256(files["model.json"]).hexdigest(),
        },
        "recipe": {
            "filename": "recipe.json",
            "sha256": hashlib.sha256(files["recipe.json"]).hexdigest(),
        },
        "checkpoint": {
            "filename": approved.model_path.name,
            "sha256": next(
                entry.sha256
                for entry in approved.staged.manifest
                if entry.path == approved.model_path.name
            ),
        },
    }
    if "release_manifest.json" in files:
        source["release_manifest"] = {
            "filename": "release_manifest.json",
            "sha256": hashlib.sha256(files["release_manifest.json"]).hexdigest(),
        }
    document = {
        "document_type": COLLECTION_DOCUMENT_TYPE,
        "format_version": COLLECTION_FORMAT_VERSION,
        "source": source,
        "policy": {
            "mode": "deterministic" if deterministic else "stochastic",
            "seed_derivation": {
                "protocol": "base-plus-episode-index-v1",
                "base_seed_stored": False,
                "episode_seed_column": "policy_seed",
            },
        },
        "execution": {
            "environment_contract_id": environment_contract_id,
            "policy_observations": "provider-selected",
            "recorded_observations": "provider-selected-rgb",
            "recorded_actions": "exact-env-step-input",
            "recorded_policy_actions": "unadapted-policy-output",
            "action_space": space_contract(session.env.action_space),
            "observation_space": space_contract(session.env.observation_space),
        },
        "runtime": {
            "inference_device": "auto",
            "packages": {
                name: _package_version(name)
                for name in ("rlab", "stable-baselines3", "torch", "numpy", session.provider_id)
            },
            "rlab_policy_adapter_version": 1,
        },
    }
    contract_id = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return CollectorArtifact(contract_id, document, files)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return __version__ if name == "rlab" else None


def _write_artifacts(
    tree: Path,
    *,
    environment: EnvironmentArtifact,
    collector: CollectorArtifact | None,
) -> None:
    environment_path = (
        tree / ENVIRONMENT_ARTIFACT_DIR / environment.contract_id / ENVIRONMENT_DOCUMENT_FILENAME
    )
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_bytes(canonical_json_bytes(environment.document))
    if collector is None:
        return
    collector_directory = tree / COLLECTOR_ARTIFACT_DIR / collector.contract_id
    collector_directory.mkdir(parents=True, exist_ok=True)
    (collector_directory / "collection.json").write_bytes(canonical_json_bytes(collector.document))
    for filename, payload in collector.files.items():
        (collector_directory / filename).write_bytes(payload)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _dataset_from_rows(rows: Sequence[Mapping[str, Any]]):
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "dataset support is not installed; run ./install.sh --extra dataset"
        ) from exc
    order = canonical_column_order(STORAGE_FORMAT_LOSSLESS_VIDEO)
    data = {name: [row[name] for row in rows] for name in order}
    dataset = datasets.Dataset.from_dict(data)
    for field in dataset_fields(STORAGE_FORMAT_LOSSLESS_VIDEO):
        if field.cast is not None:
            dataset = dataset.cast_column(field.name, datasets.Value(field.cast))
    return dataset


def _transition_row(
    *,
    context: Mapping[str, Any],
    step: int,
    action: Any,
    policy_action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    video_path: str,
    frame: np.ndarray,
) -> dict[str, Any]:
    return {
        **context,
        "step_index": step,
        "actions": _json_value(action),
        "policy_actions": _json_value(policy_action),
        "rewards": float(reward),
        "terminations": bool(terminated),
        "truncations": bool(truncated),
        "infos": json.dumps(
            _json_value(info), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "collector_terminated": False,
        "video_path": video_path,
        "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
    }


def _terminal_row(
    *,
    context: Mapping[str, Any],
    step: int,
    collector_terminated: bool,
    video_path: str,
    frame: np.ndarray,
) -> dict[str, Any]:
    return {
        **context,
        "step_index": step,
        "actions": None,
        "policy_actions": None,
        "rewards": None,
        "terminations": None,
        "truncations": None,
        "infos": None,
        "collector_terminated": bool(collector_terminated),
        "video_path": video_path,
        "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
    }


def _write_episode_tree(
    package: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    video_source: Path,
    video_relative: str,
    environment: EnvironmentArtifact,
    collector: CollectorArtifact | None,
) -> None:
    temporary = package.with_name(f".{package.name}.{uuid.uuid4().hex}.tmp")
    dataset = _dataset_from_rows(rows)
    try:
        dataset.save_to_disk(str(temporary))
        video_target = temporary.joinpath(*Path(video_relative).parts)
        video_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_source, video_target)
        _write_artifacts(temporary, environment=environment, collector=collector)
        validate_tree(temporary)
        os.replace(temporary, package)
        _fsync_directory(package.parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _record_one(
    *,
    session: ProviderSession,
    environment: EnvironmentArtifact,
    collector: CollectorArtifact | None,
    model: Any | None,
    agent: str,
    deterministic: bool,
    seed: int,
    episode_directory: Path,
    package: Path,
    session_id: str,
    headless: bool,
    projected_rebuild: int,
    ui: str = "pygame",
    no_open: bool = False,
    port: int = 0,
) -> str | None:
    episode_id = str(uuid.uuid4())
    policy_seed = seed
    video_relative = f"{VIDEO_ARTIFACT_DIR}/{episode_id}.rgb.mkv.bin"
    video_output = episode_directory / f"{episode_id}.rgb.mkv.bin"
    journal = EpisodeJournal(episode_directory / "active")
    writer = LosslessVideoWriter(video_output, fps=session.fps)
    controller: Any | None = None
    transitions: list[dict[str, Any]] = []
    try:
        _write_artifacts(journal.directory, environment=environment, collector=collector)
        _fsync_tree(journal.directory)
        observation, _reset_info = session.env.reset(seed=seed)
        frame = observation_to_rgb(session.recording_observation(observation))
        if not _has_headroom(
            episode_directory, frame_bytes=frame.nbytes, projected_rebuild=projected_rebuild
        ):
            return None
        writer.write(frame)
        if agent == "human":
            if ui == "web":
                from argparse import Namespace

                from rlab.play_web import WebHumanController

                controller = WebHumanController(
                    session,
                    Namespace(
                        fps=session.fps,
                        episodes=1,
                        port=port,
                        no_open=no_open,
                        debug=False,
                    ),
                )
            else:
                controller = HumanController(session, frame, headless=headless)
        elif agent == "random":
            session.env.action_space.seed(policy_seed)
        else:
            model.set_random_seed(policy_seed)
        context = {
            "episode_id": episode_id,
            "seed": seed,
            "session_id": session_id,
            "dataset_format_version": DATASET_FORMAT_VERSION,
            "collector": agent
            if collector is None
            else str(collector.document["source"]["identity"]),
            "gymrec_version": f"{PRODUCER_PREFIX}{__version__}",
            "storage_format": STORAGE_FORMAT_LOSSLESS_VIDEO,
            "provider_id": session.provider_id,
            "env_id": session.environment_id,
            "environment_contract_id": environment.contract_id,
            "collector_contract_id": collector.contract_id if collector else None,
            "policy_mode": (
                "manual"
                if agent == "human"
                else "random"
                if agent == "random"
                else "deterministic"
                if deterministic
                else "stochastic"
            ),
            "policy_seed": policy_seed,
        }
        provider_ended = False
        while not provider_ended:
            if not _has_headroom(
                episode_directory,
                frame_bytes=frame.nbytes,
                projected_rebuild=projected_rebuild + video_output.stat().st_size
                if video_output.exists()
                else projected_rebuild,
            ):
                if transitions:
                    break
                return None
            policy_action = None
            if agent == "human":
                assert controller is not None
                action, keep_recording = controller.action(frame)
                if not keep_recording:
                    break
            elif agent == "random":
                action = session.env.action_space.sample()
            else:
                policy_action, _state = model.predict(
                    session.policy_observation(observation), deterministic=deterministic
                )
                raw = np.asarray(policy_action)
                policy_action = int(raw.reshape(-1)[0]) if raw.size == 1 else raw.tolist()
                action = session.adapt_policy_action(policy_action)
            current_frame = frame
            next_observation, reward, terminated, truncated, info = session.env.step(action)
            next_frame = observation_to_rgb(session.recording_observation(next_observation))
            observe_transition = getattr(controller, "observe_transition", None)
            if callable(observe_transition):
                observe_transition(
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    next_frame=next_frame,
                )
            _candidate, next_hash = journal.write_candidate(
                step=len(transitions) + 1, frame=next_frame
            )
            writer.write(next_frame)
            transition = _transition_row(
                context=context,
                step=len(transitions),
                action=action,
                policy_action=policy_action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                video_path=video_relative,
                frame=current_frame,
            )
            journal.append(
                {
                    "version": JOURNAL_VERSION,
                    "episode_id": episode_id,
                    "step_index": len(transitions),
                    "row": transition,
                    "next_frame_sha256": next_hash,
                }
            )
            transitions.append(transition)
            observation = next_observation
            frame = next_frame
            provider_ended = bool(terminated) or bool(truncated)
        if not transitions:
            return None
        rows = [
            *transitions,
            _terminal_row(
                context=context,
                step=len(transitions),
                collector_terminated=not provider_ended,
                video_path=video_relative,
                frame=frame,
            ),
        ]
        writer.close()
        _write_episode_tree(
            package,
            rows=rows,
            video_source=video_output,
            video_relative=video_relative,
            environment=environment,
            collector=collector,
        )
        journal.finalize()
        return episode_id
    finally:
        if controller is not None:
            controller.close()
        journal.close()
        if not package.exists():
            writer.abort(preserve=journal.path.is_file() and journal.path.stat().st_size > 0)
        session.env.close()


def _session_manifest(path: Path, *, reference: str, session_id: str) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "reference": reference,
        "session_id": session_id,
        "state": "recording",
        "packages": [],
    }


def _resolve_deterministic(args: Any, approved: ApprovedModelInput | None) -> bool:
    if args.deterministic is not None:
        return bool(args.deterministic)
    if approved is None:
        return False
    candidates = (approved.root / "recipe.json", approved.model_path.with_suffix(".recipe.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid approved recipe document: {path.name}") from exc
        evaluation = document.get("eval", {}) if isinstance(document, Mapping) else {}
        sampling = str(evaluation.get("action_sampling") or "").strip().lower()
        if sampling:
            if sampling not in {"deterministic", "stochastic"}:
                raise ValueError(f"unsupported recipe eval.action_sampling {sampling!r}")
            return sampling == "deterministic"
    return False


@dataclass
class RecordingReservation:
    session_id: str
    pending_root: Path
    directory: Path
    manifest_path: Path
    manifest: dict[str, Any]


@contextlib.contextmanager
def _recording_reservation(args: Any):
    root = dataset_root(args.root)
    pending_root = root / "pending" / hashlib.sha256(args.reference.encode()).hexdigest()[:32]
    pending_root.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    session_directory = pending_root / session_id
    session_directory.mkdir(parents=True, exist_ok=False)
    lock_descriptor = os.open(session_directory / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
    manifest_path = session_directory / "manifest.json"
    manifest = _session_manifest(manifest_path, reference=args.reference, session_id=session_id)
    _atomic_json(manifest_path, manifest)
    reservation = RecordingReservation(
        session_id, pending_root, session_directory, manifest_path, manifest
    )
    try:
        yield reservation
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
        if session_directory.is_dir() and not list(session_directory.glob("episode-*")):
            shutil.rmtree(session_directory)
            _fsync_directory(pending_root)


def _record_session(
    args: Any,
    approved: ApprovedModelInput | None,
    reservation: RecordingReservation,
) -> int:
    root = dataset_root(args.root)
    session_id = reservation.session_id
    pending_root = reservation.pending_root
    session_directory = reservation.directory
    manifest_path = reservation.manifest_path
    manifest = reservation.manifest
    deterministic = _resolve_deterministic(args, approved)
    model = (
        load_policy_model(approved, device="auto", metadata={"algorithm_id": "ppo"})
        if approved is not None
        else None
    )
    session = create_provider_session(args.provider, args.env_id, args.env_config)
    if args.fps is not None:
        session.fps = float(args.fps)
    environment = build_environment_artifact(
        provider_id=args.provider,
        environment_id=args.env_id,
        declared_config=args.env_config,
        session=session,
    )
    collector = None
    if approved is not None:
        assert model is not None
        session.validate_policy(model)
        collector = _collector_artifact(
            approved,
            environment_contract_id=environment.contract_id,
            deterministic=deterministic,
            session=session,
        )
    session.env.close()
    count = args.episodes if args.episodes is not None else (None if args.agent == "human" else 1)
    completed = 0
    packages: list[Path] = []
    projected_rebuild = _current_tree_size(root, args.reference)
    live_appender = None
    if args.upload_live:
        from rlab.dataset_hub import HubAppendSession

        live_appender = HubAppendSession(str(args.reference).removeprefix("hf://"))
    for episode_index in range(count) if count is not None else iter(int, 1):
        if int(args.seed) + episode_index > MAX_EPISODE_SEED:
            raise ValueError("the next episode seed is outside the supported range")
        session = create_provider_session(args.provider, args.env_id, args.env_config)
        if args.fps is not None:
            session.fps = float(args.fps)
        episode_directory = session_directory / f"episode-{episode_index:06d}"
        package = episode_directory / "package"
        episode_directory.mkdir(parents=True, exist_ok=False)
        episode_id = _record_one(
            session=session,
            environment=environment,
            collector=collector,
            model=model,
            agent=args.agent,
            deterministic=deterministic,
            seed=int(args.seed) + episode_index,
            episode_directory=episode_directory,
            package=package,
            session_id=session_id,
            headless=bool(args.headless),
            projected_rebuild=projected_rebuild,
            ui=str(getattr(args, "ui", "web")),
            no_open=bool(getattr(args, "no_open", False)),
            port=int(getattr(args, "port", 0)),
        )
        if episode_id is None:
            break
        packages.append(package)
        manifest["packages"].append(
            {
                "episode_id": episode_id,
                "path": str(package.relative_to(session_directory)),
                "state": "pending",
            }
        )
        _atomic_json(manifest_path, manifest)
        completed += 1
        print(f"recorded episode {completed}: {episode_id}", flush=True)
        if live_appender is not None:
            try:
                live_appender.append(package)
            except Exception as exc:
                print(f"live upload stopped; episode remains local: {exc}", file=sys.stderr)
                live_appender = None
            else:
                manifest["packages"][-1]["state"] = "uploaded_pending_local"
                _atomic_json(manifest_path, manifest)
    if not packages:
        manifest["state"] = "empty"
        _atomic_json(manifest_path, manifest)
        print("no complete episode was recorded")
        return 0
    combined = session_directory / "combined"
    combine_trees(packages, combined)
    result = adopt_tree(combined, args.reference, root=root)
    manifest["state"] = "committed"
    manifest["collection_fingerprint"] = result.collection_fingerprint
    _atomic_json(manifest_path, manifest)
    shutil.rmtree(session_directory)
    _fsync_directory(pending_root)
    print(
        f"committed {completed} episode(s) to {args.reference!r}; "
        f"fingerprint={result.collection_fingerprint}"
    )
    return 0


def record_command(args: Any) -> int:
    if args.agent == "ppo" and not args.model:
        raise ValueError("--agent ppo requires --model")
    if args.agent != "ppo" and args.model:
        raise ValueError("--model is valid only with --agent ppo")
    if args.agent == "human" and args.headless:
        raise ValueError("human recording cannot be headless")
    if args.upload_live and "/" not in args.reference:
        raise ValueError("--upload-live requires reference to be a Hugging Face owner/repository")
    validate_provider_request(args.env_config)
    seed = int(args.seed)
    episodes = args.episodes
    if seed < 0 or seed > MAX_EPISODE_SEED:
        raise ValueError(f"--seed must be between 0 and {MAX_EPISODE_SEED}")
    if episodes is not None and (episodes < 1 or seed + episodes - 1 > MAX_EPISODE_SEED):
        raise ValueError("the requested episode seed interval is outside the supported range")
    recover_pending(args.reference, root=args.root)
    preflight_recording_reference(args.reference, root=args.root)
    with _recording_reservation(args) as reservation:
        if args.model:
            source = str(args.model)
            model_path = Path(source).expanduser()
            source_identity = source
            download_root = dataset_root(args.root) / "model-sources"
            if is_huggingface_model_ref(source):
                resolved = download_huggingface_model_source(source, root=download_root)
                model_path = resolved.model_path
                source_identity = resolved.artifact_name or source
            elif is_wandb_run_ref(source):
                artifact_ref = model_ref_from_run_path(
                    source,
                    default_project=None,
                    kind="checkpoint",
                    version="latest",
                )
                if artifact_ref is None:
                    raise ValueError(f"could not resolve W&B run model source: {source}")
                resolved = download_artifact_ref_source(artifact_ref, download_root)
                model_path = resolved.model_path
                source_identity = resolved.artifact_name or artifact_ref
            elif ":" in source and "/" in source and not model_path.exists():
                resolved = download_artifact_ref_source(source, download_root)
                model_path = resolved.model_path
                source_identity = resolved.artifact_name or source
            with stage_and_approve_model(model_path, source_identity=source_identity) as approved:
                return _record_session(args, approved, reservation)
        return _record_session(args, None, reservation)


def recover_pending(reference: str, *, root: Path | None) -> None:
    base = dataset_root(root) / "pending" / hashlib.sha256(reference.encode()).hexdigest()[:32]
    if not base.exists():
        return
    for session_directory in sorted(path for path in base.iterdir() if path.is_dir()):
        lock_path = session_directory / ".lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            manifest_path = session_directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("version") != MANIFEST_VERSION
                or manifest.get("reference") != reference
            ):
                continue
            known_paths = {
                str(entry.get("path") or "")
                for entry in manifest.get("packages", [])
                if isinstance(entry, Mapping)
            }
            failures = []
            for episode_directory in sorted(session_directory.glob("episode-*")):
                package = episode_directory / "package"
                if package.is_dir() or not (episode_directory / "active").is_dir():
                    continue
                try:
                    episode_id = _recover_active_episode(episode_directory, package)
                except Exception as exc:
                    failures.append(
                        {"path": episode_directory.name, "reason": str(exc), "state": "abandoned"}
                    )
                    continue
                relative = str(package.relative_to(session_directory))
                if relative not in known_paths:
                    manifest.setdefault("packages", []).append(
                        {"episode_id": episode_id, "path": relative, "state": "recovered"}
                    )
                    known_paths.add(relative)
            if failures:
                manifest["recovery_failures"] = failures
            _atomic_json(manifest_path, manifest)
            packages = []
            inventoried = {
                path for path in session_directory.glob("episode-*/package") if path.is_dir()
            }
            for entry in manifest.get("packages", []):
                relative = Path(str(entry.get("path") or ""))
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                package = session_directory / relative
                if package.is_dir():
                    validate_tree(package)
                    inventoried.add(package)
            for package in sorted(inventoried):
                validate_tree(package)
                packages.append(package)
            if not packages or manifest.get("state") == "committed":
                continue
            combined = session_directory / "recovered-combined"
            combine_trees(packages, combined)
            result = adopt_tree(combined, reference, root=root)
            manifest["state"] = "committed"
            manifest["collection_fingerprint"] = result.collection_fingerprint
            _atomic_json(manifest_path, manifest)
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _read_recovery_journal(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open("rb") as stream:
        expected_step = 0
        while True:
            line = stream.readline(MAX_JOURNAL_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_JOURNAL_LINE_BYTES:
                raise ValueError("recovery journal line exceeds its bound")
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith(b"\n") and not stream.read(1):
                    break
                raise ValueError("recovery journal has corrupt non-tail JSON") from None
            if (
                not isinstance(value, Mapping)
                or value.get("version") != JOURNAL_VERSION
                or value.get("step_index") != expected_step
                or not isinstance(value.get("row"), Mapping)
            ):
                raise ValueError("recovery journal record is malformed or non-contiguous")
            records.append(value)
            expected_step += 1
    return records


def _recover_active_episode(episode_directory: Path, package: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required; install Rlab with --extra dataset") from exc

    active = episode_directory / "active"
    records = _read_recovery_journal(active / "journal.jsonl")
    if not records:
        raise ValueError("active episode has no durable transitions")
    episode_ids = {str(record.get("episode_id") or "") for record in records}
    if len(episode_ids) != 1:
        raise ValueError("recovery journal changes episode UUID")
    episode_id = episode_ids.pop()
    if str(uuid.UUID(episode_id)) != episode_id:
        raise ValueError("recovery journal episode UUID is not canonical")
    rows = [dict(record["row"]) for record in records]
    if any(str(row.get("episode_id")) != episode_id for row in rows):
        raise ValueError("recovery row episode UUID mismatch")
    final_candidate = active / f"candidate-{len(rows):012d}.png"
    if final_candidate.is_symlink() or not final_candidate.is_file():
        raise ValueError("recovery lacks the final durable observation candidate")
    with Image.open(final_candidate) as image:
        image.load()
        final_frame = observation_to_rgb(image)
    final_hash = hashlib.sha256(final_frame.tobytes()).hexdigest()
    if final_hash != str(records[-1].get("next_frame_sha256") or ""):
        raise ValueError("recovery candidate hash does not match the journal")
    width = int(rows[0].get("frame_width") or 0)
    height = int(rows[0].get("frame_height") or 0)
    if tuple(final_frame.shape) != (height, width, 3):
        raise ValueError("recovery candidate dimensions do not match the journal")
    video_relative = str(rows[0].get("video_path") or "")
    source_video = episode_directory / Path(video_relative).name
    repaired_video = episode_directory / f".{episode_id}.recovered.rgb.mkv.bin"
    environment_id = str(rows[0].get("environment_contract_id") or "")
    environment_document = active / ENVIRONMENT_ARTIFACT_DIR / environment_id / "environment.json"
    environment = json.loads(environment_document.read_text(encoding="utf-8"))
    fps = float(environment.get("fps") or 60.0)
    writer = LosslessVideoWriter(repaired_video, fps=fps)
    try:
        decoded = iter_recoverable_video_prefix(
            source_video,
            width=width,
            height=height,
            expected_frames=len(rows) + 1,
        )
        for index, frame in enumerate(decoded):
            expected = (
                str(rows[index].get("frame_sha256") or "") if index < len(rows) else final_hash
            )
            if hashlib.sha256(frame.tobytes()).hexdigest() != expected:
                raise ValueError(f"recovery video hash mismatch at frame {index}")
            writer.write(frame)
        writer.close()
        context = {name: rows[0][name] for name in ROW_CONTEXT_COLUMNS}
        terminal = _terminal_row(
            context=context,
            step=len(rows),
            collector_terminated=not (
                bool(rows[-1].get("terminations")) or bool(rows[-1].get("truncations"))
            ),
            video_path=video_relative,
            frame=final_frame,
        )
        temporary = package.with_name(f".{package.name}.{uuid.uuid4().hex}.tmp")
        dataset = _dataset_from_rows([*rows, terminal])
        try:
            dataset.save_to_disk(str(temporary))
            video_target = temporary.joinpath(*Path(video_relative).parts)
            video_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repaired_video, video_target)
            for artifact_dir in (ENVIRONMENT_ARTIFACT_DIR, COLLECTOR_ARTIFACT_DIR):
                source = active / artifact_dir
                if source.is_dir():
                    shutil.copytree(source, temporary / artifact_dir)
            validate_tree(temporary)
            _fsync_tree(temporary)
            os.replace(temporary, package)
            _fsync_directory(package.parent)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    except Exception:
        writer.abort()
        raise
    finally:
        repaired_video.unlink(missing_ok=True)
    return episode_id


def reexecute_dataset(validation: Any) -> None:
    from rlab.dataset_media import iter_episode_frames

    episodes = validation.dataset
    from rlab.dataset_contract import grouped_episode_rows

    for episode_id, rows in grouped_episode_rows(episodes).items():
        first = rows[0]
        contract_id = str(first["environment_contract_id"])
        document_path = (
            validation.path / ENVIRONMENT_ARTIFACT_DIR / contract_id / ENVIRONMENT_DOCUMENT_FILENAME
        )
        document = json.loads(document_path.read_text(encoding="utf-8"))
        session = create_provider_session(
            str(document["provider_id"]),
            str(document["environment_id"]),
            document["declared_config"],
        )
        try:
            actual_artifact = build_environment_artifact(
                provider_id=str(document["provider_id"]),
                environment_id=str(document["environment_id"]),
                declared_config=document["declared_config"],
                session=session,
            )
            if actual_artifact.contract_id != contract_id:
                raise ValueError(
                    f"episode {episode_id} cannot be reexecuted: runtime environment contract differs"
                )
            frames = iter(iter_episode_frames(rows, root=validation.path))
            observation, _info = session.env.reset(seed=int(first["seed"]))
            actual_frame = observation_to_rgb(session.recording_observation(observation))
            if not np.array_equal(actual_frame, next(frames)):
                raise ValueError(f"episode {episode_id} reset observation differs")
            for step, row in enumerate(rows[:-1]):
                action = row["actions"]
                observation, reward, terminated, truncated, info = session.env.step(action)
                actual_frame = observation_to_rgb(session.recording_observation(observation))
                if float(reward) != float(row["rewards"]):
                    raise ValueError(f"episode {episode_id} reward differs at step {step}")
                if bool(terminated) != bool(row["terminations"]) or bool(truncated) != bool(
                    row["truncations"]
                ):
                    raise ValueError(f"episode {episode_id} boundary differs at step {step}")
                encoded_info = json.dumps(
                    _json_value(info), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                if encoded_info != str(row["infos"]):
                    raise ValueError(f"episode {episode_id} info differs at step {step}")
                if not np.array_equal(actual_frame, next(frames)):
                    raise ValueError(f"episode {episode_id} observation differs at step {step + 1}")
        finally:
            session.env.close()
    print(f"exactly reexecuted {len(validation.episode_fingerprints)} episode(s)")
