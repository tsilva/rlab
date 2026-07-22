from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rlab.dataset_contract import canonical_json_bytes, grouped_episode_rows
from rlab.dataset_media import iter_episode_frames


MAX_MATERIALIZED_BYTES = 16 * 1024**3


def _action_space(document: Mapping[str, Any]):
    import gymnasium as gym

    kind = str(document.get("type") or "")
    dtype = np.dtype(str(document.get("dtype") or "int64"))
    if kind == "Discrete":
        return gym.spaces.Discrete(int(document["n"]), start=int(document.get("start", 0)))
    if kind == "MultiBinary":
        value = document.get("n") or document.get("shape")
        return gym.spaces.MultiBinary(value)
    if kind == "MultiDiscrete":
        return gym.spaces.MultiDiscrete(np.asarray(document["nvec"], dtype=dtype))
    if kind == "Box":
        shape = tuple(int(value) for value in document["shape"])
        low = np.asarray(document.get("low", -np.inf), dtype=dtype)
        high = np.asarray(document.get("high", np.inf), dtype=dtype)
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    raise ValueError(f"Minari export does not support action space {kind!r}")


def _environment_document(validation: Any, contract_id: str) -> Mapping[str, Any]:
    path = validation.path / "environments" / contract_id / "environment.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid environment document {contract_id}")
    return value


def _info_buffer(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    decoded = [json.loads(str(row["infos"])) for row in rows[:-1]]
    keys = sorted({str(key) for info in decoded for key in info})
    return {key: [info.get(key) for info in decoded] for key in keys}


def _materialize(validation: Any):
    try:
        from minari.data_collector import EpisodeBuffer
    except ImportError as exc:
        raise RuntimeError(
            "Minari export is not installed; run ./install.sh --extra dataset-minari"
        ) from exc
    episodes = grouped_episode_rows(validation.dataset)
    action_documents: dict[bytes, Mapping[str, Any]] = {}
    buffers = []
    observation_shape: tuple[int, int, int] | None = None
    materialized_bytes = 0
    for index, (episode_id, rows) in enumerate(episodes.items()):
        if bool(rows[-1]["collector_terminated"]):
            raise ValueError(
                f"episode {episode_id} ended at a collector boundary and cannot be exported "
                "to Minari without fabricating a provider boundary"
            )
        contract_id = str(rows[0]["environment_contract_id"])
        environment = _environment_document(validation, contract_id)
        action_document = environment["action_space"]
        action_documents[canonical_json_bytes(action_document)] = action_document
        frames = list(iter_episode_frames(rows, root=validation.path))
        if len(frames) != len(rows):
            raise ValueError(f"episode {episode_id} does not have N+1 observations")
        for frame in frames:
            if observation_shape is None:
                observation_shape = tuple(frame.shape)
            elif tuple(frame.shape) != observation_shape:
                raise ValueError(
                    "Minari export requires uniform decoded recorded RGB dimensions; "
                    f"got {frame.shape} and {observation_shape}"
                )
            materialized_bytes += frame.nbytes
            if materialized_bytes > MAX_MATERIALIZED_BYTES:
                raise ValueError(f"Minari materialization exceeds {MAX_MATERIALIZED_BYTES} bytes")
        actions = [row["actions"] for row in rows[:-1]]
        buffers.append(
            EpisodeBuffer(
                id=index,
                seed=int(rows[0]["seed"]),
                observations=frames,
                actions=actions,
                rewards=[float(row["rewards"]) for row in rows[:-1]],
                terminations=[bool(row["terminations"]) for row in rows[:-1]],
                truncations=[bool(row["truncations"]) for row in rows[:-1]],
                infos=_info_buffer(rows),
            )
        )
    if len(action_documents) != 1:
        raise ValueError("Minari export requires one common provider-native action space")
    if observation_shape is None:
        raise ValueError("Minari export source has no observations")
    import gymnasium as gym

    action_space = _action_space(next(iter(action_documents.values())))
    for episode, rows in zip(buffers, episodes.values(), strict=True):
        for step, action in enumerate(episode.actions or []):
            normalized = (
                int(action)
                if isinstance(action_space, gym.spaces.Discrete)
                else np.asarray(action, dtype=action_space.dtype)
            )
            if not action_space.contains(normalized):
                raise ValueError(
                    f"episode {rows[0]['episode_id']} action {step} is outside {action_space}"
                )
    observation_space = gym.spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
    return buffers, observation_space, action_space


def export_minari_command(args: Any) -> int:
    try:
        import minari
        from minari.utils import get_dataset_path
    except ImportError as exc:
        raise RuntimeError(
            "Minari export is not installed; run ./install.sh --extra dataset-minari"
        ) from exc
    from rlab.dataset_store import open_source

    target = Path(get_dataset_path(args.name))
    if target.exists():
        raise FileExistsError(f"Minari dataset {args.name!r} already exists at {target}")
    with open_source(args.source, root=args.root) as loaded:
        buffers, observation_space, action_space = _materialize(loaded.validation)
        if target.exists():
            raise FileExistsError(f"Minari dataset {args.name!r} appeared during validation")
        dataset = minari.create_dataset_from_buffers(
            args.name,
            buffers,
            observation_space=observation_space,
            action_space=action_space,
            algorithm_name="Rlab Gymrec v3 recording",
            author=args.author,
            description=(
                "Provider-bounded gameplay observations exported by Rlab. "
                "This dataset is not checkpoint-promotion or research-goal evidence."
            ),
            data_format="arrow",
        )
    print(
        f"created Minari dataset {args.name!r}: "
        f"{dataset.total_episodes} episodes, {dataset.total_steps} transitions"
    )
    return 0
