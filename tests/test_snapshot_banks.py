from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from breakout_turbo_env import BreakoutVecEnv

from rlab.env import EnvConfig, make_training_batch_runtime
from rlab.snapshot_banks import load_breakout_snapshot_bank


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _native_env(*, num_envs: int) -> BreakoutVecEnv:
    return BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        state="Start",
        scenario="scenario",
        info="data",
        use_restricted_actions="simple",
        num_envs=num_envs,
        num_threads=1,
        obs_resize=(84, 84),
        obs_crop=(17, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_resize_algorithm="area",
        frame_skip=4,
        frame_stack=4,
        obs_layout="chw",
        obs_grayscale=True,
        obs_copy="copy",
        info_filter="all",
    )


def _snapshot_archive(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    source = _native_env(num_envs=2)
    try:
        mask = np.ones(2, dtype=np.bool_)
        observations, _infos = source.reset(seed=[1, 2], options={"reset_mask": mask})
        source.step(np.asarray([1, 1], dtype=np.int64))
        for _ in range(8):
            observations, _rewards, _terminated, _truncated, _infos = source.step(
                np.asarray([2, 3], dtype=np.int64)
            )
        states = source.get_state()
    finally:
        source.close()

    root = tmp_path / "bank"
    state_root = root / "states"
    state_root.mkdir(parents=True)
    entries = []
    observation_hashes = {}
    for lane, state in enumerate(states):
        state_id = f"snapshot-{lane}"
        relative = f"states/{state_id}.bto"
        (root / relative).write_bytes(state)
        observation_hashes[state_id] = _sha256(observations[lane].tobytes(order="C"))
        entries.append(
            {
                "id": state_id,
                "state_file": relative,
                "state_sha256": _sha256(state),
                "state_size_bytes": len(state),
                "observation_sha256": observation_hashes[state_id],
            }
        )
    manifest = {
        "document_type": "rlab.breakout_snapshot_bank",
        "format_version": 1,
        "environment": {
            "game": "Breakout-Atari2600-v0",
            "provider": "breakout-turbo-env",
            "frame_skip": 4,
            "frame_stack": 4,
            "observation_size": 84,
            "obs_crop": [17, 0, 0, 0],
            "obs_crop_mode": "mask",
            "obs_crop_fill": 0,
            "obs_resize_algorithm": "area",
            "obs_grayscale": True,
            "obs_layout": "chw",
            "sticky_action_prob": 0.0,
            "reward_clip": False,
            "action_meanings": ["noop", "button", "right", "left"],
        },
        "snapshots": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = tmp_path / "bank.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(root, arcname="bank")
    return archive, _sha256(archive.read_bytes()), observation_hashes


def _runtime_config(archive: Path, archive_sha256: str) -> EnvConfig:
    state_ids = ("snapshot-0", "snapshot-1")
    return EnvConfig(
        env_provider="breakout-turbo-env",
        game="Breakout-Atari2600-v0",
        state="Start",
        states=state_ids,
        state_probs=(0.5, 0.5),
        frame_skip=4,
        max_pool_frames=False,
        sticky_action_prob=0.0,
        observation_size=84,
        obs_crop=(17, 0, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_resize_algorithm="area",
        env_args={
            "scenario": "scenario",
            "info": "data",
            "use_restricted_actions": "simple",
            "record": False,
            "players": 1,
            "inttype": "stable",
            "obs_type": "image",
            "render_mode": "rgb_array",
            "info_filter": "all",
            "num_threads": 1,
            "rom_path": None,
            "obs_copy": "safe_view",
            "obs_grayscale": True,
            "obs_layout": "chw",
            "frame_stack": 4,
            "noop_reset_max": 0,
            "reward_clip": False,
            "use_fire_reset": False,
            "snapshot_bank_uri": archive.resolve().as_uri(),
            "snapshot_bank_sha256": archive_sha256,
        },
        task={
            "id": "identity",
            "action": {"set": "native"},
            "signals": {"ball_y": "ball_y"},
            "events": {},
            "termination": {"max_episode_steps": 54_000},
            "reward": {"reward_mode": "native"},
        },
    )


def test_loads_hash_bound_breakout_snapshot_bank(tmp_path: Path) -> None:
    archive, archive_sha256, _observation_hashes = _snapshot_archive(tmp_path)

    bank = load_breakout_snapshot_bank(archive.resolve().as_uri(), archive_sha256)

    assert bank.archive_sha256 == archive_sha256
    assert bank.state_ids == ("snapshot-0", "snapshot-1")
    assert all(value.startswith(b"BTO1") for value in bank.states.values())
    assert bank.observation_sha256 == _observation_hashes


def test_wandb_snapshot_bank_locator_is_not_supported() -> None:
    with pytest.raises(ValueError, match="https:// or file://"):
        load_breakout_snapshot_bank(
            "wandb-artifact:entity/project/bank:v0/snapshot-bank.tar.gz",
            "0" * 64,
        )


def test_rejects_snapshot_bank_archive_hash_mismatch(tmp_path: Path) -> None:
    archive, _archive_sha256, _observation_hashes = _snapshot_archive(tmp_path)

    with pytest.raises(ValueError, match="archive hash mismatch"):
        load_breakout_snapshot_bank(archive.resolve().as_uri(), "0" * 64)


def test_rejects_duplicate_policy_observation_stacks(tmp_path: Path) -> None:
    archive, _archive_sha256, _observation_hashes = _snapshot_archive(tmp_path)
    extract_root = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extract_root, filter="data")
    manifest_path = extract_root / "bank" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshots"][1]["observation_sha256"] = manifest["snapshots"][0][
        "observation_sha256"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    duplicate_archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(duplicate_archive, "w:gz") as handle:
        handle.add(extract_root / "bank", arcname="bank")
    duplicate_sha = _sha256(duplicate_archive.read_bytes())

    with pytest.raises(ValueError, match="duplicate policy observation stacks"):
        load_breakout_snapshot_bank(duplicate_archive.resolve().as_uri(), duplicate_sha)


def test_runtime_resets_from_persisted_snapshot_distribution(tmp_path: Path) -> None:
    archive, archive_sha256, observation_hashes = _snapshot_archive(tmp_path)
    runtime = make_training_batch_runtime(
        _runtime_config(archive, archive_sha256),
        n_envs=2,
        seed=123,
    )
    try:
        observations = runtime.reset()

        starts = [str(info["start_id"]) for info in runtime.reset_infos]
        assert set(starts) <= set(observation_hashes)
        assert [info["start_source"] for info in runtime.reset_infos] == [
            "snapshot",
            "snapshot",
        ]
        for lane, state_id in enumerate(starts):
            assert _sha256(observations[lane].tobytes(order="C")) == observation_hashes[state_id]
        assert runtime.provider.get_images().shape == (210, 160, 3)
        step = runtime.step(np.asarray([0, 1], dtype=np.int64))
        assert step.observations.shape == (2, 4, 84, 84)
    finally:
        runtime.close()
