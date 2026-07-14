from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from rlab.modal_eval_protocol import execution_key
from rlab.modal_eval_storage import file_sha256, write_downloaded_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="utf-8")


def _provider_rom_identity(path: Path, algorithm: str) -> str:
    if algorithm != "sha1-provider-body-v1":
        raise ValueError(f"unsupported provider ROM identity algorithm: {algorithm}")
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        if path.suffix.lower() == ".nes":
            handle.read(16)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_json(url: str, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    parsed = urlparse(url)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if path.read_bytes() != payload:
            raise RuntimeError("immutable attempt result already exists with different content")
        return digest
    import urllib.request

    request = urllib.request.Request(
        url,
        data=payload,
        method="PUT",
        headers={"Content-Type": "application/json", "If-None-Match": "*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if int(response.status) >= 300:
                raise RuntimeError(f"result upload failed with HTTP {response.status}")
    except Exception as exc:
        status = getattr(exc, "code", None)
        raise RuntimeError(
            f"result upload failed: {type(exc).__name__}"
            + (f" HTTP {status}" if status is not None else "")
        ) from None
    return digest


def run_child(input_path: Path, output_path: Path) -> int:
    request = json.loads(input_path.read_text(encoding="utf-8"))
    contract = request["contract"]
    rom_path = Path(request["rom_path"])
    model_path = Path(request["model_path"])
    rom_dir = rom_path.parent
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", str(rom_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    from stable_baselines3 import PPO

    from rlab.env import resolve_env_config
    from rlab.env_metadata import env_config_from_config_dict
    from rlab.eval_runner import evaluate_model_episodes

    config = env_config_from_config_dict(dict(contract["environment"]))
    if config is None:
        raise ValueError("remote eval environment contract is invalid")
    config = resolve_env_config(config)
    model = PPO.load(model_path, device="cpu")
    metrics, _video = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=int(contract["episodes"]),
        seed=int(contract["seed"]),
        max_steps=int(contract["max_steps"]),
        deterministic=False,
        n_envs=int(contract["n_envs"]),
        progress=True,
        progress_description="modal checkpoint eval",
    )
    episode_results = metrics.pop("episode_results")
    _write_json(output_path, {"metrics": metrics, "episode_results": episode_results})
    return 0


def execute_attempt(payload: Mapping[str, Any]) -> dict[str, Any]:
    attempt_id = str(payload["attempt_id"])
    contract = dict(payload["contract"])
    execution = execution_key(contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract_schema_version": int(contract["schema_version"]),
        "attempt_id": attempt_id,
        "execution_key": execution,
        "checkpoint_sha256": str(contract["checkpoint_sha256"]),
        "runtime_image_ref": str(contract["runtime_image_ref"]),
        "rom_sha256": str(contract["asset"]["sha256"]),
        "seed_protocol": str(contract["seed_protocol"]),
        "n_envs": int(contract["n_envs"]),
        "episodes": int(contract["episodes"]),
    }
    if time.time() >= float(payload["expires_at"]):
        result.update(status="expired", error="attempt expired before execution")
        result_sha = _upload_json(str(payload["result_put_url"]), result)
        return {"result_uri": payload["result_uri"], "result_sha256": result_sha}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="rlab-modal-eval-") as temporary:
            root = Path(temporary)
            model_path = write_downloaded_file(str(payload["model_get_url"]), root / "model.zip")
            if file_sha256(model_path) != str(contract["checkpoint_sha256"]):
                raise ValueError("downloaded checkpoint hash mismatch")
            metadata_path = write_downloaded_file(
                str(payload["metadata_get_url"]), root / "metadata.json"
            )
            if file_sha256(metadata_path) != str(payload["metadata_sha256"]):
                raise ValueError("downloaded checkpoint metadata hash mismatch")
            json.loads(metadata_path.read_text(encoding="utf-8"))
            asset = contract["asset"]
            cache_dir = Path("/tmp/rlab-modal-assets") / str(asset["sha256"])
            cached_rom = cache_dir / str(asset["filename"])
            cache_valid = cached_rom.is_file() and file_sha256(cached_rom) == str(asset["sha256"])
            if not cache_valid:
                cached_rom.unlink(missing_ok=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                downloaded_rom = write_downloaded_file(
                    str(payload["rom_get_url"]), root / "roms" / str(asset["filename"])
                )
                if file_sha256(downloaded_rom) != str(asset["sha256"]):
                    raise ValueError("downloaded ROM hash mismatch")
                temporary_cache = cache_dir / f".{asset['filename']}.{attempt_id}.tmp"
                temporary_cache.write_bytes(downloaded_rom.read_bytes())
                try:
                    os.link(temporary_cache, cached_rom)
                except FileExistsError:
                    pass
                finally:
                    temporary_cache.unlink(missing_ok=True)
            rom_path = cached_rom
            if _provider_rom_identity(
                rom_path,
                str(asset.get("provider_rom_identity_algorithm") or ""),
            ) != str(asset["provider_rom_identity"]):
                raise ValueError("downloaded ROM provider identity mismatch")
            child_input = root / "child-input.json"
            child_output = root / "child-output.json"
            _write_json(
                child_input,
                {
                    "contract": contract,
                    "model_path": str(model_path),
                    "rom_path": str(rom_path),
                },
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rlab.modal_eval_worker",
                    "child",
                    "--input",
                    str(child_input),
                    "--output",
                    str(child_output),
                ],
                stdout=None,
                stderr=None,
                text=True,
                timeout=float(payload["child_timeout_seconds"]),
            )
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout or "see Modal logs")[-4000:]
                raise RuntimeError(f"eval child exited {completed.returncode}: {error}")
            child_result = json.loads(child_output.read_text(encoding="utf-8"))
            result.update(
                status="succeeded",
                duration_seconds=time.monotonic() - started,
                metrics=child_result["metrics"],
                episode_results=child_result["episode_results"],
            )
    except subprocess.TimeoutExpired:
        result.update(status="failed", error="eval child timeout")
    except BaseException as exc:
        result.update(status="failed", error=repr(exc)[:4000])
    result_sha = _upload_json(str(payload["result_put_url"]), result)
    return {"result_uri": payload["result_uri"], "result_sha256": result_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--input", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "child":
        return run_child(args.input, args.output)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
