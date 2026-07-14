from __future__ import annotations

import os
from pathlib import Path

import modal

from rlab.modal_eval_config import load_modal_eval_config, modal_app_name
from rlab.runtime_contract import runtime_contract


repo_root = Path(__file__).resolve().parents[2]
config = load_modal_eval_config(repo_root / "experiments" / "modal_eval.yaml")
runtime_image_ref = os.environ.get("RLAB_MODAL_EVAL_RUNTIME_IMAGE", "").strip()
if not runtime_image_ref:
    raise RuntimeError("RLAB_MODAL_EVAL_RUNTIME_IMAGE must be an immutable docker image ref")
registry_ref = runtime_image_ref.removeprefix("docker:")
app_name = modal_app_name(config.app_name_prefix, runtime_image_ref)
registry_secret_name = os.environ.get("RLAB_MODAL_REGISTRY_SECRET", "").strip()
registry_secret = modal.Secret.from_name(registry_secret_name) if registry_secret_name else None
image = modal.Image.from_registry(registry_ref, secret=registry_secret)
image = image.env({"RLAB_MODAL_EVAL_RUNTIME_IMAGE": runtime_image_ref})
app = modal.App(app_name)


@app.function(
    name=config.function_name,
    image=image,
    cpu=config.cpu,
    memory=config.memory_mib,
    min_containers=config.min_containers,
    buffer_containers=config.buffer_containers,
    max_containers=config.max_containers,
    scaledown_window=config.scaledown_window_seconds,
    retries=0,
    timeout=config.promotion_timeout_seconds,
    startup_timeout=config.startup_timeout_seconds,
    single_use_containers=config.single_use_containers,
    include_source=False,
)
def evaluate_checkpoint(payload: dict) -> dict:
    from rlab.modal_eval_worker import execute_attempt

    return execute_attempt(payload)


@app.function(
    name="startup_probe",
    image=image,
    cpu=0.125,
    memory=128,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    retries=0,
    timeout=30,
    startup_timeout=config.startup_timeout_seconds,
    single_use_containers=True,
    include_source=False,
)
def startup_probe() -> dict[str, str]:
    """Prove the deployed image can import its packaged evaluator contract."""
    return {
        **runtime_contract(runtime_image_ref=runtime_image_ref),
        "app_name": app_name,
    }
