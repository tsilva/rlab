from __future__ import annotations

import shlex
from collections.abc import Sequence


def render_command(argv: Sequence[str]) -> str:
    return shlex.join([str(value) for value in argv])


def experiment_retry_finalization_command(run_id: int) -> list[str]:
    return ["rlab", "experiment", "retry-finalization", "--run", str(int(run_id))]


def eval_modal_retry_command(eval_job_id: int) -> list[str]:
    return ["rlab", "eval", "modal", "retry", "--eval-job", str(int(eval_job_id))]


def eval_modal_recover_command(run_id: int) -> list[str]:
    return ["rlab", "eval", "modal", "recover", "--run", str(int(run_id))]


def eval_modal_abandon_command(run_id: int) -> list[str]:
    return ["rlab", "eval", "modal", "abandon", "--run", str(int(run_id))]
