from __future__ import annotations

from typing import Any

from rlab.eval_backend import EvalHandle, EvalPoll


class ModalEvalBackend:
    def __init__(
        self,
        *,
        app_name: str,
        function_name: str = "evaluate_checkpoint",
        environment_name: str = "rlab-eval",
    ):
        self.app_name = str(app_name).strip()
        self.function_name = str(function_name).strip()
        self.environment_name = str(environment_name).strip()
        if not self.app_name or not self.function_name or not self.environment_name:
            raise ValueError("Modal environment, app, and function names are required")

    def submit(self, intent: dict[str, Any]) -> EvalHandle:
        import modal

        call = modal.Function.from_name(
            self.app_name,
            self.function_name,
            environment_name=self.environment_name,
        ).spawn(intent)
        return EvalHandle(provider="modal", call_id=str(call.object_id))

    def poll(self, handle: EvalHandle) -> EvalPoll:
        if handle.provider != "modal":
            raise ValueError(f"unsupported eval handle provider: {handle.provider}")
        import modal

        call = modal.FunctionCall.from_id(handle.call_id)
        try:
            result = call.get(timeout=0)
        except modal.exception.TimeoutError:
            return EvalPoll(status="running")
        except Exception as exc:
            return EvalPoll(status="failed", error=repr(exc)[:4000])
        return EvalPoll(status="succeeded", provider_result=result)

    def cancel(self, handle: EvalHandle) -> None:
        if handle.provider != "modal":
            raise ValueError(f"unsupported eval handle provider: {handle.provider}")
        import modal

        modal.FunctionCall.from_id(handle.call_id).cancel()
