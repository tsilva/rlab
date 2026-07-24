from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class EvalHandle:
    provider: str
    call_id: str


@dataclass(frozen=True)
class EvalPoll:
    status: Literal["running", "succeeded", "failed", "canceled"]
    provider_result: Any | None = None
    error: str | None = None


class EvalBackend(Protocol):
    def submit(self, intent: dict[str, Any]) -> EvalHandle: ...

    def poll(self, handle: EvalHandle) -> EvalPoll: ...

    def cancel(self, handle: EvalHandle) -> None: ...
