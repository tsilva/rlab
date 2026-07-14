from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


class FakeLoggedArtifact:
    def __init__(self) -> None:
        self.wait_called = False

    def wait(self) -> None:
        self.wait_called = True


class FakeArtifact:
    def __init__(
        self,
        name: str = "artifact",
        type: str = "model",
        metadata: Mapping[str, Any] | None = None,
        *,
        version: str = "",
        qualified_name: str = "",
        aliases: Iterable[Any] = (),
        filename: str | None = None,
        download: Callable[[Path, "FakeArtifact"], None] | None = None,
        logged_by: Any = None,
    ) -> None:
        self.name = name
        self.type = type
        self.metadata = dict(metadata or {})
        self.version = version
        self.qualified_name = qualified_name
        self.aliases = list(aliases)
        self.filename = filename or str(self.metadata.get("filename") or "")
        self.references: list[tuple[str, str | None]] = []
        self.files: list[tuple[str, str]] = []
        self._download = download
        self._logged_by = logged_by

    def add_reference(self, uri: str, name: str | None = None) -> None:
        self.references.append((uri, name))

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))

    def download(self, root: str) -> str:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        if self._download is not None:
            self._download(path, self)
        elif self.filename:
            (path / self.filename).write_bytes(b"model")
        return str(path)

    def logged_by(self):
        return self._logged_by


class FakeRun:
    def __init__(
        self,
        *,
        id: str = "run-id",
        name: str = "run-name",
        path: Iterable[str] | None = None,
        config: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
        notes: str = "",
        logged_artifacts: Iterable[FakeArtifact] = (),
        logged_artifact_result: FakeLoggedArtifact | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.path = tuple(path or ("entity", "project", "runs", id))
        self.config = dict(config or {})
        self.summary = dict(summary or {})
        self.notes = notes
        self._logged_artifacts = list(logged_artifacts)
        self.logged_artifact_result = logged_artifact_result
        self.artifact_logs: list[tuple[FakeArtifact, list[str] | None]] = []
        self.metric_logs: list[tuple[dict[str, Any], int | None]] = []

    def logged_artifacts(self) -> list[FakeArtifact]:
        return list(self._logged_artifacts)

    def log_artifact(
        self, artifact: FakeArtifact, aliases: list[str] | None = None
    ) -> FakeLoggedArtifact | None:
        self.artifact_logs.append((artifact, aliases))
        return self.logged_artifact_result

    def log(self, payload: Mapping[str, Any], step: int | None = None) -> None:
        self.metric_logs.append((dict(payload), step))


class FakeApi:
    def __init__(
        self,
        *,
        artifact: Callable[[str, str | None], Any] | None = None,
        run: Callable[[str], Any] | None = None,
        runs: Callable[[str, Mapping[str, Any] | None], Iterable[Any]] | None = None,
    ) -> None:
        self._artifact = artifact
        self._run = run
        self._runs = runs
        self.artifact_calls: list[tuple[str, str | None]] = []
        self.run_calls: list[str] = []
        self.runs_calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def artifact(self, ref: str, type: str | None = None):
        self.artifact_calls.append((ref, type))
        if self._artifact is None:
            raise RuntimeError("artifact not found")
        return self._artifact(ref, type)

    def run(self, path: str):
        self.run_calls.append(path)
        if self._run is None:
            raise RuntimeError("run not found")
        return self._run(path)

    def runs(self, project: str, filters: Mapping[str, Any] | None = None) -> list[Any]:
        self.runs_calls.append((project, filters))
        if self._runs is None:
            return []
        return list(self._runs(project, filters))


class FakeWandb:
    def __init__(self, *, api: FakeApi | None = None) -> None:
        self.api = api or FakeApi()
        self.created_artifacts: list[FakeArtifact] = []

    def Api(self) -> FakeApi:
        return self.api

    def Artifact(self, name: str, type: str, metadata: Mapping[str, Any]) -> FakeArtifact:
        artifact = FakeArtifact(name, type, metadata)
        self.created_artifacts.append(artifact)
        return artifact
