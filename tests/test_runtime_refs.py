from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from rlab import runtime_refs


SOURCE_SHA = "1" * 40
RUNTIME_IMAGE_REF = "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "a" * 64


def image_payload() -> dict:
    return {
        "schema_version": 3,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "digest": "sha256:" + "a" * 64,
        "source_sha": SOURCE_SHA,
        "workflow_run_id": "11",
    }


def image_info() -> runtime_refs.RuntimeImageInfo:
    return runtime_refs.runtime_release_from_payload(
        image_payload(),
        label="test image",
        expected_source_sha=SOURCE_SHA,
    )


class RuntimeRefsTests(unittest.TestCase):
    def test_failed_workflow_image_receipt_is_still_usable(self) -> None:
        run = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "failure",
            "url": "https://example.test/run/11",
        }
        with (
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[run]),
            mock.patch.object(
                runtime_refs,
                "_artifact_payload_for_run",
                return_value=image_payload(),
            ),
        ):
            release = runtime_refs.runtime_release_for_source(source_sha=SOURCE_SHA)

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        self.assertEqual(release.workflow_run_id, "11")

    def test_active_workflow_is_reused_without_dispatch(self) -> None:
        active = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "in_progress",
            "conclusion": "",
            "url": "https://example.test/run/11",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=[RuntimeError("not yet"), image_info()],
            ),
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[active]),
            mock.patch.object(runtime_refs.time, "sleep"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
        ):
            release = runtime_refs.wait_for_runtime_release(
                source_sha=SOURCE_SHA,
                workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                branch="main",
                artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                timeout=60,
            )

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        dispatch.assert_not_called()

    def test_missing_workflow_dispatches_exact_source_once(self) -> None:
        active = {
            "databaseId": 12,
            "headSha": SOURCE_SHA,
            "status": "queued",
            "conclusion": "",
            "url": "https://example.test/run/12",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=[RuntimeError("missing"), RuntimeError("not yet"), image_info()],
            ),
            mock.patch.object(
                runtime_refs,
                "_matching_runs",
                side_effect=[[], [active]],
            ),
            mock.patch.object(runtime_refs, "require_remote_source") as remote,
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            release = runtime_refs.wait_for_runtime_release(
                source_sha=SOURCE_SHA,
                workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                branch="main",
                artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                timeout=60,
            )

        self.assertEqual(release.runtime_image_ref, RUNTIME_IMAGE_REF)
        remote.assert_called_once_with(SOURCE_SHA, branch="main", repo_root=".")
        dispatch.assert_called_once()
        self.assertIn(f"source_sha={SOURCE_SHA}", dispatch.call_args.args[0])

    def test_dispatched_workflow_failure_stops_without_retry_loop(self) -> None:
        failed = {
            "databaseId": 12,
            "headSha": SOURCE_SHA,
            "status": "completed",
            "conclusion": "failure",
            "url": "https://example.test/run/12",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(runtime_refs, "_matching_runs", side_effect=[[], [failed]]),
            mock.patch.object(runtime_refs, "require_remote_source"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "completed without a usable image receipt"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_called_once()

    def test_reused_active_workflow_failure_stops_without_new_dispatch(self) -> None:
        active = {
            "databaseId": 11,
            "headSha": SOURCE_SHA,
            "status": "in_progress",
            "conclusion": "",
            "url": "https://example.test/run/11",
        }
        failed = {
            **active,
            "status": "completed",
            "conclusion": "failure",
        }
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(
                runtime_refs,
                "_matching_runs",
                side_effect=[[active], [failed]],
            ),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "completed without a usable image receipt"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_not_called()

    def test_runtime_readiness_timeout_reports_without_canceling(self) -> None:
        with (
            mock.patch.object(
                runtime_refs,
                "runtime_release_for_source",
                side_effect=RuntimeError("missing"),
            ),
            mock.patch.object(runtime_refs, "_matching_runs", return_value=[]),
            mock.patch.object(runtime_refs, "require_remote_source"),
            mock.patch.object(runtime_refs, "_run_gh") as dispatch,
            mock.patch.object(runtime_refs.time, "monotonic", side_effect=[0.0, 61.0]),
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out waiting"):
                runtime_refs.wait_for_runtime_release(
                    source_sha=SOURCE_SHA,
                    workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
                    branch="main",
                    artifact_name=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
                    timeout=60,
                )

        dispatch.assert_called_once()

    def test_remote_source_requires_exact_branch_head(self) -> None:
        completed = mock.Mock(returncode=0, stdout=f"{'2' * 40}\trefs/heads/main\n", stderr="")
        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "push the commit"):
                runtime_refs.require_remote_source(SOURCE_SHA, branch="main")

    def test_only_modal_backend_waits_for_modal_readiness(self) -> None:
        args = SimpleNamespace(
            image_workflow=runtime_refs.DEFAULT_IMAGE_WORKFLOW,
            image_branch="main",
            image_artifact=runtime_refs.DEFAULT_IMAGE_ARTIFACT,
            runtime_readiness_timeout=60,
            runtime_image_ref_file=None,
        )
        release = image_info()
        with (
            mock.patch.object(runtime_refs, "clean_git_source_sha", return_value=SOURCE_SHA),
            mock.patch.object(
                runtime_refs,
                "wait_for_runtime_release",
                return_value=release,
            ),
            mock.patch.object(
                runtime_refs,
                "wait_for_modal_readiness",
                return_value=release,
            ) as modal_wait,
        ):
            runtime_refs.runtime_release_from_args(args, checkpoint_eval_backend="local")
            runtime_refs.runtime_release_from_args(args, checkpoint_eval_backend="none")
            runtime_refs.runtime_release_from_args(args, checkpoint_eval_backend="modal")

        modal_wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
