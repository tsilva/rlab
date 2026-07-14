from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from containers.train.dependency_key import dependency_key
from containers.train.runtime_key import RUNTIME_INPUT_PATHS, runtime_key


class TrainImageTests(unittest.TestCase):
    def test_runtime_key_covers_runtime_inputs_but_excludes_payload_only_files(self) -> None:
        key = runtime_key(repo_root=Path(".").resolve())

        self.assertEqual(len(key), 64)
        self.assertIn("src", RUNTIME_INPUT_PATHS)
        self.assertIn("experiments/modal_eval.yaml", RUNTIME_INPUT_PATHS)
        self.assertNotIn("experiments/goals", RUNTIME_INPUT_PATHS)
        self.assertNotIn("tests", RUNTIME_INPUT_PATHS)
        self.assertNotIn("README.md", RUNTIME_INPUT_PATHS)
        self.assertNotIn("METRICS.md", RUNTIME_INPUT_PATHS)
        self.assertNotIn("INSTANCES.md", RUNTIME_INPUT_PATHS)

    def test_runtime_key_tracks_content_path_and_executable_mode_only_for_runtime_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "src" / "module.py"
            goal = root / "experiments" / "goals" / "goal.yaml"
            test_file = root / "tests" / "test_module.py"
            readme = root / "README.md"
            for path, text in (
                (source, "VALUE = 1\n"),
                (goal, "target: one\n"),
                (test_file, "def test_one(): pass\n"),
                (readme, "first docs\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            baseline = runtime_key(repo_root=root)

            goal.write_text("target: two\n", encoding="utf-8")
            test_file.write_text("def test_two(): pass\n", encoding="utf-8")
            readme.write_text("changed docs\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            self.assertEqual(runtime_key(repo_root=root), baseline)

            source.write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", str(source)], cwd=root, check=True)
            content_key = runtime_key(repo_root=root)
            self.assertNotEqual(content_key, baseline)

            source.chmod(source.stat().st_mode | 0o111)
            subprocess.run(["git", "add", str(source)], cwd=root, check=True)
            mode_key = runtime_key(repo_root=root)
            self.assertNotEqual(mode_key, content_key)

            renamed = source.with_name("renamed.py")
            source.rename(renamed)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            self.assertNotEqual(runtime_key(repo_root=root), mode_key)

    def test_runtime_uses_overrideable_linked_dependency_base(self) -> None:
        dockerfile = Path("containers/train/Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(dockerfile.count("# dependency-image-inputs-begin"), 1)
        self.assertEqual(dockerfile.count("# dependency-image-inputs-end"), 1)
        self.assertIn("ARG RUNTIME_BASE=dependencies", dockerfile)
        self.assertIn("FROM scratch AS runtime-overlay", dockerfile)
        self.assertIn("FROM ${RUNTIME_BASE} AS runtime", dockerfile)

        runtime = dockerfile.split("FROM ${RUNTIME_BASE} AS runtime", maxsplit=1)[1]
        runtime_instructions = [
            line.strip() for line in runtime.splitlines() if line and not line.startswith(" ")
        ]
        self.assertFalse(any(line.startswith("RUN ") for line in runtime_instructions))
        copies = [line for line in runtime_instructions if line.startswith("COPY ")]
        self.assertEqual(copies, ["COPY --link --from=runtime-overlay / /"])

    def test_dependency_key_ignores_runtime_changes(self) -> None:
        source = Path("containers/train/Dockerfile").read_text(encoding="utf-8")
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        lockfile = Path("uv.lock").read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dockerfile_path = root / "Dockerfile"
            pyproject_path = root / "pyproject.toml"
            lockfile_path = root / "uv.lock"
            dockerfile_path.write_text(source, encoding="utf-8")
            pyproject_path.write_text(pyproject, encoding="utf-8")
            lockfile_path.write_text(lockfile, encoding="utf-8")

            def current_key() -> str:
                return dependency_key(
                    dockerfile=dockerfile_path,
                    pyproject=pyproject_path,
                    lockfile=lockfile_path,
                )

            baseline = current_key()
            dockerfile_path.write_text(source + "\n# runtime-only change\n", encoding="utf-8")
            self.assertEqual(current_key(), baseline)

            dockerfile_path.write_text(
                source.replace("ARG RUNTIME_BASE=dependencies", "ARG RUNTIME_BASE=other-base"),
                encoding="utf-8",
            )
            self.assertEqual(current_key(), baseline)

            dockerfile_path.write_text(
                source.replace(
                    'LABEL io.rlab.uv-lock.sha256="${UV_LOCK_SHA256}"',
                    'LABEL io.rlab.uv-lock.sha256="changed-${UV_LOCK_SHA256}"',
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(current_key(), baseline)

            dockerfile_path.write_text(source, encoding="utf-8")
            pyproject_path.write_text(pyproject + "\n# dependency change\n", encoding="utf-8")
            self.assertNotEqual(current_key(), baseline)

            pyproject_path.write_text(pyproject, encoding="utf-8")
            lockfile_path.write_text(lockfile + "\n# resolution change\n", encoding="utf-8")
            self.assertNotEqual(current_key(), baseline)

    def test_workflow_reuses_published_base_without_runtime_cache_export(self) -> None:
        workflow = Path(".github/workflows/rlab-train-image.yml").read_text(encoding="utf-8")

        self.assertIn("dependency_key=$(python containers/train/dependency_key.py)", workflow)
        self.assertIn(
            "RUNTIME_BASE=${{ steps.dependency_base.outputs.runtime_base }}",
            workflow,
        )
        self.assertIn('echo "runtime_base=dependencies"', workflow)

        runtime_build = workflow.split("      - name: Build image", maxsplit=1)[1].split(
            "      - name: Resolve published digest", maxsplit=1
        )[0]
        self.assertIn(
            "ref=${{ env.REGISTRY }}/${{ env.DEPENDENCY_IMAGE_NAME }}:buildcache",
            runtime_build,
        )
        self.assertNotIn(
            "ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache",
            runtime_build,
        )
        self.assertNotIn("type=gha", runtime_build)
        self.assertNotIn("cache-to:", runtime_build)

    def test_image_receipt_is_published_before_independent_modal_readiness(self) -> None:
        workflow = Path(".github/workflows/rlab-train-image.yml").read_text(encoding="utf-8")
        modal_workflow = Path(".github/workflows/rlab-modal-eval.yml").read_text(encoding="utf-8")

        self.assertIn("workflow_call:", modal_workflow)
        self.assertIn("workflow_dispatch:", modal_workflow)
        self.assertIn("deploy-modal-evaluator:", workflow)
        self.assertIn("uses: ./.github/workflows/rlab-modal-eval.yml", workflow)
        build = workflow.split("  build:", maxsplit=1)[1].split(
            "  deploy-modal-evaluator:", maxsplit=1
        )[0]
        self.assertIn("name: rlab-train-image", build)
        self.assertIn('"schema_version": 4', build)
        self.assertIn("runtime-${{ steps.build_meta.outputs.runtime_input_sha256 }}", build)
        self.assertIn("runtime_build_source_sha", build)
        self.assertNotIn("docker pull", build)
        self.assertNotIn("--validate-config-stdin", build)
        deploy = workflow.split("  deploy-modal-evaluator:", maxsplit=1)[1]
        self.assertIn("if: github.event_name != 'pull_request'", deploy)
        self.assertNotIn("publish-runtime:", workflow)
        self.assertIn("name: rlab-modal-eval-readiness", modal_workflow)
        self.assertIn("--only-group modal-deploy --no-install-project", modal_workflow)
        self.assertIn("uv run --no-sync modal deploy", modal_workflow)
        self.assertIn("if: inputs.deploy_required", modal_workflow)
        self.assertIn('"schema_version": 2', modal_workflow)
        self.assertIn(
            "from rlab.contract_versions import TRAIN_CONFIG_CONTRACT_SCHEMA_VERSION",
            modal_workflow,
        )
        self.assertIn(
            'int(result.get("schema_version") or 0) != TRAIN_CONFIG_CONTRACT_SCHEMA_VERSION',
            modal_workflow,
        )
        self.assertIn("required: false", modal_workflow)

    def test_main_push_resolves_every_source_but_reuses_existing_runtime(self) -> None:
        workflow = Path(".github/workflows/rlab-train-image.yml").read_text(encoding="utf-8")
        push_trigger = workflow.split("  push:", maxsplit=1)[1].split(
            "  pull_request:", maxsplit=1
        )[0]
        self.assertIn("branches: [main]", push_trigger)
        self.assertNotIn("paths:", push_trigger)
        self.assertIn(
            "if: github.event_name == 'pull_request' || "
            "steps.existing_image.outputs.exists != 'true'",
            workflow,
        )
        self.assertIn('echo "deploy_required=false"', workflow)
        self.assertIn("if: inputs.deploy_required", Path(
            ".github/workflows/rlab-modal-eval.yml"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
