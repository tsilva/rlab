from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from containers.train.dependency_key import dependency_key
from containers.train.gpu_key import gpu_key
from containers.train.runtime_key import RUNTIME_INPUT_PATHS, overlay_key, runtime_key


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


class TrainImageTests(unittest.TestCase):
    def test_runtime_key_covers_overlay_inputs_not_dependency_contracts(self) -> None:
        root = Path(".").resolve()
        overlay = overlay_key(repo_root=root)
        key = runtime_key(repo_root=root, dependency_digest=DIGEST_A)

        self.assertEqual(len(overlay), 64)
        self.assertEqual(len(key), 64)
        self.assertNotEqual(key, runtime_key(repo_root=root, dependency_digest=DIGEST_B))
        self.assertIn("src", RUNTIME_INPUT_PATHS)
        self.assertIn("pyproject.toml", RUNTIME_INPUT_PATHS)
        self.assertNotIn("uv.lock", RUNTIME_INPUT_PATHS)
        self.assertNotIn("containers/train/gpu-linux-amd64.lock", RUNTIME_INPUT_PATHS)
        self.assertNotIn("tests", RUNTIME_INPUT_PATHS)
        self.assertNotIn("README.md", RUNTIME_INPUT_PATHS)

    def test_overlay_key_tracks_indexed_content_path_mode_and_runtime_docker_section(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "src" / "module.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            dockerfile = root / "containers" / "train" / "Dockerfile"
            dockerfile.parent.mkdir(parents=True)
            dockerfile.write_text(
                "# syntax=docker/dockerfile:1.7\n"
                "ARG PYTHON_IMAGE=python@sha256:one\n"
                "ARG UV_IMAGE=uv@sha256:two\n"
                "# dependency-image-inputs-begin\n"
                "FROM base AS dependencies\n"
                "# dependency-image-inputs-end\n"
                "# runtime-image-inputs-begin\n"
                "FROM dependencies AS runtime\n"
                "COPY src /src\n"
                "# runtime-image-inputs-end\n",
                encoding="utf-8",
            )
            ignored = root / "uv.lock"
            ignored.write_text("version = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            baseline = overlay_key(repo_root=root)

            ignored.write_text("version = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "uv.lock"], cwd=root, check=True)
            self.assertEqual(overlay_key(repo_root=root), baseline)

            source.write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "add", "src/module.py"], cwd=root, check=True)
            content_key = overlay_key(repo_root=root)
            self.assertNotEqual(content_key, baseline)

            source.chmod(source.stat().st_mode | 0o111)
            subprocess.run(["git", "add", "src/module.py"], cwd=root, check=True)
            mode_key = overlay_key(repo_root=root)
            self.assertNotEqual(mode_key, content_key)

            dockerfile.write_text(
                dockerfile.read_text(encoding="utf-8").replace("COPY src /src", "COPY src /app"),
                encoding="utf-8",
            )
            self.assertNotEqual(overlay_key(repo_root=root), mode_key)

    def test_gpu_and_dependency_keys_follow_the_layer_dag(self) -> None:
        source = Path("containers/train/Dockerfile").read_text(encoding="utf-8")
        gpu_lock = Path("containers/train/gpu-linux-amd64.lock").read_text(encoding="utf-8")
        train_lock = Path("containers/train/train-linux-amd64.lock").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dockerfile = root / "Dockerfile"
            gpu_plan = root / "gpu.lock"
            train_plan = root / "train.lock"
            dockerfile.write_text(source, encoding="utf-8")
            gpu_plan.write_text(gpu_lock, encoding="utf-8")
            train_plan.write_text(train_lock, encoding="utf-8")

            gpu_baseline = gpu_key(dockerfile=dockerfile, lockfile=gpu_plan)
            dependency_baseline = dependency_key(
                dockerfile=dockerfile,
                lockfile=train_plan,
                gpu_digest=DIGEST_A,
            )
            dockerfile.write_text(source + "\n# unrelated runtime edit\n", encoding="utf-8")
            self.assertEqual(gpu_key(dockerfile=dockerfile, lockfile=gpu_plan), gpu_baseline)
            self.assertEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=train_plan,
                    gpu_digest=DIGEST_A,
                ),
                dependency_baseline,
            )

            train_plan.write_text(train_lock + "# changed\n", encoding="utf-8")
            self.assertEqual(gpu_key(dockerfile=dockerfile, lockfile=gpu_plan), gpu_baseline)
            self.assertNotEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=train_plan,
                    gpu_digest=DIGEST_A,
                ),
                dependency_baseline,
            )
            self.assertNotEqual(
                dependency_key(
                    dockerfile=dockerfile,
                    lockfile=Path("containers/train/train-linux-amd64.lock"),
                    gpu_digest=DIGEST_B,
                ),
                dependency_baseline,
            )

    def test_dockerfile_has_linked_three_layer_runtime(self) -> None:
        dockerfile = Path("containers/train/Dockerfile").read_text(encoding="utf-8")

        for section in ("gpu", "dependency", "runtime"):
            self.assertEqual(dockerfile.count(f"# {section}-image-inputs-begin"), 1)
            self.assertEqual(dockerfile.count(f"# {section}-image-inputs-end"), 1)
        self.assertIn("FROM ${PYTHON_IMAGE} AS gpu", dockerfile)
        self.assertIn("FROM ${GPU_BASE} AS dependencies", dockerfile)
        self.assertIn("FROM scratch AS runtime-overlay", dockerfile)
        self.assertIn("FROM ${RUNTIME_BASE} AS runtime", dockerfile)
        runtime = dockerfile.split("FROM ${RUNTIME_BASE} AS runtime", maxsplit=1)[1]
        instructions = [line.strip() for line in runtime.splitlines() if line and not line.startswith(" ")]
        self.assertFalse(any(line.startswith("RUN ") for line in instructions))
        self.assertEqual(
            [line for line in instructions if line.startswith("COPY ")],
            ["COPY --link --from=runtime-overlay / /"],
        )

    def test_projections_exclude_host_tools_and_isolate_gpu_packages(self) -> None:
        train = Path("containers/train/train-linux-amd64.lock").read_text(encoding="utf-8")
        gpu = Path("containers/train/gpu-linux-amd64.lock").read_text(encoding="utf-8")

        self.assertNotIn("textual==", train)
        self.assertNotIn("wandb-workspaces==", train)
        self.assertIn("torch==2.12.0", gpu)
        for line in gpu.splitlines():
            name = line.split("==", maxsplit=1)[0]
            self.assertTrue(name in {"torch", "triton"} or name.startswith(("cuda-", "nvidia-")))

    def test_workflows_publish_only_immutable_layer_tags(self) -> None:
        dependency = Path(".github/workflows/rlab-train-dependencies.yml").read_text(
            encoding="utf-8"
        )
        runtime = Path(".github/workflows/rlab-train-image.yml").read_text(encoding="utf-8")

        self.assertIn("branches-ignore: [main]", dependency)
        self.assertIn(":build-${{ needs.metadata.outputs.gpu_key }}", dependency)
        self.assertIn(":build-${{ needs.dependency-metadata.outputs.dependency_key }}", dependency)
        self.assertIn("uses: ./.github/workflows/rlab-train-dependencies.yml", runtime)
        self.assertIn('docker buildx imagetools inspect "$dependency_image"', runtime)
        self.assertIn("runtime-${{ steps.build_meta.outputs.runtime_input_sha256 }}", runtime)
        self.assertIn('"schema_version": 5', runtime)
        self.assertNotIn("buildcache", dependency + runtime)
        self.assertNotIn("cache-to:", dependency + runtime)

    def test_image_receipt_precedes_modal_readiness(self) -> None:
        workflow = Path(".github/workflows/rlab-train-image.yml").read_text(encoding="utf-8")
        modal = Path(".github/workflows/rlab-modal-eval.yml").read_text(encoding="utf-8")

        build = workflow.split("  build:", maxsplit=1)[1].split(
            "  deploy-modal-evaluator:", maxsplit=1
        )[0]
        self.assertIn("name: rlab-train-image", build)
        self.assertIn('"schema_version": 5', build)
        self.assertIn('"gpu_plan_sha256"', build)
        self.assertIn("workflow_call:", modal)
        self.assertIn("name: rlab-modal-eval-readiness", modal)


if __name__ == "__main__":
    unittest.main()
