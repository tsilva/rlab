from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


def test_provider_runtime_import_does_not_require_stable_retro_atari_vec_env(
    tmp_path: Path,
) -> None:
    (tmp_path / "stable_retro.py").write_text("class RetroVecEnv: pass\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).parents[1] / "src"))
    )

    result = subprocess.run(
        [sys.executable, "-c", "import rlab.env_providers"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("distribution_name", "module_name"),
    [
        ("stable-retro-turbo", "stable_retro"),
        ("supermariobrosnes-turbo", "supermariobrosnes_turbo"),
    ],
)
def test_provider_runtime_is_loaded_from_its_installed_distribution(
    distribution_name: str,
    module_name: str,
) -> None:
    """Prevent sibling development checkouts from shadowing the locked PyPI wheels."""
    distribution = importlib.metadata.distribution(distribution_name)
    distribution_root = Path(distribution.locate_file(".")).resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()

    assert distribution_root == site_packages, (
        f"{distribution_name} metadata is loaded from {distribution_root}, not {site_packages}"
    )
    assert module_path.is_relative_to(site_packages), (
        f"{module_name} is loaded from {module_path}, not the installed PyPI wheel"
    )
