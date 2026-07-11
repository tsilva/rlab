from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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
