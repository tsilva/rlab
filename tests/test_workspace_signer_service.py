from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path

from rlab.workspace_signer_service import (
    SIGNER_LABEL,
    SignerServicePaths,
    render_launch_daemon,
    validate_launch_daemon_payload,
)


class WorkspaceSignerServiceTests(unittest.TestCase):
    def test_launch_daemon_keeps_database_and_private_key_out_of_plist_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = SignerServicePaths(
                python=Path("/usr/bin/python3"),
                state_dir=root,
                plist=root / "signer.plist",
            )
            payload = plistlib.loads(render_launch_daemon(paths, key_revision="key-v1"))
            validate_launch_daemon_payload(payload, paths, key_revision="key-v1")
            self.assertEqual(payload["Label"], SIGNER_LABEL)
            self.assertNotIn("EnvironmentVariables", payload)
            self.assertNotIn("WORKSPACE_SIGNER_DATABASE_URL", str(payload))
            self.assertIn(str(paths.private_key), payload["ProgramArguments"])
            self.assertIn(str(paths.database_env), payload["ProgramArguments"])


if __name__ == "__main__":
    unittest.main()
