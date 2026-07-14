from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab import fleet_service_entrypoint


class FleetServiceEntrypointTests(unittest.TestCase):
    def test_bootstrap_failure_is_persisted_and_notified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            with (
                mock.patch(
                    "rlab.fleet_service.main",
                    side_effect=ImportError("broken working tree"),
                ),
                mock.patch.object(fleet_service_entrypoint, "_notify") as notify,
            ):
                result = fleet_service_entrypoint.main(
                    ["run-once", "--state-dir", str(state_dir)]
                )

            last_pass = json.loads((state_dir / "last-pass.json").read_text())
            health = json.loads((state_dir / "health.json").read_text())

        self.assertEqual(result, 1)
        self.assertEqual(last_pass["status"], "error")
        self.assertNotIn("broken working tree", last_pass["error"])
        self.assertTrue(health["alert_active"])
        notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
