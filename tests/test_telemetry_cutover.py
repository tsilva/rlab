from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rlab.telemetry_cutover import (
    read_cutover_marker,
    require_runtime_admission,
    write_cutover_marker,
)
from rlab.telemetry_schema import TELEMETRY_V2_SCHEMA_SQL


class TelemetryCutoverTests(unittest.TestCase):
    def test_marker_is_fsynced_and_digest_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutover.json"
            written = write_cutover_marker(path, generation=7, reason="test")
            self.assertEqual(written, read_cutover_marker(path))

    def test_fenced_runtime_requires_exact_generation_and_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cutover.json"
            write_cutover_marker(path, generation=7, reason="test")
            environment = {
                "RLAB_TELEMETRY_CUTOVER_MARKER_PATH": str(path),
                "RLAB_TELEMETRY_ADMISSION_FENCED": "1",
                "RLAB_TELEMETRY_CUTOVER_GENERATION": "7",
            }
            require_runtime_admission(
                {"telemetry_protocol_version": 2, "telemetry_generation": 7},
                environment=environment,
            )
            with self.assertRaisesRegex(RuntimeError, "generation fence"):
                require_runtime_admission(
                    {"telemetry_protocol_version": 2, "telemetry_generation": 6},
                    environment=environment,
                )
            with self.assertRaisesRegex(RuntimeError, "generation fence"):
                require_runtime_admission(
                    {"telemetry_protocol_version": 1, "telemetry_generation": 7},
                    environment=environment,
                )

    def test_unfenced_development_runtime_remains_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            require_runtime_admission(
                {"telemetry_protocol_version": 1},
                environment={},
                marker_path=Path(directory) / "absent.json",
            )

    def test_delete_guard_avoids_reserved_plpgsql_authorization_identifier(self):
        self.assertIn("delete_authorization TEXT", TELEMETRY_V2_SCHEMA_SQL)
        self.assertNotIn("\n  authorization TEXT;", TELEMETRY_V2_SCHEMA_SQL)

    def test_bounded_projection_schema_has_cursors_components_and_watermarks(self):
        for fragment in (
            "CREATE TABLE IF NOT EXISTS wandb_projection_source_cursors",
            "CREATE TABLE IF NOT EXISTS wandb_publication_components",
            "submitted_through_ordinal",
            "projection_chain_sha256",
            "close_ordinal",
            "ambiguous_at",
            "'history', 'artifacts', 'terminal'",
        ):
            self.assertIn(fragment, TELEMETRY_V2_SCHEMA_SQL)


if __name__ == "__main__":
    unittest.main()
