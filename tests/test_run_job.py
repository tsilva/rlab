from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlab.run_job import write_result


class RunJobResultTests(unittest.TestCase):
    def test_write_result_atomically_replaces_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            path = output_dir / "result.json"
            path.write_text('{"status": "old"}\n', encoding="utf-8")

            written = write_result(
                output_dir,
                {
                    "schema_version": 1,
                    "job_id": 7,
                    "launch_id": "train-7-stable",
                    "status": "succeeded",
                },
            )

            self.assertEqual(written, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["job_id"], 7)
            self.assertEqual(list(output_dir.glob(".result-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
