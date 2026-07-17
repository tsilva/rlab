from __future__ import annotations

import contextlib
import io
import unittest

from rlab.main import main


class PublicCliHelpTests(unittest.TestCase):
    def test_delegated_help_uses_complete_public_command(self) -> None:
        cases = (
            (("experiment", "launch", "--help"), "usage: rlab experiment launch"),
            (("experiment", "follow", "--help"), "usage: rlab experiment follow"),
            (("eval", "--help"), "usage: rlab eval"),
            (("eval", "run", "--help"), "usage: rlab eval run"),
            (("eval", "modal", "status", "--help"), "usage: rlab eval modal status"),
            (("play", "--help"), "usage: rlab play"),
            (("import-roms", "--help"), "usage: rlab import-roms"),
            (("benchmark", "run", "--help"), "usage: rlab benchmark run"),
            (("validate", "--help"), "usage: rlab validate"),
            (("env", "preflight", "--help"), "usage: rlab env preflight"),
            (("leaders", "runs", "--help"), "usage: rlab leaders runs"),
            (("reports", "plan", "--help"), "usage: rlab reports plan"),
            (("fleet", "service", "status", "--help"), "usage: rlab fleet service status"),
            (("fleet", "service", "watch", "--help"), "usage: rlab fleet service watch"),
        )
        for argv, expected_usage in cases:
            with self.subTest(command=" ".join(argv)):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    main(list(argv))
                self.assertEqual(raised.exception.code, 0)
                self.assertTrue(stdout.getvalue().startswith(expected_usage), stdout.getvalue())

    def test_launch_help_describes_exact_source_runtime_resolution(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["experiment", "launch", "--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        normalized_help = " ".join(help_text.split())
        self.assertIn("current clean source revision", normalized_help)
        self.assertIn("never falls back to an older image", normalized_help)
        self.assertNotIn("defaults to latest", normalized_help)

    def test_eval_and_play_help_are_sb3_backend_neutral(self) -> None:
        for command in (("eval", "run"), ("play",)):
            with self.subTest(command=command):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    main([*command, "--help"])
                self.assertEqual(raised.exception.code, 0)
                help_text = stdout.getvalue()
                self.assertIn("rlab", help_text)
                self.assertNotIn("SB3", help_text)
                self.assertNotIn("PPO", help_text)


if __name__ == "__main__":
    unittest.main()
