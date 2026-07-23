from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab.database_tls import postgres_ssl_options


class DatabaseTlsTests(unittest.TestCase):
    def test_verify_full_uses_certifi_when_no_root_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            certificate = Path(temporary) / "cacert.pem"
            certificate.write_text("test certificate", encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("certifi.where", return_value=str(certificate)),
            ):
                options = postgres_ssl_options()

        self.assertEqual(options["sslmode"], "verify-full")
        self.assertEqual(options["sslrootcert"], str(certificate.resolve()))

    def test_explicit_missing_root_fails_before_database_connection(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "RLAB_DATABASE_SSLMODE": "verify-full",
                    "PGSSLROOTCERT": "/missing/rlab-ca.pem",
                },
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "does not exist"),
        ):
            postgres_ssl_options()

    def test_nonverifying_mode_does_not_require_a_ca_bundle(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RLAB_DATABASE_SSLMODE": "require"},
            clear=True,
        ):
            self.assertEqual(postgres_ssl_options(), {"sslmode": "require"})
