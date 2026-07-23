from __future__ import annotations

import os
from pathlib import Path


VERIFYING_SSL_MODES = frozenset({"verify-ca", "verify-full"})


def postgres_ssl_options() -> dict[str, str]:
    """Return fail-closed PostgreSQL TLS options with a deterministic CA bundle."""

    sslmode = str(os.environ.get("RLAB_DATABASE_SSLMODE") or "verify-full").strip()
    options = {"sslmode": sslmode}
    root_cert = str(os.environ.get("PGSSLROOTCERT") or "").strip()
    if not root_cert and sslmode in VERIFYING_SSL_MODES:
        try:
            import certifi
        except ImportError as exc:  # pragma: no cover - packaging contract
            raise RuntimeError("verified PostgreSQL TLS requires certifi or PGSSLROOTCERT") from exc
        root_cert = str(certifi.where()).strip()
    if root_cert:
        if root_cert != "system":
            path = Path(root_cert).expanduser()
            if not path.is_file():
                raise RuntimeError(f"PostgreSQL TLS root certificate does not exist: {path}")
            root_cert = str(path.resolve())
        options["sslrootcert"] = root_cert
    return options
