from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SIGNER_LABEL = "com.rlab.workspace-signer"
DEFAULT_ROOT = Path("/Library/Application Support/rlab/workspace-signer")
DEFAULT_PLIST = Path("/Library/LaunchDaemons/com.rlab.workspace-signer.plist")


@dataclass(frozen=True)
class SignerServicePaths:
    python: Path
    state_dir: Path = DEFAULT_ROOT
    plist: Path = DEFAULT_PLIST

    @property
    def private_key(self) -> Path:
        return self.state_dir / "private-key.pem"

    @property
    def public_key(self) -> Path:
        return self.state_dir.parent / "workspace-signer-public.pem"

    @property
    def database_env(self) -> Path:
        return self.state_dir / "database.env"

    @property
    def status_file(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def stdout(self) -> Path:
        return self.state_dir / "stdout.log"

    @property
    def stderr(self) -> Path:
        return self.state_dir / "stderr.log"


def launch_daemon_payload(paths: SignerServicePaths, *, key_revision: str) -> dict[str, Any]:
    return {
        "Label": SIGNER_LABEL,
        "ProgramArguments": [
            str(paths.python),
            "-m",
            "rlab.workspace_signer",
            "--private-key",
            str(paths.private_key),
            "--database-env-file",
            str(paths.database_env),
            "--key-revision",
            key_revision,
            "--status-file",
            str(paths.status_file),
        ],
        "KeepAlive": True,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "ThrottleInterval": 2,
        "StandardOutPath": str(paths.stdout),
        "StandardErrorPath": str(paths.stderr),
        "Umask": 0o077,
    }


def validate_launch_daemon_payload(
    payload: Mapping[str, Any], paths: SignerServicePaths, *, key_revision: str
) -> None:
    if dict(payload) != launch_daemon_payload(paths, key_revision=key_revision):
        raise ValueError("workspace signer LaunchDaemon payload changed")
    if "EnvironmentVariables" in payload:
        raise ValueError("workspace signer secrets must not be stored in the plist")
    if not all(path.is_absolute() for path in (paths.python, paths.state_dir, paths.plist)):
        raise ValueError("workspace signer paths must be absolute")


def render_launch_daemon(paths: SignerServicePaths, *, key_revision: str) -> bytes:
    payload = launch_daemon_payload(paths, key_revision=key_revision)
    validate_launch_daemon_payload(payload, paths, key_revision=key_revision)
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def _require_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit("workspace signer service installation must run as root")


def install(paths: SignerServicePaths, *, key_revision: str, replace: bool) -> dict[str, Any]:
    _require_root()
    paths.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(paths.state_dir, 0o700)
    os.chmod(paths.state_dir.parent, 0o755)
    if not paths.database_env.is_file():
        raise SystemExit(
            f"create root-only {paths.database_env} containing "
            "WORKSPACE_SIGNER_DATABASE_URL before installation"
        )
    if paths.database_env.stat().st_mode & 0o077:
        raise SystemExit("workspace signer database.env must have mode 0600")
    if not paths.private_key.exists():
        subprocess.run(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:3072",
                "-out",
                str(paths.private_key),
            ],
            check=True,
            capture_output=True,
        )
        os.chmod(paths.private_key, 0o600)
        subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(paths.private_key),
                "-pubout",
                "-out",
                str(paths.public_key),
            ],
            check=True,
            capture_output=True,
        )
        os.chmod(paths.public_key, 0o644)
    private_stat = paths.private_key.stat()
    if private_stat.st_uid != 0 or private_stat.st_mode & 0o077:
        raise SystemExit("workspace signer private key must be root-owned with mode 0600")
    if paths.database_env.stat().st_uid != 0:
        raise SystemExit("workspace signer database.env must be root-owned")
    if paths.plist.exists() and not replace:
        raise SystemExit(f"workspace signer plist already exists: {paths.plist}")
    temporary = paths.plist.with_suffix(".plist.tmp")
    temporary.write_bytes(render_launch_daemon(paths, key_revision=key_revision))
    os.chmod(temporary, 0o644)
    os.replace(temporary, paths.plist)
    subprocess.run(["launchctl", "bootout", "system", str(paths.plist)], check=False)
    subprocess.run(["launchctl", "bootstrap", "system", str(paths.plist)], check=True)
    return {
        "label": SIGNER_LABEL,
        "plist": str(paths.plist),
        "public_key": str(paths.public_key),
        "key_revision": key_revision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="administer the isolated workspace signer")
    commands = parser.add_subparsers(dest="command", required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--key-revision", required=True)
    install_parser.add_argument("--replace", action="store_true")
    install_parser.add_argument("--state-dir", type=Path, default=DEFAULT_ROOT)
    install_parser.add_argument("--plist", type=Path, default=DEFAULT_PLIST)
    args = parser.parse_args(argv)
    if args.command == "install":
        result = install(
            SignerServicePaths(
                python=Path(sys.executable).resolve(),
                state_dir=args.state_dir.expanduser().resolve(),
                plist=args.plist.expanduser().resolve(),
            ),
            key_revision=str(args.key_revision),
            replace=bool(args.replace),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
