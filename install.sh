#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
    exit 1
fi

CONSTRAINTS="$(mktemp "${TMPDIR:-/tmp}/rlab-lock.XXXXXX.txt")"
trap 'rm -f "$CONSTRAINTS"' EXIT
uv export \
    --frozen \
    --no-dev \
    --no-emit-project \
    --no-hashes \
    --output-file "$CONSTRAINTS"

if uv tool list | grep -q "^rlab "; then
    echo "Existing rlab tool detected; reinstalling from the frozen lock."
    uv tool install --project . . \
        -e \
        --force \
        --constraints "$CONSTRAINTS" \
        "$@"
else
    echo "Installing rlab as an editable uv tool from the frozen lock."
    uv tool install --project . . \
        -e \
        --constraints "$CONSTRAINTS" \
        "$@"
fi

rlab --help >/dev/null
uv tool list | grep -A1 '^rlab '
