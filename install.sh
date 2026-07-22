#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
    exit 1
fi

EXTRA_NAME=""
if (( $# > 0 )); then
    if (( $# != 2 )) || [[ "$1" != "--extra" ]]; then
        echo "usage: ./install.sh [--extra dataset|dataset-minari]" >&2
        exit 2
    fi
    case "$2" in
        dataset|dataset-minari)
            EXTRA_NAME="$2"
            ;;
        *)
            echo "unknown extra: $2 (expected dataset or dataset-minari)" >&2
            exit 2
            ;;
    esac
fi

PACKAGE_TARGET="."
if [[ -n "$EXTRA_NAME" ]]; then
    PACKAGE_TARGET=".[$EXTRA_NAME]"
fi

CONSTRAINTS="$(mktemp "${TMPDIR:-/tmp}/rlab-lock.XXXXXX.txt")"
trap 'rm -f "$CONSTRAINTS"' EXIT
if [[ -n "$EXTRA_NAME" ]]; then
    uv export \
        --frozen \
        --no-dev \
        --no-emit-project \
        --no-hashes \
        --extra "$EXTRA_NAME" \
        --output-file "$CONSTRAINTS"
else
    uv export \
        --frozen \
        --no-dev \
        --no-emit-project \
        --no-hashes \
        --output-file "$CONSTRAINTS"
fi

if uv tool list | grep -q "^rlab "; then
    echo "Existing rlab tool detected; reinstalling from the frozen lock."
    uv tool install --project . "$PACKAGE_TARGET" \
        -e \
        --force \
        --constraints "$CONSTRAINTS"
else
    echo "Installing rlab as an editable uv tool from the frozen lock."
    uv tool install --project . "$PACKAGE_TARGET" \
        -e \
        --constraints "$CONSTRAINTS"
fi

rlab --help >/dev/null
uv tool list | grep -A1 '^rlab '
