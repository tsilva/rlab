#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PACKAGES=(
    breakout-turbo-env
    stable-retro-turbo
    supermariobrosnes-turbo
)

CUTOFF="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
    exit 1
fi

update_cutoffs() {
    local file="$1"
    [[ -f "$file" ]] || return 0

    local package
    for package in "${PACKAGES[@]}"; do
        if grep -Eq "^${package} = " "$file"; then
            perl -0pi -e "s/^${package} = \"[^\"]*\"/${package} = \"$CUTOFF\"/mg" "$file"
        fi
    done
}

update_cutoffs "$ROOT/pyproject.toml"
update_cutoffs "$ROOT/uv-tool.toml"
update_cutoffs "${UV_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml}"

uv lock \
    --upgrade-package breakout-turbo-env \
    --upgrade-package stable-retro-turbo \
    --upgrade-package supermariobrosnes-turbo

"$ROOT/install.sh" "$@"
