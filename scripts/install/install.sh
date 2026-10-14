#!/bin/sh
# NeoCore: one command on Linux and macOS.
#
#   sh install.sh setup                                 # this machine keeps its own memory
#   sh install.sh connect --url URL --key-file FILE     # use a home hub, VPS or hosted server
#   sh install.sh server init --person NAME --device pc # this machine serves the memory
#
# Makes (or reuses) a private Python in ~/.neocore/venv, installs NeoCore into it and runs
# `neocore` with the arguments given. Running it again upgrades and changes nothing twice.
# NEOCORE_PACKAGE picks what to install (a wheel file, for instance); NEOCORE_PYTHON the Python.
set -eu

PACKAGE="${NEOCORE_PACKAGE:-neocore-memory[embed] @ git+https://github.com/VincentPorquet777/neocore}"
VENV="${NEOCORE_VENV:-$HOME/.neocore/venv}"

fail() { printf 'neocore install: %s\n' "$1" >&2; exit 1; }

[ "$#" -gt 0 ] || fail "say what to set up: setup, connect or server init (see the top of this file)"

find_python() {
    for candidate in "${NEOCORE_PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
        [ -n "$candidate" ] || continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$(dirname "$VENV")"
    if command -v uv >/dev/null 2>&1; then
        uv venv -q --python ">=3.10" "$VENV" || fail "uv could not make $VENV"
    else
        PYTHON="$(find_python)" || fail "NeoCore needs Python 3.10 or newer (or uv): install one and run this again"
        "$PYTHON" -m venv "$VENV" || fail "could not make $VENV (on Debian/Ubuntu: apt install python3-venv)"
    fi
fi

echo "Installing $PACKAGE into $VENV ..."
if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$VENV" uv pip install -q --upgrade "$PACKAGE" || fail "install failed"
else
    "$VENV/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
    "$VENV/bin/python" -m pip install -q --upgrade "$PACKAGE" || fail "install failed"
fi

exec "$VENV/bin/python" -m neocore "$@"
