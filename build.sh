#!/bin/bash
# SimCCL one-command standalone build (root wrapper).
# Delegates to standalone/build.sh so the standalone binary can be built
# directly from the repository root.
#
# Usage:
#   ./build.sh [v2.20|v2.30]   # default: v2.30
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STANDALONE_DIR="$SCRIPT_DIR/standalone"

if [ ! -f "$STANDALONE_DIR/build.sh" ]; then
  echo "[build.sh] standalone/build.sh not found under: $STANDALONE_DIR" >&2
  exit 1
fi

cd "$STANDALONE_DIR"
exec bash build.sh "$@"
