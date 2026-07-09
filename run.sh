#!/bin/bash
# SimCCL one-command standalone run (root wrapper).
# Runs the prebuilt standalone binary. Build it first with ./build.sh
#
# Usage:
#   ./run.sh --op AllReduce --size 4194304 \
#            --nRanks 8 --nNodes 1 --gpus_per_node 8 [--gpu_type H20]
#   ./run.sh -w <workload.txt> --nRanks 16 --nNodes 2 --gpus_per_node 8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/standalone/build/simccl-standalone"

if [ ! -x "$BIN" ]; then
  echo "[run.sh] standalone binary not found: $BIN" >&2
  echo "[run.sh] Build it first:  bash build.sh v2.30" >&2
  exit 1
fi

exec "$BIN" "$@"
