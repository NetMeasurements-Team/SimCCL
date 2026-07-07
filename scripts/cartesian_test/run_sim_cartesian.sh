#!/bin/bash
# SimCCL Cartesian-product simulation test executor
# Reads case CSV and runs simccl-standalone for each case
# Usage: bash run_sim_cartesian.sh <observe|override> [binary_path]
set -o pipefail

MODE="${1:-observe}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${2:-$(cd "$SCRIPT_DIR/../../standalone/build" && pwd)/simccl-standalone}"
WORKDIR=$(mktemp -d /tmp/simccl_cartesian_XXXXX)

if [ "$MODE" = "observe" ]; then
  CASES_CSV="$SCRIPT_DIR/configs/sim_cases_observe.csv"
  RESULTS_DIR="$SCRIPT_DIR/results/sim_observe"
elif [ "$MODE" = "override" ]; then
  CASES_CSV="$SCRIPT_DIR/configs/sim_cases_override.csv"
  RESULTS_DIR="$SCRIPT_DIR/results/sim_override"
else
  echo "Usage: $0 <observe|override>"; exit 1
fi

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.csv"
echo "case_id,op,size,mock_version,gpus_per_node,nNodes,nRanks,gpu_type,force_proto,exit_code,csv_lines,auto_algo,auto_proto,status" > "$SUMMARY"

PASS=0; FAIL=0; TOTAL=0; SKIP=0

echo "======================================"
echo "SimCCL Cartesian Test: mode=$MODE"
echo "Binary: $BINARY"
echo "Cases: $CASES_CSV"
echo "======================================"

# Need v2.20 binary too for version switching
BINARY_V220=""
V220_BUILD="$(cd "$SCRIPT_DIR/../../standalone" && pwd)/build_v220"
if [ -f "$V220_BUILD/simccl-standalone" ]; then
  BINARY_V220="$V220_BUILD/simccl-standalone"
fi

# Read CSV (skip header), strip \r from Windows line endings
tail -n +2 "$CASES_CSV" | tr -d '\r' | while IFS=, read -r CASE_ID OP SIZE MOCK_VER GPN NNODES NRANKS GTYPE FORCE_PROTO; do
  TOTAL=$((TOTAL+1))
  rm -f "$WORKDIR/ncclFlowModel_detailed_flows.csv"

  # Select binary based on mock version
  BIN="$BINARY"
  if [ "$MOCK_VER" = "v2.20" ] && [ -n "$BINARY_V220" ]; then
    BIN="$BINARY_V220"
  elif [ "$MOCK_VER" = "v2.20" ] && [ -z "$BINARY_V220" ]; then
    echo "[$CASE_ID] SKIP (no v2.20 binary)"
    SKIP=$((SKIP+1))
    echo "$CASE_ID,$OP,$SIZE,$MOCK_VER,$GPN,$NNODES,$NRANKS,$GTYPE,$FORCE_PROTO,-1,0,,,SKIP_NO_BINARY" >> "$SUMMARY"
    continue
  fi

  # Set force proto if specified
  ENV_PREFIX=""
  if [ -n "$FORCE_PROTO" ]; then
    ENV_PREFIX="SIMAI_FORCE_PROTO=$FORCE_PROTO"
  fi

  # Run
  OUTPUT=$(cd "$WORKDIR" && env $ENV_PREFIX "$BIN" --op "$OP" --size "$SIZE" --nRanks "$NRANKS" --nNodes "$NNODES" --gpus_per_node "$GPN" --gpu_type "$GTYPE" 2>&1)
  EC=$?

  CSV_LINES=0; ALGO=""; PROTO=""
  if [ -f "$WORKDIR/ncclFlowModel_detailed_flows.csv" ]; then
    CSV_LINES=$(wc -l < "$WORKDIR/ncclFlowModel_detailed_flows.csv")
    if [ "$CSV_LINES" -gt 1 ]; then
      ALGO=$(tail -1 "$WORKDIR/ncclFlowModel_detailed_flows.csv" | cut -d, -f4)
      PROTO=$(tail -1 "$WORKDIR/ncclFlowModel_detailed_flows.csv" | cut -d, -f5)
    fi
  fi

  STATUS="PASS"
  [ "$EC" -ne 0 ] && STATUS="FAIL"
  # Empty CSV with exit 0: mark as SKIP (degenerate topology, e.g., nRanks<=2 gpn=1)
  [ "$CSV_LINES" -le 1 ] && [ "$EC" -eq 0 ] && STATUS="SKIP_EMPTY_CSV"
  [ "$STATUS" = "PASS" ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

  echo "$CASE_ID,$OP,$SIZE,$MOCK_VER,$GPN,$NNODES,$NRANKS,$GTYPE,$FORCE_PROTO,$EC,$CSV_LINES,$ALGO,$PROTO,$STATUS" >> "$SUMMARY"

  if [ $((TOTAL % 50)) -eq 0 ]; then
    echo "[progress] $TOTAL cases done ($PASS pass, $FAIL fail, $SKIP skip)"
  fi
done

rm -rf "$WORKDIR"

# Count final results from CSV
TOTAL_LINES=$(tail -n +2 "$SUMMARY" | wc -l)
PASS_COUNT=$(grep -c ",PASS$" "$SUMMARY" || true)
FAIL_COUNT=$(grep -c ",FAIL$" "$SUMMARY" || true)
SKIP_COUNT=$(grep -c ",SKIP" "$SUMMARY" || true)

echo ""
echo "======================================"
echo "Results: PASS=$PASS_COUNT FAIL=$FAIL_COUNT SKIP=$SKIP_COUNT TOTAL=$TOTAL_LINES"
echo "Summary: $SUMMARY"
echo "======================================"
