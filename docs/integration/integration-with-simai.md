# SimCCL Integration with SimAI Main Library

> [中文版](../CN/integration/integration-with-simai.md)

## Applicable Scenario

This document describes environment variables and configuration relevant when SimCCL is used as part of the **full SimAI ns3 simulation** (not in standalone mode). These variables are either:
- Consumed by the astra-sim ns3 frontend layer (not by SimCCL code itself), or
- Only meaningful when running the full SimAI pipeline

For SimCCL standalone environment variables, see [env-variables.md](../configuration/env-variables.md).

---

## Integration Environment Variables

### SIMAI_DUMP_DETAILED_FLOWS (Integration Usage)

| Property | Value |
|---|---|
| Code location | `SimCCL/mock/v2.30/MockNcclGroup.cc` L327-332 |
| Default | `1` (enabled) |
| Standalone behavior | Controls CSV output (same as integration) |
| Integration behavior | When running full SimAI ns3 simulation, you may want to disable CSV generation for performance: set `SIMAI_DUMP_DETAILED_FLOWS=0` |

**Note**: This variable is also documented in [env-variables.md](../configuration/env-variables.md) with a brief standalone-focused description. This section provides the full integration context.

#### Quick Commands

```bash
# Prerequisites: compile SimAI-Simulation
cd SimAI/
./scripts/build.sh -c ns3

# Generate topology
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# Run (default SIMAI_DUMP_DETAILED_FLOWS=1, generates CSV)
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# Disable detailed_flows CSV (improves large-scale simulation performance)
SIMAI_DUMP_DETAILED_FLOWS=0 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# AS_SEND_LAT override experiment (unit: nanoseconds)
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

For detailed parameter explanations, see [Quick Start Guide](../../docs/getting_started/quickstart.md).

### AS_SEND_LAT

| Property | Value |
|---|---|
| Code location | `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h` L168-177 |
| Default | Not set (use send_latency_table) |
| Scope | ns3 network frontend only — NOT consumed by SimCCL |
| Priority | **Highest** — overrides all table-based send_lat lookups |

When set, all flows use this single value (in nanoseconds) as their send latency, regardless of algorithm, protocol, or link type.

**Use cases**:
- A/B experiments: compare simulation results with different send_lat values
- Quick calibration: set a known value and compare with real-machine measurements
- Debugging: eliminate send_lat variability to isolate other factors

**Interaction with send_latency_table**: The table in `entry.h` provides per-(algorithm, protocol, link_type) send latency values. `AS_SEND_LAT` overrides ALL table lookups when set. For detailed analysis of the table mechanism, see `SimAI/docs/configuration/send-lat-analysis.md`.

### AS_DUMP_DETAILED_FLOWS (Legacy Fallback)

Legacy name for `SIMAI_DUMP_DETAILED_FLOWS`. Kept for backward compatibility with external scripts.

---

## FlowModel CSV Data Flow

```
SimCCL (MockNcclGroup.cc)
  → generates ncclFlowModel_detailed_flows.csv
  → CSV contains: algorithm, protocol, flow_size, src, dest, conn_type, etc.

ns3 Frontend (entry.h: SendFlow())
  → reads flowTag from CSV (algorithm, protocol, gpus_per_node)
  → looks up send_lat_table[algo][proto] based on link type
  → applies AS_SEND_LAT override if set
  → starts ns3 application connection with computed delay
```

---

## NCCL Native Variables (Real-Machine Testing)

These are standard NCCL environment variables, used only during real-machine cross-node tests. They are **NOT consumed by SimCCL mock** — they configure the real NCCL library.

| Variable | Value | Description |
|---|---|---|
| `NCCL_DEBUG` | `INFO` / `TRACE` | Print NCCL algorithm/protocol selection info. TRACE for PAT step-level details. |
| `NCCL_DEBUG_SUBSYS` | `COLL,TUNING,GRAPH` | Filter NCCL debug output to specific subsystems |
| `NCCL_IB_DISABLE` | `0` | Enable InfiniBand (default) |
| `NCCL_ALGO` | `PAT` / `RING` / `TREE` | Force specific algorithm (for testing) |
| `CUDA_VISIBLE_DEVICES` | `0` | Restrict to 1 GPU per node (for PAT testing) |

---

## End-to-End Running Guide

For complete compilation, topology generation, and simulation running instructions, see the SimAI main documentation:
- [Installation Guide](../../docs/getting_started/installation.md)
- [Quick Start Guide](../../docs/getting_started/quickstart.md)
- [Environment Variables](../../docs/configuration/env-variables.md)
- [Build Options](../../docs/configuration/build-options.md)

---

## Difference from Standalone Mode

| Aspect | Standalone | Full SimAI Integration |
|---|---|---|
| Binary | `simccl-standalone` | `astra_ns3` (full simulator) |
| Network simulation | None | ns3 full simulation |
| Output | CSV only | CSV + EndToEnd.csv + timing |
| `AS_SEND_LAT` | Not applicable | Overrides send_lat |
| NCCL native vars | Not applicable | Used in real-machine validation |
| `SIMAI_DUMP_DETAILED_FLOWS` | Controls CSV (default ON) | Controls CSV (may want OFF for perf) |

---

> Last edited: 2026-06-25
