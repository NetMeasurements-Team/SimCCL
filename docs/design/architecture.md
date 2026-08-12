# SimCCL Architecture

> [中文版](../CN/design/architecture.md)

## Overview

SimCCL is the NCCL communication translation layer within the SimAI network simulator. It converts NCCL collective decisions (algorithm/protocol/channel/chunk) into FlowModels — a structured description of network flows — that drive the ns3 packet-level simulation or can be exported as standalone CSV for offline analysis.

SimCCL does NOT implement actual GPU kernels or network I/O. It translates "what NCCL would decide" into "what flows would result", enabling accurate communication modeling without running real NCCL.

## Module Structure

```
SimCCL/
├── mock/
│   ├── v2.20/          # NCCL v2.20.5 translation semantics (frozen baseline)
│   │   ├── MockNcclGroup.cc/h    # Core: group init, algorithm selection, FlowModel generation
│   │   ├── MockNcclChannel.cc/h  # Channel topology: ring/tree/NVLS construction
│   │   ├── MockNccl.h            # Constants: algorithms, protocols, latency tables
│   │   ├── MockNcclLog.cc/h      # Logging infrastructure
│   │   └── MockNcclQps.h         # Queue pair simulation
│   ├── v2.30/          # NCCL v2.30.7 translation semantics (active development)
│   │   └── (same files, with PAT algorithm + protocol-aware + CSV algo/proto columns)
│   └── README.md
├── src/         # Independent binary (no ns3 dependency)
│   ├── CMakeLists.txt
│   ├── main.cc
│   ├── workload_parser.cc/h
│   └── build.sh
└── docs/               # This documentation
```

## Version Selection

At build time, the `SIMAI_NCCL_VERSION` environment variable selects which mock version to compile:

```bash
# Default: v2.30 (protocol-aware, PAT support)
./build.sh -c ns3

# Select v2.20 mock (legacy)
SIMAI_NCCL_VERSION=v2.20 ./build.sh -c ns3
```

The selected version's files are flat-copied into the ns3 application tree by `build/astra_ns3/build.sh`. CMakeLists.txt globs `SimCCL/mock/*.cc` and is unaffected by the version switch.

## Key Differences: v2.20 vs v2.30

| Feature | v2.20 | v2.30 |
|---|---|---|
| PAT algorithm | Constants defined but never selected | Selected when `nRanks == nNodes && nNodes > 1 && size >= 512KB` |
| Protocol awareness | `info->protocol = NCCL_PROTO_UNDEF` (always) | Protocol assigned based on message size (LL/LL128/Simple) |
| CSV output | Standard columns | Adds `algorithm`, `protocol` columns for provenance |
| PROTO_AWARE default | N/A (no switch) | Default ON (`SIMAI_PROTO_AWARE=1`) |

## Data Flow

The translation pipeline has three stages, each with a clear input/output contract:

**Stage 1: Algorithm + Protocol Selection** (`get_algo_proto_info()`)
- Input: (GroupType, rank, ComType, data_size)
- Output: `ncclInfo*` with `algorithm` (RING/TREE/NVLS/PAT) and `protocol` (LL/LL128/Simple)
- Decision factors: GPU type, NVSwitch presence, message size, number of nodes/ranks

**Stage 2: FlowModel Generation** (`genFlowModels()` → `genAllReduceFlowModels()` etc.)
- Input: (GroupType, rank, ComType, data_size) + algorithm from Stage 1
- Output: `map<rank, shared_ptr<FlowModels>>` — per-rank flow descriptions
- Each flow specifies: src, dest, size, channel_id, chunk_id, connection type

**Stage 3: CSV Dump** (`dumpDetailedFlowModels()`)
- Input: FlowModels + ncclInfo
- Output: `ncclFlowModel_detailed_flows.csv`
- Controlled by `SIMAI_DUMP_DETAILED_FLOWS` (default: enabled)

## Integration with ns3

In the full simulation mode, FlowModels are consumed by `NcclFlowModel` (formerly NcclTreeFlowModel) in `astra-sim/system/collective/`. This class converts FlowModels into ns3 packet send requests with `flowTag` carrying algorithm, protocol, and gpus_per_node for send_lat bucketing in `entry.h`.

## send_lat Bucketing

The ns3 entry point (`entry.h`) uses two lookup tables to determine per-flow send latency:

| Table | When Used | Source |
|---|---|---|
| `send_lat_table_nvlink[7][3]` | Same-node (intra-node) flows | nccl-2.30 tuning.cc baseLat + hwLat[NVLINK] |
| `send_lat_table_net[7][3]` | Cross-node (inter-node) flows | nccl-2.30 tuning.cc baseLat + hwLat[NET] |

Link type detection uses `flowTag.gpus_per_node` (from Sys configuration) to determine if src and dst are on the same node: `same_node = (src / gpus_per_node) == (dst / gpus_per_node)`.

`AS_SEND_LAT` environment variable overrides all table lookups (highest priority, for A/B experiments).
