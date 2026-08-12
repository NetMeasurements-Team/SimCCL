# SimCCL Environment Variables Reference

> [中文版](../CN/configuration/env-variables.md)

This document covers environment variables consumed directly by SimCCL code (MockNcclGroup.cc, MockNcclLog.h). For variables related to SimAI ns3 integration (AS_SEND_LAT, NCCL native variables, etc.), see [integration-with-simai.md](../integration/integration-with-simai.md).

## SIMAI_* Variables (New, Recommended)

| Variable | Default | Scope | Impact on Output | Description |
|---|---|---|---|---|
| `SIMAI_NCCL_VERSION` | `v2.20` | build.sh (compile-time) | Build-time only | Select mock NCCL version (v2.20 or v2.30) |
| `SIMAI_DUMP_DETAILED_FLOWS` | `1` | MockNcclGroup.cc | **Yes** (on/off switch) | Control CSV output switch. `0`=disable, `1`=enable |
| `SIMAI_PROTO_AWARE` | `0` (v2.20) / `1` (v2.30) | MockNcclGroup.cc | **Yes** (affects ns3 latency via send_lat table) | Enable protocol assignment (LL/LL128/Simple) based on message size. Affects send_lat table lookup in ns3. |
| `SIMAI_PAT_MIN_BYTES` | `524288` (512KB) | MockNcclGroup.cc (v2.30 only) | **Yes** (algorithm selection) | PAT algorithm lower activation threshold in bytes |
| `SIMAI_PAT_MAX_BYTES` | `1048576` (1MB) | MockNcclGroup.cc (v2.30 only) | **Yes** (algorithm selection) | PAT algorithm upper threshold. Above this, RING is used |
| `SIMAI_PAT_ENABLE` | `2` | MockNcclGroup.cc (v2.30 only) | **Yes** (algorithm selection) | 0=disable PAT, 1=force PAT, 2=auto (dual-threshold) |
| `SIMAI_FORCE_PROTO` | not set | MockNcclGroup.cc (v2.30 only) | **Yes** (affects send_lat) | Override protocol: `LL`, `LL128`, or `Simple`. Bypasses auto-selection. For testing. |

### SIMAI_PROTO_AWARE Impact Analysis

`SIMAI_PROTO_AWARE` controls the `protocol` field in the FlowModel CSV output. Its call chain:

```
get_algo_proto_info() → info->protocol = LL/LL128/Simple or UNDEF
  → dumpDetailedFlowModels() → CSV "protocol" column
  → entry.h: SendFlow() → send_lat_table[algo][proto] → send_lat value
```

**Current status**: The protocol value IS consumed by `entry.h:SendFlow()` for send_lat table lookup. When `SIMAI_PROTO_AWARE=0`, `protocol=-1(UNDEF)` causes a fallback to default 6000ns send_lat. When `SIMAI_PROTO_AWARE=1`, the per-(algo,proto) table value is used (e.g., Ring+LL=7200ns for NVLINK).

**Conclusion**: `SIMAI_PROTO_AWARE` DOES affect end-to-end simulation latency through the send_lat table bucketing mechanism. It is NOT merely a CSV annotation.

## AS_* Variables (Legacy, Still Active)

| Variable | Default | Scope | Description |
|---|---|---|---|
| `AS_NVLS_ENABLE` | `0` | MockNcclGroup.cc | Enable NVLS algorithm for AllReduce on H20/H100/H800 |
| `AS_NVLS_MIN_BYTES` | `2097152` (2MB) | MockNcclGroup.cc | NVLS activation threshold for AllReduce |
| `AS_LOG_LEVEL` | - | MockNcclLog.h | Log verbosity level |

## Fallback Behavior

New `SIMAI_*` variables fall back to legacy `AS_*` names if the new name is not set:

```
SIMAI_DUMP_DETAILED_FLOWS → AS_DUMP_DETAILED_FLOWS (fallback)
SIMAI_PROTO_AWARE         → AS_PROTO_AWARE (fallback)
```

## Priority Rules

1. **`SIMAI_*`** takes precedence over **`AS_*`** when both are set
2. **New variables set** → new value used. **Only old set** → old value used. **Neither set** → default applies

## Quick Usage Examples

### SimCCL Standalone

```bash
cd SimCCL/src && bash build.sh v2.30

# Basic run (default env)
./build/simccl-standalone --op AllGather --size 524288 --nRanks 4 --nNodes 4 --gpus_per_node 1

# Force PAT algorithm
SIMAI_PAT_ENABLE=1 ./build/simccl-standalone --op AllGather --size 524288 --nRanks 2 --nNodes 2 --gpus_per_node 1

# Disable CSV output
SIMAI_DUMP_DETAILED_FLOWS=0 ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 4 --nNodes 4 --gpus_per_node 1

# Force Simple protocol
SIMAI_FORCE_PROTO=Simple ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 2 --nNodes 2 --gpus_per_node 1
```

### Full SimAI Integration

```bash
cd SimAI/
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

For detailed explanations, see [integration-with-simai.md](../integration/integration-with-simai.md).
