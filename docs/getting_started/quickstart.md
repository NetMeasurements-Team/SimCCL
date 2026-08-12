# SimCCL Standalone Guide

> [中文版](../CN/getting_started/quickstart.md)

## What is Standalone Mode?

SimCCL standalone is a self-contained binary that generates FlowModel CSV output without linking or running ns3 network simulation. It compiles only the MockNccl translation layer files and a minimal workload parser, producing the same `ncclFlowModel_detailed_flows.csv` as the full simulator.

This mode exists to satisfy the GitHub PR requirement: SimCCL should be independently runnable and testable without the ns3 dependency chain.

## Building

```bash
cd SimCCL/src

# Build with v2.30 mock (default)
bash build.sh v2.30

# Build with v2.20 mock
bash build.sh v2.20

# Or manual CMake
mkdir -p build && cd build
cmake .. -DMOCK_VERSION=v2.30
make -j$(nproc)
```

The output binary is `build/simccl-standalone` (~300KB, no ns3 dependency).

## Running

### Mode 1: Single Collective Operation

Specify a single (op, size) combination via command-line arguments:

```bash
./simccl-standalone \
  --op AllReduce \
  --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 \
  --gpu_type H20
```

Required parameters:
- `--op`: AllReduce, AllGather, ReduceScatter, or AlltoAll
- `--size`: Data size in bytes
- `--nRanks`: Total number of GPU ranks
- `--nNodes`: Number of nodes
- `--gpus_per_node`: GPUs per node

Optional:
- `--gpu_type`: H20 (default), H100, H800, A100, A800

### Mode 2: Workload File

Parse a SimAI workload file and enumerate all layers:

```bash
./simccl-standalone \
  -w example/microAllReduce.txt \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 \
  --gpu_type H20
```

### Environment Variables

All configurable switches for SimCCL standalone:

| Variable | Default | Scope | Description |
|---|---|---|---|
| `MOCK_VERSION` | `v2.30` | CMake (compile-time) | Select mock NCCL version: `v2.20` or `v2.30` |
| `SIMAI_DUMP_DETAILED_FLOWS` | `1` | MockNcclGroup.cc | Enable CSV dump (`0` to disable). Legacy fallback: `AS_DUMP_DETAILED_FLOWS` |
| `SIMAI_PROTO_AWARE` | `1` (v2.30) / `0` (v2.20) | MockNcclGroup.cc | Enable protocol-aware selection (LL/LL128/Simple by size). Legacy: `AS_PROTO_AWARE` |
| `SIMAI_PAT_MIN_BYTES` | `524288` | MockNcclGroup.cc | PAT algorithm lower activation threshold in bytes |
| `SIMAI_PAT_MAX_BYTES` | `1048576` | MockNcclGroup.cc | PAT algorithm upper threshold (above this: RING) |
| `SIMAI_PAT_ENABLE` | `2` | MockNcclGroup.cc | 0=disable PAT, 1=force PAT, 2=auto (dual-threshold) |
| `AS_NVLS_ENABLE` | `0` | MockNcclGroup.cc | Enable NVLS algorithm for AllReduce (H20/H100 workloads) |
| `AS_NVLS_MIN_BYTES` | `2097152` | MockNcclGroup.cc | NVLS activation threshold for AllReduce |
| `AS_LOG_LEVEL` | - | MockNcclLog.cc | Log verbosity level |
| `SIMAI_FORCE_PROTO` | not set | MockNcclGroup.cc | Override protocol: `LL`, `LL128`, or `Simple`. Bypasses auto-selection. For testing. |

### Input Constraints

The topology parameters must satisfy:
- `nRanks == nNodes * gpus_per_node` (enforced at startup)
- `nRanks % gpus_per_node == 0`
- `gpus_per_node <= nRanks`
- PAT triggers when `nNodes > 1 && nRanks == nNodes` (i.e., `gpus_per_node == 1`) for AllGather/ReduceScatter with `size >= SIMAI_PAT_MIN_BYTES`

Example with NVLS enabled:
```bash
AS_NVLS_ENABLE=1 ./simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8
```

## Output Format

The output file `ncclFlowModel_detailed_flows.csv` has the following columns:

```
collective,op,data_size,algorithm,protocol,channel_id,flow_id,src,dest,
flow_size,chunk_id,chunk_count,conn_type,parent_flow_ids,prev_flow_ids
```

- `algorithm`: 0=Tree, 1=Ring, 2=CollNetDirect(*), 3=CollNetChain(*), 4=NVLS, 5=NVLS_TREE, 6=PAT
  - (*) CollNetDirect/CollNetChain: constants exist in SimCCL but auto-selection and flow generation are NOT implemented. In NCCL, these algorithms require CollNet-capable fabric/SHARP hardware; without such hardware, NCCL itself also never selects them.
- `protocol`: 0=LL, 1=LL128, 2=Simple, -1=UNDEF
- `conn_type`: NVLINK, NET, PCI, NVLS, etc.

### Column Descriptions

| Column | Description | NCCL Counterpart |
|---|---|---|
| `collective` | Collective operation name (e.g., AllReduce) | ncclFunc* enum |
| `op` | Reduction operation (sum, etc.) | ncclRedOp_t |
| `data_size` | Total data size in bytes | info->sendbytes |
| `algorithm` | Algorithm index (see above) | info->algorithm (NCCL_ALGO_*) |
| `protocol` | Protocol index (see above) | info->protocol (NCCL_PROTO_*) |
| `channel_id` | Communication channel index | channel->id |
| `flow_id` | Unique flow identifier within this collective | SimCCL internal |
| `src` | Source GPU rank | - |
| `dest` | Destination GPU rank | - |
| `flow_size` | Bytes transferred in this flow | chunkSize |
| `chunk_id` | Chunk index within the collective | - |
| `chunk_count` | Total number of chunks | nChunksPerLoop |
| `conn_type` | Connection type (NVLINK/NET/PCI/NVLS) | connector->transportComm type |
| `parent_flow_ids` | Flow dependencies (data dependency) | SimCCL internal |
| `prev_flow_ids` | Flow ordering (sequential dependency) | SimCCL internal |

## Testing

Run the full-feature test script to validate all configurations:

```bash
cd SimCCL
bash scripts/run_all.sh
# Results: results/standalone_test_summary.csv
```

The script tests 56 cases: 52 valid topology combinations (1n8g, 2n8g, 2n1g-PAT, 1n4g, 2n2g) + 4 invalid inputs (expected failures).

## Known Limitations

> **Checkout layout requirement**: the standalone build includes `astra-sim/system/Common.hh` and references `../../astra-sim-alibabacloud` in CMake, so SimCCL must live inside a SimAI tree (`SimAI/SimCCL/` next to `SimAI/astra-sim-alibabacloud/`). A bare clone of SimCCL alone cannot build the standalone binary.

1. **NVSwitch topology**: NVSwitch nodes are only created when `gpus_per_node > 4`. For smaller configurations, NVSwitch-related group entries are skipped (bounds-checked).

2. **PAT implementation**: The current PAT FlowModel generation (`genPATFlowModels()`) uses a binomial tree state machine for AllGather/ReduceScatter when `nNodes > 1 && gpus_per_node == 1`. PAT only triggers for AllGather/ReduceScatter, NOT AllReduce.

3. **Group type mapping**: In workload file mode, the group type (TP/DP/PP/EP) must match the nRanks/gpus_per_node configuration. If a workload specifies DP communication but DP_size=1, no flows are generated (correct behavior).

4. **No ns3 timing**: This mode only generates FlowModels. It does not simulate network latency, congestion, or end-to-end timing. For full simulation, use the SimAI ns3 binary.

5. **CollNetDirect/CollNetChain not implemented**: SimCCL has algorithm constants and send_lat table entries for CollNet algorithms (indices 2 and 3), but the auto-selection logic, topology modeling, and flow generation are not implemented. In NCCL 2.30, these algorithms depend on CollNet-capable transport plugins and SHARP fabric hardware (`collNetChainGraph->nChannels > 0` check in `init.cc`). Without such hardware, NCCL itself also does not select these algorithms.
