# Adding a New Mock Collective Operation

> [中文版](../CN/development/adding-new-collective.md)

This guide documents the complete workflow for adding a new collective operation to SimCCL, using Broadcast as the reference implementation.

## Prerequisites

- NCCL 2.30 source code (for algorithm selection analysis)
- SimCCL standalone binary compiles with `bash build.sh v2.30`
- Understanding of FlowModel CSV format

## Step-by-Step Workflow

### Step 1: NCCL Source Analysis

Find the target collective in NCCL's tuning logic:
- `nccl-2.30/src/graph/tuning.cc` — algorithm selection and cost model
- `nccl-2.30/src/collectives/<collective>.cc` — implementation details

**Example (Broadcast)**: tuning.cc L294 shows Broadcast only uses Ring:
```cpp
if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
```

### Step 2: ComType Enum

Add the new operation to both enum locations:

1. **astra-sim Common.hh**: `astra-sim-alibabacloud/astra-sim/system/Common.hh`
```cpp
enum class ComType {
  None, Reduce_Scatter, All_Gather, All_Reduce,
  All_to_All, All_Reduce_All_to_All, All_Reduce_NVLS,
  Broadcast  // <-- add here
};
```

2. **MockNcclChannel.h**: `mock/v2.30/MockNcclChannel.h` (if separate enum exists)

### Step 3: Algorithm Selection

In `MockNcclGroup.cc` function `get_algo_proto_info()`, add a branch:
```cpp
case AstraSim::ComType::Broadcast:
  info->algorithm = NCCL_ALGO_RING;  // NCCL only uses Ring for Broadcast
  break;
```

### Step 4: Flow Generation Function

Implement `genBroadcastFlowModels()` in `MockNcclGroup.cc`:
- Follow the pattern of existing functions (e.g., `genAllGatherFlowModels()`)
- Key formula: nSteps = nRanks - 1 (full ring traversal from root)
- Each channel generates independent flows
- Set `conn_type` based on src/dst node (NVLINK for intra-node, NET for inter-node)

### Step 5: Dispatch Entry

In `genFlowModels()` switch statement:
```cpp
case AstraSim::ComType::Broadcast:
  return genBroadcastFlowModels(type, rank, data_size);
```

### Step 6: Header Declaration

In `MockNcclGroup.h`:
```cpp
std::map<int,std::shared_ptr<FlowModels>> genBroadcastFlowModels(GroupType type, int rank, uint64_t data_size);
```

### Step 7: Standalone Parser

In `src/main.cc`, add to `parseOp()`:
```cpp
if (s == "Broadcast") return AstraSim::ComType::Broadcast;
```
And update usage string.

### Step 8: Build Verification

```bash
cd SimCCL/src && bash build.sh v2.30
# Must compile with 0 errors
```

### Step 9: Smoke Test

```bash
./build/simccl-standalone --op Broadcast --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 --gpu_type H20
# Verify: CSV generated with >1 rows
wc -l ncclFlowModel_detailed_flows.csv  # Expected: 57 (header + 56 flows)
```

### Step 10: Real-Machine Calibration

Run the corresponding nccl-tests binary:
```bash
docker exec <container> bash -c "
  cd /path/to/nccl-and-nccl-test &&
  LD_LIBRARY_PATH=nccl-2.30/build/lib \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,TUNING \
  nccl-tests/build/broadcast_perf -b 512K -e 256M -f 2 -g 8 -n 5
"
```

Record: op name, sizes, busbw, time(us), algorithm, protocol, NCCL version, GPU model.

### Step 11: Documentation Updates

Update these files (EN + CN):
- `docs/design/nccl-comparison.md` — algorithm selection table + per-op comparison
- `docs/README.md` — if adding new navigation entries

## Limitations to Document

- root parameter: if standalone doesn't support `--root`, note "fixed root=0"
- Protocol: if NCCL restricts to specific protocols, note in comparison table
- Topology constraints: if only Ring/certain algorithms are valid

## Checklist Before Commit

- [ ] `bash build.sh v2.30` passes
- [ ] Smoke test produces CSV with >1 rows
- [ ] `git diff mock/v2.20/` is empty (v2.20 unchanged)
- [ ] Real-machine data recorded
- [ ] EN and CN docs both updated
- [ ] No hardcoded paths in docs/scripts
