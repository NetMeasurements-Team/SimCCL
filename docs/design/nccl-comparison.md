# NCCL 2.30 vs SimCCL v2.30 Source Comparison

> [中文版](../CN/design/nccl-comparison.md)

## Overview

This document provides a detailed comparison between NCCL 2.30 source code and SimCCL mock v2.30 implementation for each collective operation.

**NCCL 2.30 source**: `nccl-2.30/src/graph/tuning.cc` (algorithm selection, cost model)
**SimCCL mock**: `SimCCL/mock/v2.30/MockNcclGroup.cc` (flow generation, algorithm selection)

---

## Algorithm Selection Comparison

### NCCL 2.30 Cost Model (tuning.cc)

NCCL uses a sophisticated cost model that computes `time = latency + nBytes / (1000 * bandwidth)` for each (algorithm, protocol) combination and selects the minimum.

Key factors:
- `baseLatencies[algo][proto]` — fixed per-algorithm startup cost
- `hwLatencies[hw_type][algo][proto]` — per-hop hardware latency (NVLINK/PCI/NET)
- `bandwidth` — derived from topology graph search (nChannels * per-channel BW)
- Various correction factors (treeCorrectionFactor, PAT 0.75x, etc.)

### SimCCL Algorithm Selection (get_algo_proto_info, L2130-2184)

SimCCL uses simplified threshold-based selection:

| Operation | Condition | Algorithm Selected |
|---|---|---|
| AllReduce (TP, H100/H800/H20) | nRanks>=8 && NVLS enabled && size>=2MB | NVLS |
| AllReduce (all other) | default | Ring |
| AllGather/ReduceScatter | nNodes>1 && nRanks==nNodes && size>=512KB | PAT |
| AllGather/ReduceScatter | default | Ring |
| AllToAll | always | Ring |
| Broadcast | always | Ring (v2.30 only, root=0) |

---

## Per-Operation Comparison

### 1. AllReduce

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Algorithms | Ring, Tree, NVLS, NVLS_TREE, CollNet | Ring, NVLS (Tree code exists but not auto-selected) |
| Tree selection | Cost model compares Tree vs Ring latency | **Not auto-selected** (L714: TREE falls through to Ring) |
| Tree flow gen | Binary tree up+down phases | `genAllReduceTreeFlowModels()` exists (L991) but has gp_idx bug |
| NVLS selection | nChannels>=2, efficiency factor per GPU arch | nRanks>=8 && NVLS enabled && size>=2MB |
| Chunk calc | chunkSize = nBytes / (nChannels * nRanks) | chunkSize = data_size / nRanks / ringchannels.size() |
| nChunks | Computed from chunkSize and loopSize | 2*(nRanks-1) for Ring; 64 for Tree |

**Known issue**: `genAllReduceFlowModels()` L714: `case NCCL_ALGO_TREE:` falls through to `case NCCL_ALGO_RING:`, calling `genAllReduceRingFlowModels()` instead of `genAllReduceTreeFlowModels()`.

### 2. AllGather

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Algorithms | Ring, PAT, NVLS, CollNet_DIRECT | Ring, PAT |
| Tree support | **Not available** (tuning.cc L295-296 excludes Tree) | Not available |
| PAT condition | nNodes==nRanks && PAT_ENABLE && SM60+ | nNodes>1 && nRanks==nNodes && size>=512KB |
| PAT flow | Binomial tree (parallel) | Delegates to Ring with PAT algorithm tag |
| Chunk calc | chunkSize = nBytes / (nChannels * nRanks) | chunkSize = data_size / nRanks / ringchannels.size() |
| nChunks | nRanks-1 | nRanks-1 |

### 3. ReduceScatter

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Algorithms | Ring, PAT, NVLS, CollNet_DIRECT | Ring, PAT |
| Implementation | Same structure as AllGather (reverse) | `genReduceScatterFlowModels()` L442-708 |
| PAT support | Same as AllGather | Same as AllGather |

### 4. AllToAll

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Algorithms | Ring | Ring |
| Implementation | P2P-based all-to-all | `genAlltoAllFlowModels()` L391-439 |
| Chunk calc | size/nRanks per pair | data_size/nRanks per pair |
| Channels | Single channel | Single channel |

### 5. Broadcast

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Algorithms | **Ring only** (tuning.cc L294 excludes non-Ring) | **Ring** (matching NCCL) |
| Implementation | Ring-based: root sends around ring | `genBroadcastFlowModels()` L449-495 |
| Parser support | Native | `--op Broadcast` in standalone |
| nSteps | nRanks-1 (full ring traversal from root) | nRanks-1 (8 channels x (nRanks-1) steps) |
| Root selection | Configurable via API | Fixed root=0 (standalone has no --root param) |
| Protocol | Cost-model based | Auto-selected by message size (same as Ring) |

**NCCL source evidence**: tuning.cc L294 explicitly excludes Tree/NVLS/CollNet for Broadcast:
```cpp
if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
```

**SimCCL limitation**: root is always rank 0. NCCL allows arbitrary root via API parameter.

---

## Protocol Selection Comparison

| Aspect | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| Mechanism | Cost model per (algo, proto) | Threshold-based on message size |
| LL128 enable | Complex conditions (GPU arch, connection type) | size 4MB-16MB → LL128 |
| NVLS protocol | Simple only | size<=1MB → LL, else Simple |
| Ring protocol | Cost-based selection | <=4MB → LL, 4-16MB → LL128, >16MB → Simple |
| Override | NCCL_PROTO env variable | SIMAI_FORCE_PROTO env variable |

---

## Missing Features in SimCCL

| Feature | NCCL 2.30 | SimCCL Status |
|---|---|---|
| Broadcast root selection | Configurable root rank | Fixed root=0 |
| Tree auto-selection | Cost model comparison | Code exists but not connected |
| CollNet Direct/Chain | Specialized transport | Not implemented |
| Dynamic nChannels | Graph search determines channels | Fixed from ring topology |
| treeCorrectionFactor | Size-dependent BW adjustment | Not implemented |
| netOverhead | CPU-vendor-specific overhead | Not modeled |
| PAT internal topology | Binomial tree steps | Ring approximation with PAT tag |

---

## Code Structure Map

```
NCCL 2.30:
  tuning.cc:ncclTopoTuneModel() → bandwidths[coll][algo][proto]
  tuning.cc:ncclTopoGetAlgoTime() → time = lat + bytes/bw
  enqueue.cc → selects min-time (algo, proto)

SimCCL v2.30:
  MockNcclGroup::get_algo_proto_info() → threshold-based selection
  MockNcclGroup::genFlowModels() → dispatches to gen*FlowModels()
  MockNcclGroup::genAllReduceFlowModels() → Ring/NVLS (Tree dispatch fixed)
  MockNcclGroup::genBroadcastFlowModels() → Ring (root=0, nSteps=nRanks-1)
```

---

> Last edited: 2026-06-25
