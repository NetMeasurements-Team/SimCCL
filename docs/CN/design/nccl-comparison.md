# NCCL 2.30 vs SimCCL v2.30 源码对照

> [English Version](../../design/nccl-comparison.md)

## 概述

本文档对 NCCL 2.30 源码与 SimCCL mock v2.30 实现进行逐操作详细对照。

**NCCL 2.30 源码**：`nccl-2.30/src/graph/tuning.cc`（算法选择、代价模型）
**SimCCL mock**：`SimCCL/mock/v2.30/MockNcclGroup.cc`（流生成、算法选择）

---

## 算法选择对比

### NCCL 2.30 代价模型（tuning.cc）

NCCL 使用复杂代价模型，为每个 (算法, 协议) 组合计算 `time = latency + nBytes / (1000 * bandwidth)`，选择最小值。

关键因素：
- `baseLatencies[algo][proto]` — 固定的算法启动开销
- `hwLatencies[hw_type][algo][proto]` — 每跳硬件延迟（NVLINK/PCI/NET）
- `bandwidth` — 由拓扑图搜索得出（nChannels * 每通道带宽）
- 各种修正因子（treeCorrectionFactor、PAT 0.75x 等）

### SimCCL 算法选择（get_algo_proto_info, L2130-2184）

SimCCL 使用基于阈值的简化选择：

| 操作 | 条件 | 选择的算法 |
|---|---|---|
| AllReduce (TP, H100/H800/H20) | nRanks>=8 且 NVLS 启用 且 size>=2MB | NVLS |
| AllReduce（其他） | 默认 | Ring |
| AllGather/ReduceScatter | nNodes>1 且 nRanks==nNodes 且 size>=512KB | PAT |
| AllGather/ReduceScatter | 默认 | Ring |
| AllToAll | 始终 | Ring |
| Broadcast | 始终 | Ring（仅 v2.30，root=0） |

---

## 逐操作对比

### 1. AllReduce

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 支持算法 | Ring, Tree, NVLS, NVLS_TREE, CollNet | Ring, NVLS（Tree 代码存在但未自动选择） |
| Tree 选择 | 代价模型比较 Tree vs Ring 延迟 | **未自动选择**（L714: TREE 分派已修复但 H20 不触发） |
| Tree 流生成 | 二叉树 up+down 阶段 | `genAllReduceTreeFlowModels()` 存在（L991），gp_idx bug 已修复 |
| NVLS 选择 | nChannels>=2，按 GPU 架构的效率因子 | nRanks>=8 且 NVLS 启用 且 size>=2MB |
| Chunk 计算 | chunkSize = nBytes / (nChannels * nRanks) | chunkSize = data_size / nRanks / ringchannels.size() |
| nChunks | 由 chunkSize 和 loopSize 计算 | Ring: 2*(nRanks-1)；Tree: 64 |

**已知问题**：`genAllReduceFlowModels()` L714 的 `case NCCL_ALGO_TREE:` 分派已修复（不再 fall-through 到 RING），但 H20x8 真机实验确认 NCCL 从不选择 Tree（始终 RING 或 NVLS）。

### 2. AllGather

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 支持算法 | Ring, PAT, NVLS, CollNet_DIRECT | Ring, PAT |
| Tree 支持 | **不可用**（tuning.cc L295-296 排除 Tree） | 不可用 |
| PAT 条件 | nNodes==nRanks 且 PAT_ENABLE 且 SM60+ | nNodes>1 且 nRanks==nNodes 且 size>=512KB |
| PAT 流生成 | 二项树（并行） | 委托给 Ring 并标记 PAT 算法 |
| Chunk 计算 | chunkSize = nBytes / (nChannels * nRanks) | chunkSize = data_size / nRanks / ringchannels.size() |
| nChunks | nRanks-1 | nRanks-1 |

### 3. ReduceScatter

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 支持算法 | Ring, PAT, NVLS, CollNet_DIRECT | Ring, PAT |
| 实现 | 与 AllGather 结构相同（反向） | `genReduceScatterFlowModels()` L442-708 |
| PAT 支持 | 与 AllGather 相同 | 与 AllGather 相同 |

### 4. AllToAll

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 算法 | Ring | Ring |
| 实现 | P2P 全对全 | `genAlltoAllFlowModels()` L391-439 |
| Chunk 计算 | size/nRanks per pair | data_size/nRanks per pair |
| 通道数 | 单通道 | 单通道 |

### 5. Broadcast

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 算法 | **仅 Ring**（tuning.cc L294 排除非 Ring） | **Ring**（匹配 NCCL） |
| 实现 | Ring-based：root 沿环发送 | `genBroadcastFlowModels()` L449-495 |
| Parser 支持 | 原生 | standalone `--op Broadcast` |
| nSteps | nRanks-1（从 root 完整环遍历） | nRanks-1（8 channels × (nRanks-1) steps） |
| Root 选择 | 通过 API 可配置 | 固定 root=0（standalone 无 --root 参数） |
| 协议 | 基于代价模型 | 按消息大小自动选择（与 Ring 相同） |

**NCCL 源码证据**：tuning.cc L294 明确排除 Broadcast 的 Tree/NVLS/CollNet：
```cpp
if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
```

**SimCCL 限制**：root 始终为 rank 0。NCCL 允许通过 API 参数指定任意 root。

---

## 协议选择对比

| 方面 | NCCL 2.30 | SimCCL v2.30 |
|---|---|---|
| 机制 | 按 (algo, proto) 代价模型 | 按消息大小阈值 |
| LL128 启用 | 复杂条件（GPU 架构、连接类型） | size 4MB-16MB → LL128 |
| NVLS 协议 | 仅 Simple | size<=1MB → LL，否则 Simple |
| Ring 协议 | 基于代价选择 | <=4MB → LL，4-16MB → LL128，>16MB → Simple |
| 覆盖 | NCCL_PROTO 环境变量 | SIMAI_FORCE_PROTO 环境变量 |

---

## SimCCL 缺失/受限功能

| 功能 | NCCL 2.30 | SimCCL 状态 |
|---|---|---|
| Broadcast root 选择 | 可配置 root rank | 固定 root=0 |
| Tree 自动选择 | 代价模型比较 | 代码存在但未连接（H20 不触发） |
| CollNet Direct/Chain | 专用传输 | 未实现 |
| 动态 nChannels | 图搜索确定通道数 | 固定使用 ring 拓扑通道 |
| treeCorrectionFactor | 按大小调整带宽 | 未实现 |
| netOverhead | CPU 厂商特定开销 | 未建模 |
| PAT 内部拓扑 | 二项树步骤 | Ring 近似+PAT 标签 |

---

## 代码结构映射

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

> 最后编辑：2026-06-25
