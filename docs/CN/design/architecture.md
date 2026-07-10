# SimCCL 软件架构

> [English Version](../../design/architecture.md)


## 概述

SimCCL 是 SimAI 网络仿真器内部的 NCCL 通信翻译层。它将 NCCL 集合通信的决策（算法/协议/通道/分块）转换为 FlowModel——一种结构化的网络流描述——驱动 ns3 包级网络仿真或导出为独立的 CSV 文件供离线分析使用。

SimCCL 不实现任何实际的 GPU kernel 或网络 I/O。它翻译的是"NCCL 会做出什么决策"到"会产生什么网络流"，从而在不运行真实 NCCL 的情况下实现精确的通信建模。

## 模块结构

```
SimCCL/
├── mock/
│   ├── v2.20/          # NCCL v2.20.5 翻译语义（冻结基线）
│   │   ├── MockNcclGroup.cc/h    # 核心：组初始化、算法选择、FlowModel 生成
│   │   ├── MockNcclChannel.cc/h  # 通道拓扑：ring/tree/NVLS 构建
│   │   ├── MockNccl.h            # 常量：算法、协议、延迟表
│   │   ├── MockNcclLog.cc/h      # 日志基础设施
│   │   └── MockNcclQps.h         # 队列对模拟
│   ├── v2.30/          # NCCL v2.30.7 翻译语义（活跃开发中）
│   │   └── （相同文件，增加 PAT 算法 + 协议感知 + CSV 算法/协议列）
│   └── README.md
├── src/         # 独立可执行文件（不依赖 ns3）
│   ├── CMakeLists.txt
│   ├── main.cc
│   ├── workload_parser.cc/h
│   └── build.sh
└── docs/               # 本文档
```

## 版本选择

构建时通过 `SIMAI_NCCL_VERSION` 环境变量选择编译哪个 mock 版本：

```bash
# 默认：v2.20（行为不变）
./build.sh -c ns3

# 选择 v2.30 mock
SIMAI_NCCL_VERSION=v2.30 ./build.sh -c ns3
```

`build/astra_ns3/build.sh` 会将选定版本的文件平铺复制到 ns3 应用树中。CMakeLists.txt 通配 `SimCCL/mock/*.cc`，不受版本切换影响。

## 核心差异：v2.20 vs v2.30

| 特性 | v2.20 | v2.30 |
|---|---|---|
| PAT 算法 | 常量已定义但从不选择 | 当 `nRanks == nNodes && nNodes > 1 && size >= 512KB` 时选择 |
| 协议感知 | `info->protocol = NCCL_PROTO_UNDEF`（始终） | 根据消息大小分配协议（LL/LL128/Simple） |
| CSV 输出 | 标准列 | 新增 `algorithm`、`protocol` 两列用于溯源 |
| PROTO_AWARE 默认值 | 无（无此开关） | 默认开启（`SIMAI_PROTO_AWARE=1`） |

## 数据流

翻译流水线分为三个阶段，每个阶段有明确的输入/输出契约：

**阶段 1：算法 + 协议选择**（`get_algo_proto_info()`）
- 输入：(GroupType, rank, ComType, data_size)
- 输出：`ncclInfo*`，包含 `algorithm`（RING/TREE/NVLS/PAT）和 `protocol`（LL/LL128/Simple）
- 决策因素：GPU 类型、NVSwitch 存在性、消息大小、节点/rank 数量

**阶段 2：FlowModel 生成**（`genFlowModels()` → `genAllReduceFlowModels()` 等）
- 输入：(GroupType, rank, ComType, data_size) + 阶段 1 的算法
- 输出：`map<rank, shared_ptr<FlowModels>>`——每个 rank 的流描述
- 每条流指定：src、dest、size、channel_id、chunk_id、连接类型

**阶段 3：CSV 输出**（`dumpDetailedFlowModels()`）
- 输入：FlowModels + ncclInfo
- 输出：`ncclFlowModel_detailed_flows.csv`
- 由 `SIMAI_DUMP_DETAILED_FLOWS` 控制（默认：启用）

## 与 ns3 的集成

在完整仿真模式下，FlowModels 被 `astra-sim/system/collective/` 中的 `NcclFlowModel`（原名 NcclTreeFlowModel）消费。该类将 FlowModels 转换为 ns3 数据包发送请求，`flowTag` 携带 algorithm、protocol 和 gpus_per_node，供 `entry.h` 中的 send_lat 分桶使用。

## send_lat 分桶

ns3 入口点（`entry.h`）使用两张查找表确定每条流的发送延迟：

| 表 | 使用场景 | 数据来源 |
|---|---|---|
| `send_lat_table_nvlink[7][3]` | 同节点（节点内）流 | nccl-2.30 tuning.cc baseLat + hwLat[NVLINK] |
| `send_lat_table_net[7][3]` | 跨节点（节点间）流 | nccl-2.30 tuning.cc baseLat + hwLat[NET] |

链路类型检测使用 `flowTag.gpus_per_node`（来自 Sys 配置）判断 src 和 dst 是否在同一节点：`same_node = (src / gpus_per_node) == (dst / gpus_per_node)`。

`AS_SEND_LAT` 环境变量覆盖所有表查询（最高优先级，用于 A/B 实验）。

---
> 最后编辑时间：2026-06-23
