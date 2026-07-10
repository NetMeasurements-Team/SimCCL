# 添加新 Mock 集合操作指南

> [English Version](../../development/adding-new-collective.md)

本文档记录向 SimCCL 添加新集合操作的完整工作流程，以 Broadcast 为参考实现范例。

## 前置条件

- NCCL 2.30 源码（用于算法选择分析）
- SimCCL standalone 可通过 `bash build.sh v2.30` 编译
- 理解 FlowModel CSV 格式

## 分步工作流

### 步骤 1：NCCL 源码分析

在 NCCL 调优逻辑中找到目标集合操作：
- `nccl-2.30/src/graph/tuning.cc` — 算法选择和代价模型
- `nccl-2.30/src/collectives/<collective>.cc` — 实现细节

**示例（Broadcast）**：tuning.cc L294 表明 Broadcast 仅使用 Ring：
```cpp
if ((coll == ncclFuncBroadcast || coll == ncclFuncReduce) && a != NCCL_ALGO_RING) continue;
```

### 步骤 2：ComType 枚举

在两个枚举位置添加新操作：

1. **astra-sim Common.hh**：`astra-sim-alibabacloud/astra-sim/system/Common.hh`
```cpp
enum class ComType {
  None, Reduce_Scatter, All_Gather, All_Reduce,
  All_to_All, All_Reduce_All_to_All, All_Reduce_NVLS,
  Broadcast  // <-- 在此添加
};
```

2. **MockNcclChannel.h**：`mock/v2.30/MockNcclChannel.h`（如有独立枚举）

### 步骤 3：算法选择

在 `MockNcclGroup.cc` 的 `get_algo_proto_info()` 中添加分支：
```cpp
case AstraSim::ComType::Broadcast:
  info->algorithm = NCCL_ALGO_RING;  // NCCL 对 Broadcast 仅使用 Ring
  break;
```

### 步骤 4：流生成函数

在 `MockNcclGroup.cc` 中实现 `genBroadcastFlowModels()`：
- 参考已有函数模式（如 `genAllGatherFlowModels()`）
- 关键公式：nSteps = nRanks - 1（从 root 完整环遍历）
- 每个 channel 独立生成 flows
- 根据 src/dst 节点设置 `conn_type`（同节点 NVLINK，跨节点 NET）

### 步骤 5：分派入口

在 `genFlowModels()` 的 switch 中添加 case：
```cpp
case AstraSim::ComType::Broadcast:
  return genBroadcastFlowModels(type, rank, data_size);
```

### 步骤 6：头文件声明

在 `MockNcclGroup.h` 中添加：
```cpp
std::map<int,std::shared_ptr<FlowModels>> genBroadcastFlowModels(GroupType type, int rank, uint64_t data_size);
```

### 步骤 7：Standalone Parser

在 `src/main.cc` 的 `parseOp()` 中添加：
```cpp
if (s == "Broadcast") return AstraSim::ComType::Broadcast;
```
并更新 usage 字符串。

### 步骤 8：编译验证

```bash
cd SimCCL/src && bash build.sh v2.30
# 必须 0 错误编译通过
```

### 步骤 9：冒烟测试

```bash
./build/simccl-standalone --op Broadcast --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 --gpu_type H20
# 验证：CSV 生成且行数 > 1
wc -l ncclFlowModel_detailed_flows.csv  # 预期：57（header + 56 flows）
```

### 步骤 10：真机校准

运行对应的 nccl-tests 二进制：
```bash
docker exec <container> bash -c "
  cd /path/to/nccl-and-nccl-test &&
  LD_LIBRARY_PATH=nccl-2.30/build/lib \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,TUNING \
  nccl-tests/build/broadcast_perf -b 512K -e 256M -f 2 -g 8 -n 5
"
```

记录：op 名称、大小、busbw、time(us)、算法、协议、NCCL 版本、GPU 型号。

### 步骤 11：文档更新

更新以下文件（EN + CN）：
- `docs/design/nccl-comparison.md` — 算法选择表 + 逐操作对比
- `docs/README.md` — 如需添加新导航条目

## 需文档记录的限制

- root 参数：如 standalone 不支持 `--root`，注明"固定 root=0"
- 协议：如 NCCL 限制为特定协议，在对比表中注明
- 拓扑约束：如仅 Ring/特定算法有效

## Commit 前检查清单

- [ ] `bash build.sh v2.30` 通过
- [ ] 冒烟测试生成 CSV 且行数 > 1
- [ ] `git diff mock/v2.20/` 为空（v2.20 未修改）
- [ ] 真机数据已记录
- [ ] EN 和 CN 文档均已更新
- [ ] 文档/脚本中无硬编码路径
