# SimCCL 独立运行指南

> [English Version](../../getting_started/quickstart.md)


## 什么是独立模式？

SimCCL standalone 是一个自包含二进制程序，无需链接或运行 ns3 网络仿真即可生成 FlowModel CSV 输出。它只编译 MockNccl 翻译层文件和一个最小的 workload 解析器，产出与完整仿真器相同的 `ncclFlowModel_detailed_flows.csv`。

该模式的存在是为了满足 GitHub PR 要求：SimCCL 应当可以独立运行和测试，无需依赖 ns3。

## 编译

```bash
cd SimCCL/src

# 编译 v2.30 mock（默认）
bash build.sh v2.30

# 编译 v2.20 mock
bash build.sh v2.20

# 或者手动 CMake
mkdir -p build && cd build
cmake .. -DMOCK_VERSION=v2.30
make -j$(nproc)
```

输出二进制为 `build/simccl-standalone`（~300KB，无 ns3 依赖）。

## 运行

### 模式 1：单个集合通信操作

通过命令行参数指定一个 (op, size) 组合：

```bash
./simccl-standalone \
  --op AllReduce \
  --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 \
  --gpu_type H20
```

必需参数：
- `--op`：AllReduce、AllGather、ReduceScatter 或 AlltoAll
- `--size`：数据大小（字节）
- `--nRanks`：GPU 总 rank 数
- `--nNodes`：节点数
- `--gpus_per_node`：每节点 GPU 数

可选参数：
- `--gpu_type`：H20（默认）、H100、H800、A100、A800

### 模式 2：Workload 文件

解析 SimAI workload 文件并枚举所有层：

```bash
./simccl-standalone \
  -w example/microAllReduce.txt \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 \
  --gpu_type H20
```

### 环境变量

完整仿真器中使用的环境变量同样适用于独立模式：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AS_NVLS_ENABLE` | `0` | 为 AllReduce 启用 NVLS 算法（H20/H100 需要） |
| `SIMAI_DUMP_DETAILED_FLOWS` | `1` | 启用 CSV 输出（设 `0` 禁用） |
| `SIMAI_PROTO_AWARE` | `1`（v2.30）/ `0`（v2.20） | 启用协议感知选择 |
| `SIMAI_PAT_MIN_BYTES` | `524288` | PAT 算法下限激活阈值（字节） |
| `SIMAI_PAT_MAX_BYTES` | `1048576` | PAT 算法上限阈值（超过则使用 RING） |
| `SIMAI_PAT_ENABLE` | `2` | 0=禁用 PAT, 1=强制 PAT, 2=自动（双阈值） |

启用 NVLS 的示例：
```bash
AS_NVLS_ENABLE=1 ./simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8
```

## 输出格式

输出文件 `ncclFlowModel_detailed_flows.csv` 包含以下列：

```
collective,op,data_size,algorithm,protocol,channel_id,flow_id,src,dest,
flow_size,chunk_id,chunk_count,conn_type,parent_flow_ids,prev_flow_ids
```

- `algorithm`：0=Tree, 1=Ring, 2=CollNetDirect(*), 3=CollNetChain(*), 4=NVLS, 5=NVLS_TREE, 6=PAT
  - (*) CollNetDirect/CollNetChain：SimCCL 中存在常量定义但未实现自动选择和流生成。NCCL 中这两个算法依赖 CollNet-capable fabric/SHARP 硬件；无此硬件时 NCCL 自身也不会选择。
- `protocol`：0=LL, 1=LL128, 2=Simple, -1=UNDEF
- `conn_type`：NVLINK、NET、PCI、NVLS 等

### 列说明

| 列名 | 说明 | NCCL 对应变量 |
|---|---|---|
| `collective` | 集合操作名称（如 AllReduce） | ncclFunc* 枚举 |
| `op` | 归约操作（sum 等） | ncclRedOp_t |
| `data_size` | 总数据大小（字节） | info->sendbytes |
| `algorithm` | 算法索引（见上表） | info->algorithm (NCCL_ALGO_*) |
| `protocol` | 协议索引（见上表） | info->protocol (NCCL_PROTO_*) |
| `channel_id` | 通信通道索引 | channel->id |
| `flow_id` | 此集合操作内的唯一流标识 | SimCCL 内部 |
| `src` | 源 GPU rank | - |
| `dest` | 目标 GPU rank | - |
| `flow_size` | 此流传输的字节数 | chunkSize |
| `chunk_id` | 集合操作内的 chunk 索引 | - |
| `chunk_count` | chunk 总数 | nChunksPerLoop |
| `conn_type` | 连接类型（NVLINK/NET/PCI/NVLS） | connector->transportComm type |
| `parent_flow_ids` | 流依赖（数据依赖） | SimCCL 内部 |
| `prev_flow_ids` | 流排序（顺序依赖） | SimCCL 内部 |

## 测试

运行完整功能测试脚本以验证所有配置：

```bash
cd SimCCL
bash scripts/run_all.sh
# 结果：results/standalone_test_summary.csv
```

脚本测试 56 个用例：52 个有效拓扑组合（1n8g, 2n8g, 2n1g-PAT, 1n4g, 2n2g）+ 4 个无效输入（预期失败）。

## 已知限制

1. **NVSwitch 拓扑**：仅在 `gpus_per_node > 4` 时创建 NVSwitch 节点。更小配置下跳过 NVSwitch 相关组条目（已做边界检查）。

2. **PAT 实现**：当前 PAT FlowModel 生成（`genPATFlowModels()`）委托给 Ring 流生成并标记 PAT 算法。流拓扑镜像 Ring 直到真机 PAT 校准（通过 `NCCL_DEBUG=TRACE` 或 nsys kernel trace）提供树特定模式。详见 [pat-algorithm.md](../design/pat-algorithm.md)。注：PAT 仅对 AllGather/ReduceScatter 触发，不对 AllReduce。

3. **Group 类型映射**：在 workload 文件模式下，group 类型（TP/DP/PP/EP）必须与 nRanks/gpus_per_node 配置匹配。如果 workload 指定了 DP 通信但 DP_size=1，则不生成 flow（正确行为）。

4. **无 ns3 时序**：此模式只生成 FlowModels，不模拟网络延迟、拥塞或端到端时序。如需完整仿真，请使用 SimAI ns3 二进制。

5. **CollNetDirect/CollNetChain 未实现**：SimCCL 存在 CollNet 算法的常量和 send_lat 表项（索引 2 和 3），但自动选择逻辑、拓扑建模和流生成均未实现。NCCL 2.30 中这些算法依赖 CollNet-capable transport plugin 和 SHARP fabric 硬件（`init.cc` 中的 `collNetChainGraph->nChannels > 0` 检查）。无此硬件时 NCCL 自身也不会选择。

