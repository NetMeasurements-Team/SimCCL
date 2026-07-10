# SimCCL 环境变量参考

> [English Version](../../configuration/env-variables.md)

本文档覆盖 SimCCL 代码直接消费的环境变量（MockNcclGroup.cc、MockNcclLog.h）。与 SimAI ns3 集成相关的变量（AS_SEND_LAT、NCCL 原生变量等），请见 [integration-with-simai.md](../../CN/integration/integration-with-simai.md)。

## SIMAI_* 变量（新版，推荐）

| 变量 | 默认值 | 作用域 | 对输出的影响 | 说明 |
|---|---|---|---|---|
| `SIMAI_NCCL_VERSION` | `v2.20` | build.sh（编译期） | 仅编译期 | 选择 mock NCCL 版本（v2.20 或 v2.30） |
| `SIMAI_DUMP_DETAILED_FLOWS` | `1` | MockNcclGroup.cc | **是**（开关） | 控制 CSV 输出开关。`0`=关闭，`1`=启用 |
| `SIMAI_PROTO_AWARE` | `0`(v2.20) / `1`(v2.30) | MockNcclGroup.cc | **是**（通过 send_lat 表影响 ns3 延迟） | 启用基于消息大小的协议分配（LL/LL128/Simple）。影响 ns3 中的 send_lat 表查询。 |
| `SIMAI_PAT_MIN_BYTES` | `524288`(512KB) | MockNcclGroup.cc（仅 v2.30） | **是**（算法选择） | PAT 算法下限激活阈值（字节） |
| `SIMAI_PAT_MAX_BYTES` | `1048576`(1MB) | MockNcclGroup.cc（仅 v2.30） | **是**（算法选择） | PAT 算法上限阈值。超过此值使用 RING |
| `SIMAI_PAT_ENABLE` | `2` | MockNcclGroup.cc（仅 v2.30） | **是**（算法选择） | 0=禁用 PAT, 1=强制 PAT, 2=自动（双阈值） |
| `SIMAI_FORCE_PROTO` | 未设置 | MockNcclGroup.cc（仅 v2.30） | **是**（影响 send_lat） | 覆盖协议：`LL`、`LL128` 或 `Simple`。绕过自动选择。用于测试。 |

### SIMAI_PROTO_AWARE 影响分析

`SIMAI_PROTO_AWARE` 控制 FlowModel CSV 输出中的 `protocol` 字段。其调用链：

```
get_algo_proto_info() → info->protocol = LL/LL128/Simple 或 UNDEF
  → dumpDetailedFlowModels() → CSV "protocol" 列
  → entry.h: SendFlow() → send_lat_table[algo][proto] → send_lat 值
```

**当前状态**：protocol 值被 `entry.h:SendFlow()` 用于 send_lat 表查询。当 `SIMAI_PROTO_AWARE=0` 时，`protocol=-1(UNDEF)` 导致回退到默认 6000ns send_lat。当 `SIMAI_PROTO_AWARE=1` 时，使用每 (algo,proto) 表值（如 Ring+LL=7200ns for NVLINK）。

**结论**：`SIMAI_PROTO_AWARE` 通过 send_lat 表分桶机制确实影响端到端仿真延迟。它不仅仅是 CSV 标注。

## AS_* 变量（旧版，仍活跃）

| 变量 | 默认值 | 作用域 | 说明 |
|---|---|---|---|
| `AS_NVLS_ENABLE` | `0` | MockNcclGroup.cc | 为 H20/H100/H800 的 AllReduce 启用 NVLS 算法 |
| `AS_NVLS_MIN_BYTES` | `2097152`(2MB) | MockNcclGroup.cc | AllReduce 的 NVLS 激活阈值 |
| `AS_LOG_LEVEL` | - | MockNcclLog.h | 日志详细级别 |

## Fallback 行为

新 `SIMAI_*` 变量在新名称未设置时回退到旧 `AS_*` 名称：

```
SIMAI_DUMP_DETAILED_FLOWS → AS_DUMP_DETAILED_FLOWS (fallback)
SIMAI_PROTO_AWARE         → AS_PROTO_AWARE (fallback)
```

## 优先级规则

1. **`SIMAI_*`** 在两者都设置时优先于 **`AS_*`**
2. **新变量已设置** → 使用新值。**仅旧变量设置** → 使用旧值。**均未设置** → 使用默认值

## 快速使用示例

### SimCCL Standalone

```bash
cd SimCCL/src && bash build.sh v2.30

# 基本运行（默认 env）
./build/simccl-standalone --op AllGather --size 524288 --nRanks 4 --nNodes 4 --gpus_per_node 1

# 强制 PAT 算法
SIMAI_PAT_ENABLE=1 ./build/simccl-standalone --op AllGather --size 524288 --nRanks 2 --nNodes 2 --gpus_per_node 1

# 禁用 CSV 输出
SIMAI_DUMP_DETAILED_FLOWS=0 ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 4 --nNodes 4 --gpus_per_node 1

# 强制 Simple 协议
SIMAI_FORCE_PROTO=Simple ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 2 --nNodes 2 --gpus_per_node 1
```

### SimAI 完整集成

```bash
cd SimAI/
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

详细说明请见 [integration-with-simai.md](../integration/integration-with-simai.md)。

## 快速使用示例

### SimCCL Standalone

```bash
cd SimCCL/src && bash build.sh v2.30

# 基本运行（默认 env）
./build/simccl-standalone --op AllGather --size 524288 --nRanks 4 --nNodes 4 --gpus_per_node 1

# 强制 PAT 算法
SIMAI_PAT_ENABLE=1 ./build/simccl-standalone --op AllGather --size 524288 --nRanks 2 --nNodes 2 --gpus_per_node 1

# 禁用 CSV 输出
SIMAI_DUMP_DETAILED_FLOWS=0 ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 4 --nNodes 4 --gpus_per_node 1

# 强制 Simple 协议
SIMAI_FORCE_PROTO=Simple ./build/simccl-standalone --op AllGather --size 1048576 --nRanks 2 --nNodes 2 --gpus_per_node 1
```

### SimAI 完整集成

```bash
cd SimAI/
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

详细说明请见 [integration-with-simai.md](../integration/integration-with-simai.md)。
