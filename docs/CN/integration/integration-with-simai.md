# SimCCL 与 SimAI 主库集成说明

> [English Version](../../integration/integration-with-simai.md)

## 适用场景

本文档描述 SimCCL 作为 **SimAI ns3 全栈仿真** 一部分使用时的相关环境变量和配置。这些变量要么：
- 被 astra-sim ns3 前端层消费（不是 SimCCL 代码本身），要么
- 仅在运行完整 SimAI 管线时才有意义

SimCCL standalone 环境变量请见 [env-variables.md](../../CN/configuration/env-variables.md)。

---

## 集成环境变量

### SIMAI_DUMP_DETAILED_FLOWS（集成场景用法）

| 属性 | 值 |
|---|---|
| 代码位置 | `SimCCL/mock/v2.30/MockNcclGroup.cc` L327-332 |
| 默认值 | `1`（启用） |
| standalone 行为 | 控制 CSV 输出（与集成相同） |
| 集成行为 | 运行完整 SimAI ns3 仿真时，可能需要关闭 CSV 生成以提升性能：设置 `SIMAI_DUMP_DETAILED_FLOWS=0` |

**注意**：此变量在 [env-variables.md](../../CN/configuration/env-variables.md) 中也有简短的 standalone 角度描述。本节提供完整的集成上下文。

#### 快速命令

```bash
# 前置：编译 SimAI-Simulation
cd SimAI/
./scripts/build.sh -c ns3

# 生成拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# 运行（默认 SIMAI_DUMP_DETAILED_FLOWS=1，生成 CSV）
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# 禁用 detailed_flows CSV（提升大规模仿真性能）
SIMAI_DUMP_DETAILED_FLOWS=0 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# AS_SEND_LAT 覆盖实验（单位：纳秒）
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

详细参数说明请见 [快速入门指南](../../docs/CN/getting_started/quickstart.md)。

### AS_SEND_LAT

| 属性 | 值 |
|---|---|
| 代码位置 | `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h` L168-177 |
| 默认值 | 未设置（使用 send_latency_table） |
| 作用域 | 仅 ns3 网络前端 — SimCCL 不消费 |
| 优先级 | **最高** — 覆盖所有基于表的 send_lat 查询 |

设置后，所有流使用此单一值（纳秒）作为其发送延迟，无论算法、协议或链路类型。

**使用场景**：
- A/B 实验：用不同 send_lat 值对比仿真结果
- 快速校准：设置已知值与真机测量对比
- 调试：消除 send_lat 变化以隔离其他因素

**与 send_latency_table 的关系**：`entry.h` 中的表提供按 (算法, 协议, 链路类型) 的 send_lat 值。`AS_SEND_LAT` 设置后覆盖所有表查询。表机制详细分析见 `SimAI/docs/CN/configuration/send-lat-analysis.md`。

### AS_DUMP_DETAILED_FLOWS（旧名称兼容）

`SIMAI_DUMP_DETAILED_FLOWS` 的旧名称。为外部脚本向后兼容而保留。

---

## FlowModel CSV 数据流

```
SimCCL (MockNcclGroup.cc)
  → 生成 ncclFlowModel_detailed_flows.csv
  → CSV 包含: algorithm, protocol, flow_size, src, dest, conn_type 等

ns3 前端 (entry.h: SendFlow())
  → 从 CSV 读取 flowTag（algorithm, protocol, gpus_per_node）
  → 根据链路类型查询 send_lat_table[algo][proto]
  → 如设置 AS_SEND_LAT 则覆盖
  → 用计算出的延迟启动 ns3 应用连接
```

---

## NCCL 原生变量（真机测试用）

这些是标准 NCCL 环境变量，仅在真机跨节点测试时使用。SimCCL mock **不消费**这些变量——它们配置真实 NCCL 库。

| 变量 | 值 | 说明 |
|---|---|---|
| `NCCL_DEBUG` | `INFO` / `TRACE` | 打印 NCCL 算法/协议选择信息。TRACE 用于 PAT 步骤级细节。 |
| `NCCL_DEBUG_SUBSYS` | `COLL,TUNING,GRAPH` | 过滤 NCCL 调试输出到特定子系统 |
| `NCCL_IB_DISABLE` | `0` | 启用 InfiniBand（默认） |
| `NCCL_ALGO` | `PAT` / `RING` / `TREE` | 强制指定算法（用于测试） |
| `CUDA_VISIBLE_DEVICES` | `0` | 限制每节点 1 GPU（用于 PAT 测试） |

跨节点测试部署详情见 [cross-node-test.md](../../CN/benchmarking/cross-node-test.md)。

---

## 端到端运行指南

完整的编译、拓扑生成和仿真运行指南，请参见 SimAI 主仓文档：
- [安装指南](../../docs/CN/getting_started/installation.md)
- [快速入门](../../docs/CN/getting_started/quickstart.md)
- [环境变量参考](../../docs/CN/configuration/env-variables.md)
- [构建选项](../../docs/CN/configuration/build-options.md)

---

## 与 standalone 模式的区别

| 方面 | Standalone | 完整 SimAI 集成 |
|---|---|---|
| 二进制 | `simccl-standalone` | `astra_ns3`（完整仿真器） |
| 网络仿真 | 无 | ns3 完整仿真 |
| 输出 | 仅 CSV | CSV + EndToEnd.csv + 时序 |
| `AS_SEND_LAT` | 不适用 | 覆盖 send_lat |
| NCCL 原生变量 | 不适用 | 用于真机验证 |
| `SIMAI_DUMP_DETAILED_FLOWS` | 控制 CSV（默认开启） | 控制 CSV（可能为性能关闭） |

---

> 最后编辑：2026-06-25
