# SimCCL 调用栈分析

> [English Version](../../design/callstack-analysis.md)


## 概述

本文档追踪 SimCCL 中从初始化到 FlowModel 输出的关键执行路径。每个调用栈均分析了函数职责、数据流和决策点。

## 调用栈 1：完整仿真模式

```
Sys::Sys() 构造函数
  ├── workload = new Workload(...)         // 解析 workload 文件（层、通信操作）
  ├── mock_nccl_comms_init()               // 用拓扑初始化 MockNcclGroup
  │   └── GlobalGroup = new MockNcclGroup(ngpus, gpus_per_node, TP, DP, PP, EP, DP_EP, NVSwitchs, gpu_type)
  │       ├── 初始化 TP 组（ring、NVSwitch 通道）
  │       ├── 初始化 DP 组
  │       └── 初始化 PP/EP 组
  └── this->initialized = true

ns3 Simulator::Run()
  └── 事件循环驱动 Workload 执行:
      Workload::fire() → Sys::generate_collective()
        └── Sys::generate_flow_model(comm_ps, data_size, collective_type)
            └── MockNcclGroup::getFlowModels(...)
                └── （与上述相同: get_algo_proto_info → genFlowModels → dump）
            └── 创建 NcclFlowModel 实例
                └── NcclFlowModel(type, id, layer, topo, data_size, dir, policy, boost,
                                  flow_models, channels, algorithm, protocol)
                    └── insert_packets()
                        └── front_end_sim_send() → entry.h::SendFlow()
                            └── send_lat 分桶:
                                ├── 读取 flowTag.algorithm, protocol, gpus_per_node
                                ├── 确定链路类型（NVLINK vs NET）
                                ├── 查找 send_lat_table_*[algo][proto]
                                └── 如果设置了 AS_SEND_LAT 则覆盖
```

## 调用栈 3：算法选择（get_algo_proto_info）

```
MockNcclGroup::get_algo_proto_info(type, rank, op, data_size)
  ├── 构建 ncclInfoName key: "{type}_{op}_{data_size}"
  ├── 缓存检查: if nccl_infos[key] 存在 → 返回缓存
  ├── 读取 AS_NVLS_ENABLE 环境变量
  └── switch(op):
      ├── AllReduce:
      │   ├── A100/A800 → RING（始终）
      │   └── H20/H100/H800:
      │       ├── NVLS 启用 + size >= 2MB + nRanks >= 8 → NVLS
      │       └── 其他 → RING
      ├── AllGather / ReduceScatter:
      │   ├── nNodes > 1 且 nRanks == nNodes 且 size >= 512KB → PAT（仅 v2.30）
      │   └── 其他 → RING
      └── AlltoAll → RING（始终）
  └── 协议选择（v2.30, SIMAI_PROTO_AWARE=1）:
      ├── AllReduce NVLS: size <= 1MB → LL, >= 4MB → SIMPLE
      ├── AllGather/ReduceScatter RING: size <= 4MB → LL, 16MB → LL128, >= 64MB → SIMPLE
      └── 默认/禁用 → NCCL_PROTO_UNDEF (-1)
```

## 调用栈 4：独立二进制

```
main()
  ├── 解析 CLI 参数（--op/--size 或 -w workload）
  ├── 推导 group 大小（TP, DP, PP, EP）
  ├── MockNcclGroup group(nRanks, gpus_per_node, TP, DP, PP, EP, DP_EP, NVSwitchs, gpu_type)
  └── 模式 1（单操作）:
  │   └── group.getFlowModels(TP, 0, op, data_size, 0, Forward_Pass)
  │       └── （相同流程: algo → gen → dump）
  └── 模式 2（workload 文件）:
      └── for each layer:
          └── for each (fwd/wg/ig) with size > 0:
              └── group.getFlowModels(groupType, 0, comType, size, layerId, Forward_Pass)
```

---

> 最后编辑时间：2026-06-23
