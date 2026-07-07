# SimCCL Call Stack Analysis

> [中文版](../CN/design/callstack-analysis.md)

## Overview

This document traces the key execution paths through SimCCL, from initialization to FlowModel output. Each call stack is analyzed with function responsibilities, data flow, and decision points.

## Call Stack 1: Full Simulation Mode

```
Sys::Sys() constructor
  ├── workload = new Workload(...)         // Parse workload file (layers, comm ops)
  ├── mock_nccl_comms_init()               // Initialize MockNcclGroup with topology
  │   └── GlobalGroup = new MockNcclGroup(ngpus, gpus_per_node, TP, DP, PP, EP, DP_EP, NVSwitchs, gpu_type)
  │       ├── Init TP groups (rings, NVSwitch channels)
  │       ├── Init DP groups
  │       └── Init PP/EP groups
  └── this->initialized = true

ns3 Simulator::Run()
  └── Event loop drives Workload execution:
      Workload::fire() → Sys::generate_collective()
        └── Sys::generate_flow_model(comm_ps, data_size, collective_type)
            └── MockNcclGroup::getFlowModels(...)
                └── (same as above: get_algo_proto_info → genFlowModels → dump)
            └── Creates NcclFlowModel instance
                └── NcclFlowModel(type, id, layer, topo, data_size, dir, policy, boost,
                                  flow_models, channels, algorithm, protocol)
                    └── insert_packets()
                        └── front_end_sim_send() → entry.h::SendFlow()
                            └── send_lat bucketing:
                                ├── Read flowTag.algorithm, protocol, gpus_per_node
                                ├── Determine link type (NVLINK vs NET)
                                ├── Look up send_lat_table_*[algo][proto]
                                └── Override with AS_SEND_LAT if set
```

## Call Stack 3: Algorithm Selection (get_algo_proto_info)

```
MockNcclGroup::get_algo_proto_info(type, rank, op, data_size)
  ├── Build ncclInfoName key: "{type}_{op}_{data_size}"
  ├── Cache check: if nccl_infos[key] exists → return cached
  ├── Read AS_NVLS_ENABLE env
  └── switch(op):
      ├── AllReduce:
      │   ├── A100/A800 → RING (always)
      │   └── H20/H100/H800:
      │       ├── NVLS enabled + size >= 2MB + nRanks >= 8 → NVLS
      │       └── else → RING
      ├── AllGather / ReduceScatter:
      │   ├── nNodes > 1 AND nRanks == nNodes AND size >= 512KB → PAT (v2.30 only)
      │   └── else → RING
      └── AlltoAll → RING (always)
  └── Protocol selection (v2.30, SIMAI_PROTO_AWARE=1):
      ├── AllReduce NVLS: size <= 1MB → LL, >= 4MB → SIMPLE
      ├── AllGather/ReduceScatter RING: size <= 4MB → LL, 16MB → LL128, >= 64MB → SIMPLE
      └── Default/disabled → NCCL_PROTO_UNDEF (-1)
```

## Call Stack 4: Standalone Binary

```
main()
  ├── Parse CLI arguments (--op/--size or -w workload)
  ├── Derive group sizes (TP, DP, PP, EP)
  ├── MockNcclGroup group(nRanks, gpus_per_node, TP, DP, PP, EP, DP_EP, NVSwitchs, gpu_type)
  └── Mode 1 (single op):
  │   └── group.getFlowModels(TP, 0, op, data_size, 0, Forward_Pass)
  │       └── (same pipeline: algo → gen → dump)
  └── Mode 2 (workload file):
      └── for each layer:
          └── for each (fwd/wg/ig) with size > 0:
              └── group.getFlowModels(groupType, 0, comType, size, layerId, Forward_Pass)
```
