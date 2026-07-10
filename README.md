# SimCCL

SimCCL is the collective communication translation layer for the [SimAI](https://github.com/aliyun/SimAI) network simulator. It converts NCCL collective operation decisions (algorithm/protocol/channel/chunk) into FlowModels — structured point-to-point flow descriptions — that drive ns3 packet-level simulation or can be exported as CSV for offline analysis.

SimCCL does NOT implement GPU kernels or network I/O. It translates "what NCCL would decide" into "what flows would result".

## Supported Operations

| Collective | Algorithms | Notes |
|---|---|---|
| AllReduce | Ring, NVLS | NVLS for single-node >=8 GPU |
| AllGather | Ring, PAT | PAT for multi-node 1-GPU/node |
| ReduceScatter | Ring, PAT | PAT for multi-node 1-GPU/node |
| AlltoAll | Ring | — |
| Broadcast | Ring | — |

> Note: Tree algorithm constants exist but auto-selection is not triggered on tested hardware (H20). CollNetDirect/CollNetChain are not implemented (require SHARP hardware).

## Quick Start (Standalone Mode)

### Build

```bash
cd SimCCL/src
bash build.sh v2.30          # default v2.30, or pass v2.20
```

Output: `build/simccl-standalone` (~300KB, no ns3 dependency, CPU-only)

> Note: the standalone build includes `astra-sim/system/Common.hh` and its CMake references `../../astra-sim-alibabacloud`, so SimCCL must sit inside a SimAI checkout (`SimAI/SimCCL/` next to `SimAI/astra-sim-alibabacloud/`). See [Installation](./docs/getting_started/installation.md) for the required layout.

### Run

```bash
# Single collective operation
./build/simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 --gpu_type H20

# Workload file mode
./build/simccl-standalone -w ../../example/microAllReduce.txt \
  --nRanks 16 --nNodes 2 --gpus_per_node 8 --gpu_type H20
```

Output: `ncclFlowModel_detailed_flows.csv`

### Full Test Suite

```bash
cd SimCCL
bash scripts/run_all.sh
```

## Integration with SimAI

When compiled as part of the full SimAI simulator, SimCCL provides FlowModels to the ns3 network simulation backend:

```bash
cd SimAI/
./scripts/build.sh -c ns3
```

See [Integration Guide](./docs/integration/integration-with-simai.md) for details.

## Documentation

See [docs/README.md](./docs/README.md) for full documentation index.

## License

MIT — See [LICENSE](./LICENSE)
