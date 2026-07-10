# Installation

> [中文版](../CN/getting_started/installation.md)

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) | Tested on Ubuntu 22.04 |
| C++ compiler | g++ 9+ or clang 10+ | C++17 support required |
| CMake | 3.14+ | For standalone build |
| Python3 | 3.8+ (optional) | Only needed for plotting scripts |
| matplotlib | latest (optional) | Only needed for `plot_results.py` |
| GPU/CUDA | **Not required** | Standalone is CPU-only |

## Build Standalone Binary

```bash
cd SimCCL/src

# Default: v2.30 mock
bash build.sh v2.30

# Or v2.20 mock
bash build.sh v2.20

# Manual CMake
mkdir -p build && cd build
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Output: `build/simccl-standalone` (~300KB, no ns3 dependency)

> **Required layout**: the standalone build includes `astra-sim/system/Common.hh` and its CMake references `../../astra-sim-alibabacloud`. SimCCL must therefore be checked out inside a SimAI tree, i.e. as `SimAI/SimCCL/` next to `SimAI/astra-sim-alibabacloud/`. A bare `git clone` of SimCCL alone cannot build the standalone binary.

## Build Full SimAI (with ns3)

```bash
cd SimAI/astra-sim-alibabacloud
bash build/astra_ns3/build.sh
```

This builds the complete simulator with ns3 network simulation support.
Additional dependencies for full build: boost, protobuf (provided by the SimAI build system).

## Docker Environment

### Option A: Build inside Docker container

```bash
# Enter Docker container
docker exec -it <container_name> bash

# Inside container: build standalone
cd /path/to/SimAI/SimCCL/src
bash build.sh v2.30

# Run
./build/simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8
```

### Option B: Run from host via docker exec

```bash
docker exec <container_name> bash -c "\
  cd /path/to/SimAI/SimCCL/src && \
  bash build.sh v2.30 && \
  ./build/simccl-standalone --op AllReduce --size 4194304 \
    --nRanks 8 --nNodes 1 --gpus_per_node 8"
```

Replace `<container_name>` with your Docker container name.

## Verify Installation

```bash
# Quick smoke test
./build/simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 --gpu_type H20

# Check output
ls -la ncclFlowModel_detailed_flows.csv
head -5 ncclFlowModel_detailed_flows.csv
```

Expected: CSV file with columns `collective,op,data_size,algorithm,protocol,...`
