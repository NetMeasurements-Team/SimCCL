# 安装指南

> [English Version](../../getting_started/installation.md)

## 前置条件

| 要求 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Linux (Ubuntu 20.04+) | 在 Ubuntu 22.04 上测试通过 |
| C++ 编译器 | g++ 9+ 或 clang 10+ | 需要 C++17 支持 |
| CMake | 3.14+ | 用于 standalone 构建 |
| Python3 | 3.8+（可选） | 仅画图脚本需要 |
| matplotlib | latest（可选） | 仅 `plot_results.py` 需要 |
| GPU/CUDA | **不需要** | Standalone 仅 CPU 构建 |

## 构建 Standalone 二进制

```bash
cd SimCCL/src

# 默认：v2.30 mock
bash build.sh v2.30

# 或 v2.20 mock
bash build.sh v2.20

# 手动 CMake
mkdir -p build && cd build
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

输出：`build/simccl-standalone`（约 300KB，无 ns3 依赖）

## 构建完整 SimAI（含 ns3）

```bash
cd SimAI/astra-sim-alibabacloud
bash build/astra_ns3/build.sh
```

此命令构建包含 ns3 网络仿真支持的完整模拟器。
完整构建的额外依赖：boost、protobuf（由 SimAI 构建系统提供）。

## Docker 环境

### 方案 A：在 Docker 容器内构建

```bash
# 进入 Docker 容器
docker exec -it <container_name> bash

# 容器内：构建 standalone
cd /path/to/SimAI/SimCCL/src
bash build.sh v2.30

# 运行
./build/simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8
```

### 方案 B：从宿主机通过 docker exec 运行

```bash
docker exec <container_name> bash -c "\
  cd /path/to/SimAI/SimCCL/src && \
  bash build.sh v2.30 && \
  ./build/simccl-standalone --op AllReduce --size 4194304 \
    --nRanks 8 --nNodes 1 --gpus_per_node 8"
```

将 `<container_name>` 替换为你的 Docker 容器名称。

## 验证安装

```bash
# 快速冒烟测试
./build/simccl-standalone --op AllReduce --size 4194304 \
  --nRanks 8 --nNodes 1 --gpus_per_node 8 --gpu_type H20

# 检查输出
ls -la ncclFlowModel_detailed_flows.csv
head -5 ncclFlowModel_detailed_flows.csv
```

预期：CSV 文件包含列 `collective,op,data_size,algorithm,protocol,...`
