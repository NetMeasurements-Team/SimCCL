# 构建选项

> [English Version](../../configuration/build-options.md)

本文档描述 **SimCCL standalone** 的编译选项。这些 CMake 变量控制 `simccl-standalone` 二进制的构建方式。

## CMake 变量

| 变量 | 取值 | 默认值 | 说明 |
|---|---|---|---|
| `MOCK_VERSION` | `v2.20`, `v2.30` | `v2.30` | 选择编译的 NCCL mock 版本 |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug` | `Release` | 构建优化级别 |
| `CMAKE_CXX_FLAGS` | 任意 | `-O2 -Wall` | 额外编译器标志 |

## 示例

```bash
mkdir -p build && cd build

# Release 构建，使用 v2.30（默认）
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug 构建，用于 gdb 调试
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g -O0"
make -j$(nproc)

# v2.20 构建
cmake .. -DMOCK_VERSION=v2.20
make -j$(nproc)
```

## 各 Mock 版本提供的功能

| 功能 | v2.20 | v2.30 |
|---|---|---|
| Ring 算法 | 是 | 是 |
| Tree 算法 | 是 | 是 |
| NVLS 算法 | 是 | 是 |
| PAT 算法 | 否 | 是 |
| 协议感知选择 | 否（始终 UNDEF） | 是（按大小选 LL/LL128/Simple） |
| SIMAI_FORCE_PROTO | 否 | 是 |
| CSV 算法/协议列 | 部分 | 完整 |
