# Build Options

> [中文版](../CN/configuration/build-options.md)

This document describes **SimCCL standalone** compilation options. These CMake variables control how the `simccl-standalone` binary is built.

## CMake Variables

| Variable | Values | Default | Description |
|---|---|---|---|
| `MOCK_VERSION` | `v2.20`, `v2.30` | `v2.30` | Select which NCCL mock version to compile |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug` | `Release` | Build optimization level |
| `CMAKE_CXX_FLAGS` | any | `-O2 -Wall` | Additional compiler flags |

## Example

```bash
mkdir -p build && cd build

# Release build with v2.30 (default)
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build for gdb
cmake .. -DMOCK_VERSION=v2.30 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g -O0"
make -j$(nproc)

# v2.20 build
cmake .. -DMOCK_VERSION=v2.20
make -j$(nproc)
```

## What Each Mock Version Provides

| Feature | v2.20 | v2.30 |
|---|---|---|
| Ring algorithm | Yes | Yes |
| Tree algorithm | Yes | Yes |
| NVLS algorithm | Yes | Yes |
| PAT algorithm | No | Yes |
| Protocol-aware selection | No (always UNDEF) | Yes (LL/LL128/Simple by size) |
| SIMAI_FORCE_PROTO | No | Yes |
| CSV algorithm/protocol columns | Partial | Full |
