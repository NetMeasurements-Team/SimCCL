#!/usr/bin/env python3
"""Generate valid test case CSVs for SimCCL Cartesian-product testing.

Outputs:
  configs/sim_cases_observe.csv  - Phase6-A: auto protocol selection
  configs/sim_cases_override.csv - Phase6-B: forced protocol
  configs/real_cases.csv         - Phase6-C: real machine nccl-tests
"""
import csv
import os
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "configs")
os.makedirs(CONFIGS_DIR, exist_ok=True)

# Dimensions
OPS = ["AllReduce", "AllGather", "ReduceScatter", "AlltoAll", "Broadcast"]
SIZES = [524288, 1048576, 4194304, 16777216, 67108864, 268435456]
MOCK_VERSIONS = ["v2.30"]
GPUS_PER_NODE = [1, 2, 4, 8]
NODES = [1, 2]
PROTOCOLS = ["LL", "LL128", "Simple"]
GPU_TYPE = "H20"


def is_valid(op, size, mock_ver, gpn, nn):
    """Check if a test case is valid per topology constraints."""
    nranks = nn * gpn
    if nranks <= 1:
        return False  # Single rank cannot do collective communication
    # Broadcast only supported in v2.30 (safety check)
    if op == "Broadcast" and mock_ver != "v2.30":
        return False
    return True


def gen_observe_cases():
    """Phase6-A: observe auto-selected protocol."""
    cases = []
    case_id = 0
    for op, size, ver, gpn, nn in itertools.product(OPS, SIZES, MOCK_VERSIONS, GPUS_PER_NODE, NODES):
        if not is_valid(op, size, ver, gpn, nn):
            continue
        nranks = nn * gpn
        case_id += 1
        cases.append({
            "case_id": case_id, "op": op, "size": size, "mock_version": ver,
            "gpus_per_node": gpn, "nNodes": nn, "nRanks": nranks,
            "gpu_type": GPU_TYPE, "force_proto": ""
        })
    return cases


def gen_override_cases():
    """Phase6-B: forced protocol enumeration."""
    cases = []
    case_id = 0
    for op, size, ver, gpn, nn, proto in itertools.product(OPS, SIZES, MOCK_VERSIONS, GPUS_PER_NODE, NODES, PROTOCOLS):
        if not is_valid(op, size, ver, gpn, nn):
            continue
        nranks = nn * gpn
        case_id += 1
        cases.append({
            "case_id": case_id, "op": op, "size": size, "mock_version": ver,
            "gpus_per_node": gpn, "nNodes": nn, "nRanks": nranks,
            "gpu_type": GPU_TYPE, "force_proto": proto
        })
    return cases


def gen_real_cases():
    """Phase6-C: real machine tests (limited to actual hardware)."""
    # Hardware: 2 nodes x 8 GPU H20
    configs = [
        (8, 1, 8),    # 1 node, 8 GPU
        (16, 2, 8),   # 2 nodes, 8 GPU each
        (2, 2, 1),    # 2 nodes, 1 GPU each (PAT trigger)
    ]
    real_ops = ["all_gather", "all_reduce", "reduce_scatter", "alltoall", "broadcast"]
    real_sizes = ["512K", "1M", "4M", "16M", "64M", "256M"]

    cases = []
    case_id = 0
    for nr, nn, gpn in configs:
        for op in real_ops:
            for size in real_sizes:
                case_id += 1
                cases.append({
                    "case_id": case_id, "op": op, "size": size,
                    "nRanks": nr, "nNodes": nn, "gpus_per_node": gpn,
                    "nccl_debug": "INFO",
                    "nccl_debug_subsys": "INIT,COLL,TUNING"
                })
    return cases


def write_csv(path, cases, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(cases)
    print(f"  {path}: {len(cases)} cases")


if __name__ == "__main__":
    print("Generating test case CSVs...")

    observe = gen_observe_cases()
    write_csv(os.path.join(CONFIGS_DIR, "sim_cases_observe.csv"), observe,
              ["case_id", "op", "size", "mock_version", "gpus_per_node", "nNodes", "nRanks", "gpu_type", "force_proto"])

    override = gen_override_cases()
    write_csv(os.path.join(CONFIGS_DIR, "sim_cases_override.csv"), override,
              ["case_id", "op", "size", "mock_version", "gpus_per_node", "nNodes", "nRanks", "gpu_type", "force_proto"])

    real = gen_real_cases()
    write_csv(os.path.join(CONFIGS_DIR, "real_cases.csv"), real,
              ["case_id", "op", "size", "nRanks", "nNodes", "gpus_per_node", "nccl_debug", "nccl_debug_subsys"])

    print(f"\nTotal: observe={len(observe)}, override={len(override)}, real={len(real)}")
