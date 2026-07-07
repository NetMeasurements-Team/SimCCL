#!/usr/bin/env python3
"""Generate publication-quality plots from SimCCL Cartesian test results.

Reads summary CSVs from results/sim_observe/ and results/sim_override/.
Outputs figures to results/figures/.
"""
import csv
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

ALGO_NAMES = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLS_TREE", 6: "PAT", "": "N/A"}
PROTO_NAMES = {0: "LL", 1: "LL128", 2: "Simple", -1: "UNDEF", "": "N/A"}
SIZE_LABELS = {524288: "512K", 1048576: "1M", 4194304: "4M", 16777216: "16M", 67108864: "64M", 268435456: "256M"}

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available, generating text reports only")


def load_summary(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def gen_protocol_distribution(observe_data):
    """Figure 2: Protocol auto-selection distribution by size and op."""
    if not HAS_MPL:
        return
    # Group by (op, size) -> count per protocol
    dist = defaultdict(lambda: defaultdict(int))
    for r in observe_data:
        if r["mock_version"] != "v2.30" or r["status"] != "PASS":
            continue
        key = (r["op"], int(r["size"]))
        proto = r.get("auto_proto", "")
        try:
            proto_name = PROTO_NAMES.get(int(proto), str(proto))
        except (ValueError, TypeError):
            proto_name = "N/A"
        dist[key][proto_name] += 1

    if not dist:
        print("[WARN] No v2.30 observe data for protocol distribution")
        return

    ops = sorted(set(k[0] for k in dist.keys()))
    sizes = sorted(set(k[1] for k in dist.keys()))
    size_labels = [SIZE_LABELS.get(s, str(s)) for s in sizes]

    fig, axes = plt.subplots(1, len(ops), figsize=(4*len(ops), 4), sharey=True)
    if len(ops) == 1:
        axes = [axes]

    colors = {"LL": "#2196F3", "LL128": "#FF9800", "Simple": "#4CAF50", "UNDEF": "#9E9E9E", "N/A": "#BDBDBD"}

    for ax, op in zip(axes, ops):
        protos_data = defaultdict(list)
        for s in sizes:
            total = sum(dist[(op, s)].values()) or 1
            for pn in ["LL", "LL128", "Simple", "UNDEF", "N/A"]:
                protos_data[pn].append(dist[(op, s)].get(pn, 0))

        bottom = [0] * len(sizes)
        for pn in ["LL", "LL128", "Simple", "UNDEF", "N/A"]:
            vals = protos_data[pn]
            if sum(vals) > 0:
                ax.bar(range(len(sizes)), vals, bottom=bottom, label=pn, color=colors.get(pn, "#999"))
                bottom = [b + v for b, v in zip(bottom, vals)]

        ax.set_title(op, fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(size_labels, rotation=45, fontsize=8)
        ax.set_xlabel("Message Size")

    axes[0].set_ylabel("Case Count")
    axes[-1].legend(loc='upper right', fontsize=8)
    fig.suptitle("Protocol Auto-Selection Distribution (v2.30)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "protocol_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {path}")


def gen_algo_heatmap(observe_data):
    """Figure: Algorithm selection heatmap across (op, topology)."""
    if not HAS_MPL:
        return
    # Collect unique (op, gpn, nn) -> algo
    algo_map = {}
    for r in observe_data:
        if r["mock_version"] != "v2.30" or r["status"] != "PASS":
            continue
        key = (r["op"], int(r["gpus_per_node"]), int(r["nNodes"]))
        algo = r.get("auto_algo", "")
        try:
            algo_name = ALGO_NAMES.get(int(algo), str(algo))
        except (ValueError, TypeError):
            algo_name = "N/A"
        algo_map[key] = algo_name

    if not algo_map:
        return

    ops = sorted(set(k[0] for k in algo_map.keys()))
    topos = sorted(set((k[1], k[2]) for k in algo_map.keys()))
    topo_labels = [f"{gpn}gpu/{nn}node" for gpn, nn in topos]

    fig, ax = plt.subplots(figsize=(max(6, len(topos)*1.5), max(3, len(ops)*0.8)))
    data = []
    for op in ops:
        row = []
        for gpn, nn in topos:
            algo = algo_map.get((op, gpn, nn), "N/A")
            row.append(algo)
        data.append(row)

    # Text heatmap
    for i, op in enumerate(ops):
        for j, tl in enumerate(topo_labels):
            val = data[i][j]
            color = {"Ring": "#E3F2FD", "PAT": "#FFF3E0", "Tree": "#E8F5E9", "NVLS": "#FCE4EC"}.get(val, "#F5F5F5")
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=color, edgecolor='gray'))
            ax.text(j, i, val, ha='center', va='center', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(topo_labels)))
    ax.set_xticklabels(topo_labels, rotation=45, fontsize=9)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops, fontsize=10)
    ax.set_xlim(-0.5, len(topo_labels)-0.5)
    ax.set_ylim(-0.5, len(ops)-0.5)
    ax.set_title("Algorithm Selection by Op x Topology (v2.30)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "algo_selection_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {path}")


def gen_text_report(observe_data, override_data):
    """Generate text summary report."""
    report_path = os.path.join(RESULTS_DIR, "cartesian_test_report.md")
    with open(report_path, "w") as f:
        f.write("# SimCCL Cartesian Test Report\n\n")
        f.write(f"## Phase 6-A: Observe Mode\n\n")
        f.write(f"- Total cases: {len(observe_data)}\n")
        f.write(f"- PASS: {sum(1 for r in observe_data if r['status']=='PASS')}\n")
        f.write(f"- FAIL: {sum(1 for r in observe_data if r['status']=='FAIL')}\n\n")

        f.write(f"## Phase 6-B: Override Mode\n\n")
        f.write(f"- Total cases: {len(override_data)}\n")
        f.write(f"- PASS: {sum(1 for r in override_data if r['status']=='PASS')}\n")
        f.write(f"- FAIL: {sum(1 for r in override_data if r['status']=='FAIL')}\n\n")

        # Protocol distribution summary
        f.write("## Protocol Distribution (v2.30, Observe Mode)\n\n")
        f.write("| Size | LL count | LL128 count | Simple count |\n")
        f.write("|---|---|---|---|\n")
        sizes = sorted(set(int(r["size"]) for r in observe_data if r["mock_version"] == "v2.30"))
        for s in sizes:
            protos = defaultdict(int)
            for r in observe_data:
                if r["mock_version"] == "v2.30" and int(r["size"]) == s:
                    try:
                        p = PROTO_NAMES.get(int(r.get("auto_proto", -1)), "N/A")
                    except (ValueError, TypeError):
                        p = "N/A"
                    protos[p] += 1
            sl = SIZE_LABELS.get(s, str(s))
            f.write(f"| {sl} | {protos.get('LL',0)} | {protos.get('LL128',0)} | {protos.get('Simple',0)} |\n")

        f.write(f"\n---\n> Generated: 2026-06-23\n")

    print(f"  Report: {report_path}")


if __name__ == "__main__":
    print("Generating plots and report...")

    observe_path = os.path.join(RESULTS_DIR, "sim_observe", "summary.csv")
    override_path = os.path.join(RESULTS_DIR, "sim_override", "summary.csv")

    observe_data = load_summary(observe_path) if os.path.exists(observe_path) else []
    override_data = load_summary(override_path) if os.path.exists(override_path) else []

    print(f"  Observe data: {len(observe_data)} rows")
    print(f"  Override data: {len(override_data)} rows")

    gen_protocol_distribution(observe_data)
    gen_algo_heatmap(observe_data)
    gen_text_report(observe_data, override_data)

    print("Done.")
