#!/usr/bin/env python3
"""
Figure 2: Ablation Study
Grouped bars showing compute kernel time for AB-Cached-32 (no block skip),
AB-Stream-32, AB-Cached-32, and ESMM at 3 density levels.
Each variant adds one optimization over the previous.
"""

import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W2

apply_style()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmarks" / "paper_data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITIES = [12.5, 25.0, 50.0]
DENSITY_LABELS = ["12_5pct", "25_0pct", "50_0pct"]

# K29 data comes from fig1_density, K26/K27/K28 from fig2_ablation
KERNEL_DIRS = {
    27: "fig2_ablation",
    28: "fig2_ablation",
    26: "fig2_ablation",
    29: "fig1_density",
}

KERNEL_LABELS = {
    27: "AB-Cached-32\n(no block skip)",
    28: "AB-Stream-32\n(gmem patterns)",
    26: "AB-Cached-32\n(+ block skip)",
    29: "ESMM\n(+ float2 A-loads) ★",
}

KERNEL_COLORS = {
    27: COLORS["AB-Cached-32-noskip"],
    28: COLORS["AB-Stream-32"],
    26: COLORS["AB-Cached-32"],
    29: COLORS["ESMM"],
}


def extract_compute_time_ms(ncu_rep_path):
    """Extract compute kernel time (last msecond Duration) from NCU report."""
    cmd = ["sudo", "/usr/local/cuda-12.1/bin/ncu", "--import", str(ncu_rep_path),
           "--csv", "--page", "details"]
    env = {"PATH": "/usr/bin:/usr/local/bin", "LD_LIBRARY_PATH": "/usr/local/lib64"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    compute_ms = 0.0
    for line in result.stdout.split("\n"):
        if '"Duration"' not in line:
            continue
        if '"msecond"' in line:
            val = re.search(r'"msecond","([^"]+)"', line)
            if val:
                compute_ms = float(val.group(1).replace(",", ""))  # last one wins
    return compute_ms


def load_data():
    data = {}
    for kernel, subdir in KERNEL_DIRS.items():
        data[kernel] = {}
        for density, label in zip(DENSITIES, DENSITY_LABELS):
            rep = DATA_DIR / subdir / f"k{kernel}_4096_{label}.ncu-rep"
            if rep.exists():
                t = extract_compute_time_ms(rep)
                data[kernel][density] = t
                print(f"K{kernel} {density}%: {t:.2f} ms")
    return data


def plot():
    print("Loading ablation data...")
    data = load_data()

    kernels = [27, 28, 26, 29]
    x = np.arange(len(DENSITIES))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(W2, 3.2))

    for i, kernel in enumerate(kernels):
        vals = [data[kernel].get(d, 0) for d in DENSITIES]
        bars = ax.bar(x + offsets[i] * width, vals, width,
                      label=KERNEL_LABELS[kernel], color=KERNEL_COLORS[kernel],
                      edgecolor="white", linewidth=0.5,
                      zorder=3 if kernel == 29 else 2)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    k27_12 = data[27].get(12.5, 0)
    k29_12 = data[29].get(12.5, 0)
    if k27_12 > 0 and k29_12 > 0:
        improvement = (k27_12 - k29_12) / k27_12 * 100
        ax.annotate(f"−{improvement:.0f}%",
                    xy=(x[0] + offsets[3] * width, k29_12),
                    xytext=(0, -22), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=COLORS["ESMM"],
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLORS["ESMM"]))

    ax.set_xlabel("Block Density (%)")
    ax.set_ylabel("Compute Time (ms, NCU)")
    ax.set_title("Ablation: Contribution of Each Optimization (4096\u00d74096, blockwise)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.1f}%" for d in DENSITIES])
    ax.legend(loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout()

    out = OUTPUT_DIR / "fig2_ablation"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
