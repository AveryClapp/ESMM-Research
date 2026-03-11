#!/usr/bin/env python3
"""
Figure 4: NCU compute times for all kernel variants at each block density.
Data from Table 2 (paper, NCU-verified values).
Grouped bar chart: cuBLAS (flat), AB-Cached-64, AB-Cached-32 (no block skip),
AB-Cached-32, AB-Stream-32, ESMM.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W2

apply_style()

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Table 2 data: NCU compute times (ms), 4096×4096, synthetic blockwise sparsity
# None = not measured ("—" in Table 2)
KERNELS = {
    "cuBLAS":              {"times": {100: 7.19, 50: 7.19, 25: 7.19, 12.5: 7.19},
                            "label": "cuBLAS",
                            "color": COLORS["cuBLAS"], "hatch": "///"},
    "AB-Cached-64":        {"times": {100: 11.65, 50: 6.32, 25: 4.16, 12.5: 3.80},
                            "label": "AB-Cached-64",
                            "color": COLORS["AB-Cached-64"], "hatch": ""},
    "AB-Cached-32-noskip": {"times": {100: None,  50: 6.01, 25: 4.51, 12.5: 4.41},
                            "label": "AB-Cached-32 (no block skip)",
                            "color": COLORS["AB-Cached-32-noskip"], "hatch": ""},
    "AB-Cached-32":        {"times": {100: 12.94, 50: 6.02, 25: 4.22, 12.5: 2.85},
                            "label": "AB-Cached-32",
                            "color": COLORS["AB-Cached-32"], "hatch": ""},
    "AB-Stream-32":        {"times": {100: 13.08, 50: 6.20, 25: 4.22, 12.5: 3.34},
                            "label": "AB-Stream-32",
                            "color": COLORS["AB-Stream-32"], "hatch": ""},
    "ESMM":                {"times": {100: 12.10, 50: 6.12, 25: 4.04, 12.5: 2.72},
                            "label": "ESMM (ours)",
                            "color": COLORS["ESMM"], "hatch": ""},
}

DENSITIES = [100, 50, 25, 12.5]
DENSITY_LABELS = ["100%\n(dense)", "50%", "25%", "12.5%\n(87.5% sparse)"]

KERNEL_ORDER = ["cuBLAS", "AB-Cached-64", "AB-Cached-32-noskip", "AB-Cached-32", "AB-Stream-32", "ESMM"]


def plot():
    n_groups = len(DENSITIES)
    n_kernels = len(KERNEL_ORDER)
    width = 0.12
    group_gap = 0.05
    group_width = n_kernels * width + group_gap
    x = np.arange(n_groups) * group_width

    fig, ax = plt.subplots(figsize=(W2, 3.5))

    offsets = np.linspace(-(n_kernels - 1) / 2, (n_kernels - 1) / 2, n_kernels) * width

    for i, kname in enumerate(KERNEL_ORDER):
        kdata = KERNELS[kname]
        vals = [kdata["times"].get(d) for d in DENSITIES]
        colors = []
        heights = []
        for v in vals:
            if v is None:
                heights.append(0)
                colors.append("white")
            else:
                heights.append(v)
                colors.append(kdata["color"])

        # Draw bars where data exists; assign label on first bar with real data
        first_bar = True
        for j, (d, h, c) in enumerate(zip(DENSITIES, heights, colors)):
            if kdata["times"].get(d) is None:
                continue
            bar = ax.bar(x[j] + offsets[i], h, width,
                         color=c, edgecolor="black", linewidth=0.6,
                         label=kdata["label"] if first_bar else "_nolegend_",
                         zorder=3)
            first_bar = False

            # Value labels on bars (only for key kernels to avoid clutter)
            if kname in ("cuBLAS", "ESMM") or (kname == "AB-Cached-64" and d == 12.5):
                ax.text(x[j] + offsets[i], h + 0.12,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color="#d62728" if kname == "ESMM" else "#333333")

    # AB-Cached-32-noskip "not measured" marker at 100%
    ax.text(x[0] + offsets[2], 0.3, "—", ha="center", va="bottom",
            fontsize=11, color="#888888")

    # Improvement annotation at 12.5% density: AB-Cached-32-noskip → ESMM
    k27_12 = KERNELS["AB-Cached-32-noskip"]["times"][12.5]
    esmm_12 = KERNELS["ESMM"]["times"][12.5]
    pct = (k27_12 - esmm_12) / k27_12 * 100
    ax.annotate(f"−{pct:.0f}% vs AB-Cached-32\n(no block skip)",
                xy=(x[3] + offsets[5], esmm_12),
                xytext=(18, 15), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

    ax.set_ylabel("Compute Time (ms, NCU)")
    ax.set_title("Kernel Ablation: Compute Times at Each Block Density (4096×4096)")
    ax.set_xticks(x)
    ax.set_xticklabels(DENSITY_LABELS)
    ax.set_xlabel("Block Density")
    ax.set_ylim(0, 16)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", ncol=2)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_ablation"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
