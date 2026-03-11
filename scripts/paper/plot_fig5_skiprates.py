#!/usr/bin/env python3
"""
Figure 5: Skip rate breakdown by level at each synthetic block density.
Data from Table 3 (paper, NCU-verified + skip-stats instrumentation).

Shows a 100%-stacked bar chart: at each density, what fraction of inner-loop
iterations are eliminated at block level, warp level, bit-masked (dotIdx=0),
or actually computed.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W2

apply_style()

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Table 3: skip rates (%), 4096×4096, synthetic blockwise sparsity
# block_skip  = % of K-tiles where ALL warps skip (block_joint[k] == 0)
# warp_skip   = % of REMAINING (non-block-skipped) warp-K-tile pairs that skip
# dotidx_exec = % of ALL inner-loop iterations that actually execute
DENSITIES   = [100.0, 50.0, 25.0, 12.5]
BLOCK_SKIP  = [0.0,    0.0,   6.0,  44.0]   # % of K-tiles
WARP_SKIP   = [0.0,   10.0,  59.8,  88.1]   # % of remaining warp-K-tile pairs
DOTIDX_EXEC = [100.0,  25.0,   6.2,   1.6]  # % of all inner iterations

# ESMM compute time (ms) from Table 3
ESMM_COMPUTE_MS = [9.99, 5.80, 3.99, 2.63]


def compute_fractions(block_skip_pct, warp_skip_pct, dotidx_exec_pct):
    """
    Convert table metrics to fractions of all inner-loop iterations:
      - block_saved_frac:   K-tiles where block_joint=0 → all 8 warps × 8 dotIdx skipped
      - warp_saved_frac:    remaining warp-K-tile pairs where js[w][k]=0
      - bitmask_frac:       active dotIdx iterations where bit is 0
      - computed_frac:      iterations that actually accumulate
    """
    b = block_skip_pct / 100.0      # fraction of K-tiles block-skipped
    w = warp_skip_pct  / 100.0      # fraction of non-block K-tile pairs warp-skipped
    d = dotidx_exec_pct / 100.0     # fraction of all iterations executed

    block_saved  = b
    warp_saved   = w * (1.0 - b)
    active_frac  = (1.0 - b) * (1.0 - w)   # reach dotIdx loop
    bitmask_frac = active_frac - d
    computed     = d

    return block_saved, warp_saved, bitmask_frac, computed


def plot():
    fracs = [compute_fractions(b, w, d)
             for b, w, d in zip(BLOCK_SKIP, WARP_SKIP, DOTIDX_EXEC)]

    block_f    = [f[0] * 100 for f in fracs]
    warp_f     = [f[1] * 100 for f in fracs]
    bitmask_f  = [f[2] * 100 for f in fracs]
    computed_f = [f[3] * 100 for f in fracs]

    x_labels = ["100%\n(dense)", "50%", "25%", "12.5%\n(87.5% sparse)"]
    x = np.arange(len(DENSITIES))
    bar_w = 0.45

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.2),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # --- Panel (a): Stacked bar (100% stacked, fraction of inner iterations) ---
    colors = {
        "Block-level skip":  COLORS["ESMM"],
        "Warp-level skip":   "#ff7f0e",
        "dotIdx bit-masked": COLORS["AB-Cached-32"],
        "Computed (FMA)":    COLORS["AB-Cached-64"],
    }
    bottom = np.zeros(len(DENSITIES))

    for label, vals in [
        ("Block-level skip",  block_f),
        ("Warp-level skip",   warp_f),
        ("dotIdx bit-masked", bitmask_f),
        ("Computed (FMA)",    computed_f),
    ]:
        bars = ax1.bar(x, vals, bar_w, bottom=bottom,
                       label=label, color=colors[label],
                       edgecolor="white", linewidth=0.4, zorder=3)

        # Annotate non-trivial segments
        for xi, (v, bot) in enumerate(zip(vals, bottom)):
            if v >= 3.0:
                ax1.text(xi, bot + v / 2, f"{v:.0f}%",
                         ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_xlabel("Block Density")
    ax1.set_ylabel("Fraction of Inner Iterations (%)")
    ax1.set_title("(a) Skip Hierarchy Breakdown")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.9,
               bbox_to_anchor=(1.0, 1.0))
    ax1.grid(True, alpha=0.2, axis="y", zorder=0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # --- Panel (b): ESMM compute time ---
    bar_colors = ["#aaaaaa", "#ff7f0e", "#d62728", "#8b0000"]
    bars = ax2.bar(x, ESMM_COMPUTE_MS, 0.5,
                   color=bar_colors, edgecolor="black", linewidth=0.7, zorder=3)
    for bar, t in zip(bars, ESMM_COMPUTE_MS):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{t:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_xlabel("Block Density")
    ax2.set_ylabel("ESMM Compute Time (ms, NCU)")
    ax2.set_title("(b) ESMM Compute Time")
    ax2.set_ylim(0, 12.5)

    ax2.axhline(y=7.19, color=COLORS["cuBLAS"], linestyle="--", linewidth=1.2,
                label="cuBLAS (7.19 ms)")
    ax2.legend(loc="upper left")

    fig.suptitle("Skip Rate Analysis (ESMM, 4096\u00d74096, synthetic blockwise sparsity)", y=1.01)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig5_skiprates"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
