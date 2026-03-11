#!/usr/bin/env python3
"""
Figure 1: Three-level skip hierarchy in ESMM.

Illustrates the three skip levels: block-level, warp-level, and dotIdx-level.
Shows a schematic of K-tiles with colored regions for each skip type.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
C_BLOCK   = "#d62728"   # block-level skip (red)
C_WARP    = "#ff7f0e"   # warp-level skip (orange)
C_DOTIDX  = "#2ca02c"   # dotIdx skip (green)
C_COMPUTE = "#1f77b4"   # computed (blue)
C_ACTIVE  = "#aec7e8"   # active but not skipped (light blue)
C_BG      = "#f7f7f7"   # background tile
C_EDGE    = "#555555"

NUM_K = 8    # K-blocks shown
NUM_W = 4    # warps shown (for display; real kernel has 8)


def draw_tile(ax, x, y, w, h, facecolor, edgecolor="#555555", lw=0.8, alpha=1.0, label=None):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    if label:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=8,
                color="white" if facecolor in (C_BLOCK, C_WARP, C_COMPUTE) else "#333",
                fontweight="bold", zorder=4)


def plot():
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.set_aspect("equal")
    ax.axis("off")

    tile_w = 0.9
    tile_h = 0.75
    pad_x  = 0.12
    pad_y  = 0.10

    # Layout: rows = warps (top=W0..W3), columns = K-tiles (K0..K7)
    # Define which K-tiles / warps skip at each level
    # Based on a representative 25% density scenario

    # K-tile skip pattern:
    # K0: block-skipped (all warps zero)
    # K1: warp-level skip for W0, W1; W2 computes; W3 skips
    # K2: W0 computes (dotIdx partially), W1/W2/W3 warp-skip
    # K3–K5: various warp skips
    # K6: active with dotIdx skip (some bits=0)
    # K7: fully computed (all warps)

    # Simplified for illustration:
    # k_type: 'block', 'warp', 'mixed', 'dotidx', 'compute'
    scenarios = {
        # (warp, k): fill_type
        # K0: full block skip
        **{(w, 0): "block"   for w in range(NUM_W)},
        # K1: mixed (W0/W1 warp-skip, W2/W3 compute)
        **{(w, 1): "warp"    for w in [0, 1]},
        **{(w, 1): "compute" for w in [2, 3]},
        # K2: W0 dotIdx skip, rest warp-skip
        (0, 2): "dotidx",
        **{(w, 2): "warp"    for w in [1, 2, 3]},
        # K3: block skip
        **{(w, 3): "block"   for w in range(NUM_W)},
        # K4: W0/W2 compute, W1/W3 warp-skip
        **{(w, 4): "warp"    for w in [1, 3]},
        **{(w, 4): "compute" for w in [0, 2]},
        # K5: warp skip all
        **{(w, 5): "warp"    for w in range(NUM_W)},
        # K6: dotIdx for W0, warp-skip others
        (0, 6): "dotidx",
        **{(w, 6): "warp"    for w in [1, 2, 3]},
        # K7: all compute
        **{(w, 7): "compute" for w in range(NUM_W)},
    }

    color_map = {
        "block":   C_BLOCK,
        "warp":    C_WARP,
        "dotidx":  C_DOTIDX,
        "compute": C_COMPUTE,
        "bg":      C_BG,
    }
    label_map = {
        "block":   "BLK",
        "warp":    "W",
        "dotidx":  "d",
        "compute": "✓",
    }

    xs0 = 1.5  # left margin
    ys0 = 1.1  # bottom margin

    for k in range(NUM_K):
        for w in range(NUM_W):
            tile_type = scenarios.get((w, k), "bg")
            color = color_map[tile_type]
            lbl   = label_map.get(tile_type, "")
            xpos  = xs0 + k * (tile_w + pad_x)
            ypos  = ys0 + w * (tile_h + pad_y)
            draw_tile(ax, xpos, ypos, tile_w, tile_h, color, lw=0.8, label=lbl)

    # Axis annotations
    total_w = NUM_K * (tile_w + pad_x) - pad_x
    total_h = NUM_W * (tile_h + pad_y) - pad_y

    # K-tile labels (x-axis)
    for k in range(NUM_K):
        xc = xs0 + k * (tile_w + pad_x) + tile_w / 2
        ax.text(xc, ys0 - 0.30, f"K{k}", ha="center", va="top",
                fontsize=9, color="#333", fontweight="bold")
    ax.text(xs0 + total_w / 2, ys0 - 0.65, "K-tile index  →", ha="center",
            va="top", fontsize=10, color="#333")

    # Warp labels (y-axis)
    for w in range(NUM_W):
        yc = ys0 + w * (tile_h + pad_y) + tile_h / 2
        ax.text(xs0 - 0.30, yc, f"W{w}", ha="right", va="center",
                fontsize=9, color="#333", fontweight="bold")
    ax.text(xs0 - 0.75, ys0 + total_h / 2, "Warp →", ha="center", va="center",
            fontsize=10, color="#333", rotation=90)

    # Block-skip bracket around K0 and K3
    for k_blk in [0, 3]:
        xc = xs0 + k_blk * (tile_w + pad_x) + tile_w / 2
        yb = ys0 + total_h + 0.15
        ax.annotate("", xy=(xc, yb + 0.35), xytext=(xc, yb),
                    arrowprops=dict(arrowstyle="-[,widthB=0.6", color=C_BLOCK, lw=1.5))
        ax.text(xc, yb + 0.45, "Block\nskip", ha="center", va="bottom",
                fontsize=7.5, color=C_BLOCK, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color=C_BLOCK,   label="Block-level skip (all warps, no smem access)"),
        mpatches.Patch(color=C_WARP,    label="Warp-level skip (individual warp, no dotIdx)"),
        mpatches.Patch(color=C_DOTIDX,  label="dotIdx bit-masked (active warp, bit=0)"),
        mpatches.Patch(color=C_COMPUTE, label="Computed (outer product accumulated)"),
    ]
    ax.legend(handles=legend_items, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, framealpha=0.9,
              bbox_transform=ax.transAxes)

    ax.set_xlim(0, xs0 + total_w + 1.8)
    ax.set_ylim(-0.3, ys0 + total_h + 1.5)

    ax.set_title("Three-Level Skip Hierarchy in ESMM\n"
                 "(illustrative example, 4 warps × 8 K-tiles)",
                 fontsize=11, pad=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_skip_hierarchy"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
