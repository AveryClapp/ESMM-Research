#!/usr/bin/env python3
"""
Figure 1: Three-level skip hierarchy in ESMM.

Grid: rows = warps (W0 top .. W3 bottom), columns = K-tiles (K0..K7).
Colors encode which skip level applies to each (warp, K-tile) cell.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS

apply_style()

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_BLOCK   = COLORS["ESMM"]          # red   — block-level skip
C_WARP    = "#f28e2b"                # orange — warp-level skip
C_DOTIDX  = COLORS["AB-Cached-32"]  # green  — dotIdx masked
C_COMPUTE = COLORS["AB-Cached-64"]  # blue   — computed

NUM_K = 8
NUM_W = 4   # warps shown (W0 top, W3 bottom in display)

# (warp_display_row, k): 0=top warp, 3=bottom warp
# W0 displayed at top for readability
SCENARIOS = {
    **{(w, 0): "block"   for w in range(NUM_W)},       # K0: full block skip
    **{(w, 1): "warp"    for w in [0, 1]},             # K1: W0/W1 warp-skip
    **{(w, 1): "compute" for w in [2, 3]},             #     W2/W3 compute
    (0, 2): "dotidx",                                   # K2: W0 dotIdx
    **{(w, 2): "warp"    for w in [1, 2, 3]},          #     W1-3 warp-skip
    **{(w, 3): "block"   for w in range(NUM_W)},       # K3: full block skip
    **{(w, 4): "warp"    for w in [1, 3]},             # K4: W1/W3 warp-skip
    **{(w, 4): "compute" for w in [0, 2]},             #     W0/W2 compute
    **{(w, 5): "warp"    for w in range(NUM_W)},       # K5: all warp-skip
    (0, 6): "dotidx",                                   # K6: W0 dotIdx
    **{(w, 6): "warp"    for w in [1, 2, 3]},          #     W1-3 warp-skip
    **{(w, 7): "compute" for w in range(NUM_W)},       # K7: all compute
}

COLOR_MAP = {
    "block":   C_BLOCK,
    "warp":    C_WARP,
    "dotidx":  C_DOTIDX,
    "compute": C_COMPUTE,
}

CELL_LABEL = {
    "block":   "skip",
    "warp":    "skip",
    "dotidx":  "mask",
    "compute": "acc",
}


def plot():
    # Grid geometry
    TW, TH = 1.0, 0.78   # tile width / height
    PX, PY = 0.10, 0.08  # padding between tiles

    LEFT   = 1.8    # space left of grid for warp labels
    BOTTOM = 0.9    # space below grid for K labels
    TOP    = 0.8    # space above grid for block-skip annotations
    RIGHT  = 0.3    # small right margin (legend placed externally)

    grid_w = NUM_K * TW + (NUM_K - 1) * PX
    grid_h = NUM_W * TH + (NUM_W - 1) * PY

    fig_w = LEFT + grid_w + RIGHT + 3.2   # +3.2 for side legend
    fig_h = BOTTOM + grid_h + TOP + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    # Origin of grid (bottom-left corner)
    gx0 = LEFT
    gy0 = BOTTOM

    # ── Draw tiles ──────────────────────────────────────────────────
    for k in range(NUM_K):
        for w_idx in range(NUM_W):
            # w_idx=0 → top row, so flip y
            row = (NUM_W - 1 - w_idx)   # row 0 = top display
            cell_type = SCENARIOS.get((w_idx, k), "compute")
            fc = COLOR_MAP[cell_type]
            lbl = CELL_LABEL[cell_type]

            x = gx0 + k * (TW + PX)
            y = gy0 + row * (TH + PY)

            rect = mpatches.FancyBboxPatch(
                (x, y), TW, TH,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor="white",
                linewidth=1.2, zorder=3)
            ax.add_patch(rect)

            # Cell label
            text_color = "white"
            ax.text(x + TW / 2, y + TH / 2, lbl,
                    ha="center", va="center",
                    fontsize=7.5, color=text_color,
                    fontweight="bold", zorder=4)

    # ── K-tile axis labels ───────────────────────────────────────────
    for k in range(NUM_K):
        xc = gx0 + k * (TW + PX) + TW / 2
        ax.text(xc, gy0 - 0.22, f"K{k}",
                ha="center", va="top", fontsize=9, color="#333", fontweight="bold")

    # X-axis label
    ax.text(gx0 + grid_w / 2, gy0 - 0.58,
            "K-tile index (K-dimension)",
            ha="center", va="top", fontsize=9, color="#555")

    # ── Warp axis labels ─────────────────────────────────────────────
    for w_idx in range(NUM_W):
        row = (NUM_W - 1 - w_idx)
        yc = gy0 + row * (TH + PY) + TH / 2
        ax.text(gx0 - 0.22, yc, f"W{w_idx}",
                ha="right", va="center", fontsize=9, color="#333", fontweight="bold")

    # Y-axis label (rotated, well clear of W labels)
    ax.text(gx0 - 1.35, gy0 + grid_h / 2,
            "Warp",
            ha="center", va="center", fontsize=9, color="#555", rotation=90)

    # ── Block-skip column brackets ───────────────────────────────────
    bracket_y = gy0 + grid_h + 0.18
    for k_blk in [0, 3]:
        xc = gx0 + k_blk * (TW + PX) + TW / 2
        # Bracket line
        bw = TW * 0.42
        ax.plot([xc - bw, xc + bw], [bracket_y, bracket_y],
                color=C_BLOCK, lw=1.8, solid_capstyle="round", zorder=5)
        ax.plot([xc - bw, xc - bw], [bracket_y, bracket_y - 0.10],
                color=C_BLOCK, lw=1.8, solid_capstyle="round", zorder=5)
        ax.plot([xc + bw, xc + bw], [bracket_y, bracket_y - 0.10],
                color=C_BLOCK, lw=1.8, solid_capstyle="round", zorder=5)
        ax.text(xc, bracket_y + 0.10, "Block\nskip",
                ha="center", va="bottom", fontsize=7.5,
                color=C_BLOCK, fontweight="bold")

    # ── Legend (right side, well clear of grid) ──────────────────────
    legend_x = gx0 + grid_w + 0.55
    legend_y_start = gy0 + grid_h

    legend_items = [
        (C_BLOCK,   "Block-level skip",   "All warps skip entire K-tile.\nNo smem load, no sync."),
        (C_WARP,    "Warp-level skip",    "Single warp skips K-tile.\nOther warps may still compute."),
        (C_DOTIDX,  "dotIdx masked",      "Warp active but this K-column\nhas joint bit = 0, skip FMA."),
        (C_COMPUTE, "Computed",           "Full outer product\naccumulated into C tile."),
    ]

    entry_h = 0.72
    box_sz  = 0.32

    for i, (color, title, desc) in enumerate(legend_items):
        ly = legend_y_start - i * entry_h - 0.1

        # Color swatch
        swatch = mpatches.FancyBboxPatch(
            (legend_x, ly - box_sz / 2), box_sz, box_sz,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.add_patch(swatch)

        # Title (bold)
        ax.text(legend_x + box_sz + 0.18, ly + box_sz * 0.15,
                title,
                ha="left", va="center", fontsize=8.5,
                fontweight="bold", color="#222", zorder=5)

        # Description
        ax.text(legend_x + box_sz + 0.18, ly - box_sz * 0.55,
                desc,
                ha="left", va="center", fontsize=7,
                color="#555", zorder=5, linespacing=1.3)

    # Legend box border
    leg_box_w = 3.0
    leg_box_h = len(legend_items) * entry_h + 0.2
    leg_rect = mpatches.FancyBboxPatch(
        (legend_x - 0.15, legend_y_start - leg_box_h + 0.05),
        leg_box_w, leg_box_h,
        boxstyle="round,pad=0.08",
        facecolor="#fafafa", edgecolor="#cccccc",
        linewidth=0.8, zorder=2)
    ax.add_patch(leg_rect)

    # ── Title ────────────────────────────────────────────────────────
    ax.set_title("Three-Level Skip Hierarchy in ESMM",
                 fontsize=11, pad=12, fontweight="bold")

    # ── Axis limits ──────────────────────────────────────────────────
    ax.set_xlim(0, gx0 + grid_w + 3.8)
    ax.set_ylim(0, gy0 + grid_h + TOP + 0.6)

    plt.tight_layout(pad=0.3)
    out = OUTPUT_DIR / "fig1_skip_hierarchy"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
