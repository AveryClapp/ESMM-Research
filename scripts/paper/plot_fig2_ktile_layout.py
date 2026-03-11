#!/usr/bin/env python3
"""
Figure 2: K-tile layout and bitmask encoding.

Shows how 32×8 K-tiles of A and B map to uint8 patterns,
and how the joint pattern is preloaded into shared memory.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme (consistent with other figures)
C_NZ    = "#1f77b4"   # nonzero element
C_ZERO  = "#f0f0f0"   # zero element
C_BIT1  = "#d62728"   # bit = 1 (nonzero column group)
C_BIT0  = "#f0f0f0"   # bit = 0 (all-zero column group)
C_JOINT = "#9467bd"   # joint (AND) bit = 1
C_EDGE  = "#555555"


def draw_matrix_tile(ax, origin, nrows, ncols, pattern,
                     title, cell_w=0.22, cell_h=0.22):
    """
    Draw a small matrix tile with cells colored by pattern (1=nonzero, 0=zero).
    pattern: list of 0/1 of length nrows*ncols, row-major.
    Returns bounding box (x0, y0, x1, y1).
    """
    x0, y0 = origin
    for r in range(nrows):
        for c in range(ncols):
            val = pattern[r * ncols + c]
            rect = mpatches.Rectangle(
                (x0 + c * cell_w, y0 + r * cell_h),
                cell_w, cell_h,
                facecolor=C_NZ if val else C_ZERO,
                edgecolor=C_EDGE, linewidth=0.5)
            ax.add_patch(rect)

    w = ncols * cell_w
    h = nrows * cell_h
    ax.text(x0 + w / 2, y0 + h + 0.08, title,
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    return x0, y0, x0 + w, y0 + h


def draw_bitmask(ax, origin, bits, label, cell_w=0.22, cell_h=0.35, joint=False, zorder=4):
    """Draw a horizontal uint8 bitmask (8 bits)."""
    x0, y0 = origin
    for i, b in enumerate(bits):
        color = (C_JOINT if (joint and b) else C_BIT1) if b else C_BIT0
        rect = mpatches.Rectangle(
            (x0 + i * cell_w, y0), cell_w, cell_h,
            facecolor=color, edgecolor=C_EDGE, linewidth=0.6, zorder=zorder)
        ax.add_patch(rect)
        ax.text(x0 + i * cell_w + cell_w / 2, y0 + cell_h / 2,
                str(b), ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if b else "#666", zorder=zorder + 1)

    ax.text(x0 - 0.12, y0 + cell_h / 2, label,
            ha="right", va="center", fontsize=9, fontweight="bold")
    return x0, y0, x0 + 8 * cell_w, y0 + cell_h


def plot():
    # Carefully chosen example: some K-columns are all-zero in A or B,
    # producing clear zero bits in the joint pattern.
    # A tile: columns 1, 4, 6 are all-zero (sparse A block)
    a_pat_full = np.array([1,0,0,1,0,1,0,1,
                            1,0,1,1,0,0,0,0,
                            0,0,0,1,0,1,0,1,
                            1,0,1,0,0,0,0,1])
    # B tile: columns 2, 5 are all-zero (sparse B block)
    b_pat_full = np.array([1,1,0,1,1,0,1,0,
                            0,1,0,0,1,0,1,1,
                            1,1,0,1,0,0,0,0,
                            1,0,0,1,1,0,1,1])

    # Column-wise OR: a_uint8[d] = any(A[:,d] != 0)
    a_bits = [int(any(a_pat_full[r * 8 + d] for r in range(4))) for d in range(8)]
    b_bits = [int(any(b_pat_full[r * 8 + d] for r in range(4))) for d in range(8)]
    j_bits = [a & b for a, b in zip(a_bits, b_bits)]
    # Expected: a_bits=[1,0,1,1,0,1,0,1], b_bits=[1,1,0,1,1,0,1,1]
    # j_bits  =[1,0,0,1,0,0,0,1]  → 5 skippable columns

    # ── Layout ──
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.axis("off")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 4.5)

    cell_w, cell_h = 0.22, 0.22
    bm_w,  bm_h   = 0.22, 0.32

    # A matrix tile
    x_a, y_a = 0.2, 2.0
    draw_matrix_tile(ax, (x_a, y_a), 4, 8, a_pat_full.tolist(),
                     "A: 32-row × BK=8 tile\n(4 rows shown)",
                     cell_w, cell_h)
    # A bitmask
    ax.annotate("", xy=(x_a + 8 * cell_w + 0.05, y_a + 2 * cell_h),
                xytext=(x_a + 8 * cell_w + 0.45, y_a + 2 * cell_h),
                arrowprops=dict(arrowstyle="<-", color=C_EDGE, lw=1.2))
    ax.text(x_a + 8 * cell_w + 0.25, y_a + 2 * cell_h + 0.12, "col OR",
            ha="center", va="bottom", fontsize=8, color=C_EDGE)

    x_bm_a = x_a + 8 * cell_w + 0.60
    draw_bitmask(ax, (x_bm_a, y_a + 2 * cell_h - bm_h / 2),
                 a_bits, "a_pat[r][k]", bm_w, bm_h)

    # B matrix tile
    x_b, y_b = 0.2, 0.6
    draw_matrix_tile(ax, (x_b, y_b), 4, 8, b_pat_full.tolist(),
                     "B: BN=32 col-group × BK=8 tile\n(4 rows shown)",
                     cell_w, cell_h)
    # B bitmask
    ax.annotate("", xy=(x_b + 8 * cell_w + 0.05, y_b + 2 * cell_h),
                xytext=(x_b + 8 * cell_w + 0.45, y_b + 2 * cell_h),
                arrowprops=dict(arrowstyle="<-", color=C_EDGE, lw=1.2))
    ax.text(x_b + 8 * cell_w + 0.25, y_b + 2 * cell_h + 0.12, "col OR",
            ha="center", va="bottom", fontsize=8, color=C_EDGE)

    x_bm_b = x_bm_a
    draw_bitmask(ax, (x_bm_b, y_b + 2 * cell_h - bm_h / 2),
                 b_bits, "b_pat[c][k]", bm_w, bm_h)

    # AND arrow
    mid_y = (y_a + 2 * cell_h) / 2 + (y_b + 2 * cell_h) / 2
    x_and = x_bm_a + 8 * bm_w + 0.25

    ax.annotate("", xy=(x_and - 0.15, mid_y + 0.40),
                xytext=(x_and - 0.15, mid_y + 0.08),
                arrowprops=dict(arrowstyle="-", color=C_EDGE, lw=1.0))
    ax.annotate("", xy=(x_and - 0.15, mid_y - 0.08),
                xytext=(x_and - 0.15, mid_y - 0.40),
                arrowprops=dict(arrowstyle="-", color=C_EDGE, lw=1.0))

    ax.text(x_and, mid_y, "&\n(bitwise AND)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_JOINT,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=C_JOINT, linewidth=1.2))

    # Joint pattern in smem
    x_joint = x_and + 0.45
    draw_bitmask(ax, (x_joint, mid_y - bm_h / 2),
                 j_bits, "js[w][k]  →", bm_w, bm_h, joint=True, zorder=5)

    # smem box around joint
    x_smem = x_joint - 0.15
    y_smem = mid_y - bm_h / 2 - 0.12
    smem_rect = mpatches.FancyBboxPatch(
        (x_smem, y_smem), 8 * bm_w + 0.30, bm_h + 0.24,
        boxstyle="round,pad=0.05",
        facecolor="#fffde7", edgecolor=C_JOINT,
        linewidth=1.5, linestyle="--", alpha=0.85, zorder=3)
    ax.add_patch(smem_rect)
    ax.text(x_joint + 4 * bm_w, y_smem - 0.08,
            "Shared memory (js[][])\npreloaded once per CUDA block",
            ha="center", va="top", fontsize=8, color=C_JOINT, style="italic")

    # Skip condition annotation
    ax.text(x_joint + 4 * bm_w, mid_y + bm_h / 2 + 0.35,
            "if js[w][k] == 0 → skip K-tile k for warp w",
            ha="center", va="bottom", fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                      edgecolor="#ff7f0e", linewidth=1.0))

    # Legend
    legend_items = [
        mpatches.Patch(color=C_NZ,    label="Nonzero element"),
        mpatches.Patch(color=C_ZERO,  label="Zero element"),
        mpatches.Patch(color=C_BIT1,  label="uint8 bit = 1 (column has nonzero)"),
        mpatches.Patch(color=C_BIT0,  label="uint8 bit = 0 (column all-zero)"),
        mpatches.Patch(color=C_JOINT, label="joint bit = 1 (both A and B nonzero)"),
    ]
    ax.legend(handles=legend_items, fontsize=8.5, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), ncol=2, framealpha=0.95,
              bbox_transform=ax.transAxes)

    ax.set_title("K-Tile Layout and uint8 Bitmask Encoding\n"
                 "Each 32×8 K-tile maps to one uint8 pattern; "
                 "joint = a_pat & b_pat loaded into shared memory",
                 fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_ktile_layout"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
