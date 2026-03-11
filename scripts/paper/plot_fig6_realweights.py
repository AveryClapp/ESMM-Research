#!/usr/bin/env python3
"""
Figure 6: ESMM end-to-end time vs. cuBLAS on real LLaMA-7B weight tensors.

Reads results/real_weights_benchmark.csv.
Shows speedup (cuBLAS/ESMM) vs. element sparsity for attention and MLP shapes,
for both Wanda and SparseGPT pruners.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W2

apply_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH  = PROJECT_ROOT / "results" / "real_weights_benchmark.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(CSV_PATH)
    # k29_speedup = k15_ms / k29_ms (>1 means ESMM faster)
    return df


def compute_mean_speedup(df, pruner, shape, group_size=None):
    """
    Average k29_speedup across projections at each sparsity level.
    shape: '4096x4096' (attn) or '4096x11008' (mlp)
    """
    mask = (df.pruner == pruner) & (df.orig_shape == shape)
    if group_size is not None:
        mask &= (df.group_size == group_size)

    sub = df[mask].groupby("elem_sparsity_pct")["k29_speedup"].mean().reset_index()
    sub = sub.sort_values("elem_sparsity_pct")
    return sub["elem_sparsity_pct"].values, sub["k29_speedup"].values


def plot():
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.2),
                                    sharey=False, sharex=False)

    MARKERS = {"wanda": "D", "sparsegpt": "s"}

    # ── Panel (a): Attention matrices (4096×4096) ──
    configs_attn = [
        ("wanda",     8,  COLORS["ESMM"],        "-",  "Wanda (grp8)"),
        ("sparsegpt", 16, COLORS["AB-Cached-64"], "--", "SparseGPT (grp16)"),
    ]

    for pruner, gs, color, ls, label in configs_attn:
        sp, su = compute_mean_speedup(df, pruner, "4096x4096", group_size=gs)
        ax1.plot(sp, su, marker=MARKERS[pruner], color=color, linestyle=ls,
                 linewidth=2.0, markersize=7, label=label, zorder=3)

    ax1.axhline(1.0, color="#555555", linestyle=":", linewidth=1.8,
                label="cuBLAS parity (1.0×)", zorder=1)

    # Shade the "no speedup" region
    ax1.axhspan(0.0, 1.0, alpha=0.04, color="#d62728", zorder=0)

    ax1.set_xlabel("Element Sparsity (%)")
    ax1.set_ylabel("ESMM Speedup over cuBLAS")
    ax1.set_title("(a) Attention Matrices (4096\u00d74096)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(48, 101)
    ax1.set_ylim(0.50, 2.6)
    ax1.set_xticks([50, 60, 70, 80, 90, 95, 99])
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}×"))

    # Annotate 99% spike
    sp_w, su_w = compute_mean_speedup(df, "wanda", "4096x4096", group_size=8)
    best_idx = np.argmax(su_w)
    ax1.annotate(f"99% sparsity:\nmean {su_w[best_idx]:.2f}× Wanda\n(80–95% block sparsity)",
                 xy=(sp_w[best_idx], su_w[best_idx]),
                 xytext=(-80, 15), textcoords="offset points",
                 fontsize=8, fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

    # Annotate the flat preprocessing-overhead region
    ax1.text(72, 0.57,
             "50–95%: 0% block sparsity\nESMM slower (preprocessing overhead, no skips)",
             ha="center", fontsize=7.5, color="#888888", style="italic")

    # ── Panel (b): MLP matrices (4096×11008) ──
    configs_mlp = [
        ("wanda",     8,  COLORS["ESMM"],        "-",  "Wanda (grp8)"),
        ("sparsegpt", 16, COLORS["AB-Cached-64"], "--", "SparseGPT (grp16)"),
    ]

    for pruner, gs, color, ls, label in configs_mlp:
        sub = df[(df.pruner == pruner) & (df.orig_shape == "4096x11008") &
                 (df.group_size == gs)].sort_values("elem_sparsity_pct")
        ax2.plot(sub["elem_sparsity_pct"], sub["k29_speedup"],
                 marker=MARKERS[pruner], color=color, linestyle=ls,
                 linewidth=2.0, markersize=7, label=label, zorder=3)

    ax2.axhline(1.0, color="#555555", linestyle=":", linewidth=1.8,
                label="cuBLAS parity (1.0×)", zorder=1)
    ax2.axhspan(0.0, 1.0, alpha=0.04, color="#d62728", zorder=0)

    ax2.set_xlabel("Element Sparsity (%)")
    ax2.set_ylabel("ESMM Speedup over cuBLAS")
    ax2.set_title("(b) MLP down\_proj (4096\u00d711008)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(48, 101)
    ax2.set_ylim(0.50, 2.6)
    ax2.set_xticks([50, 60, 70, 80, 90, 95, 99])
    ax2.xaxis.set_tick_params(labelsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}×"))

    # Annotate MLP 99% spike
    mlp_w = df[(df.pruner == "wanda") & (df.orig_shape == "4096x11008") &
               (df.group_size == 8)].sort_values("elem_sparsity_pct")
    v99 = mlp_w[mlp_w.elem_sparsity_pct == 99]["k29_speedup"].values[0]
    ax2.annotate(f"99% sparsity:\n~75% block sparsity\n{v99:.2f}× (Wanda)",
                 xy=(99, v99),
                 xytext=(-85, 15), textcoords="offset points",
                 fontsize=8, fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
    ax2.text(72, 0.57,
             "50–95%: 0% block sparsity\n(K=11008 too wide for tile-level zeros)",
             ha="center", fontsize=7.5, color="#888888", style="italic")

    fig.suptitle(
        "ESMM on Real LLaMA-7B Weights (Wanda / SparseGPT pruning, LLaMA Layer 0)\n"
        "Group-structured pruning does not produce 32×8 block sparsity below 99% element sparsity",
        fontsize=10, y=1.03)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig6_real_weights"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
