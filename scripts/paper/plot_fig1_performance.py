#!/usr/bin/env python3
"""
Figure 1: Performance vs Density
ESMM ★, AB-Cached-64 (baseline A+B), cuBLAS, cuSPARSE at 4096×4096 blockwise.
Dual panel: (a) Speedup over cuBLAS, (b) Absolute runtime.
"""

import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W2

apply_style()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmarks" / "paper_data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITIES = [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 100.0]
DENSITY_LABELS = ["12_5pct", "25_0pct", "37_5pct", "50_0pct", "62_5pct", "75_0pct", "100_0pct"]


def extract_e2e_time_us(ncu_rep_path):
    """Extract end-to-end time (all kernels summed) from NCU report."""
    cmd = ["sudo", "/usr/local/cuda-12.1/bin/ncu", "--import", str(ncu_rep_path),
           "--csv", "--page", "details"]
    env = {"PATH": "/usr/bin:/usr/local/bin", "LD_LIBRARY_PATH": "/usr/local/lib64"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    total_us = 0.0
    for line in result.stdout.split("\n"):
        if '"Duration"' not in line:
            continue
        if '"msecond"' in line:
            val = re.search(r'"msecond","([^"]+)"', line)
            if val:
                total_us += float(val.group(1).replace(",", "")) * 1000
        elif '"usecond"' in line:
            val = re.search(r'"usecond","([^"]+)"', line)
            if val:
                total_us += float(val.group(1).replace(",", ""))
    return total_us


def load_ncu_data():
    """Load K15, K20, K29 end-to-end times from NCU reports."""
    rows = []
    for kernel in [15, 20, 29]:
        for density, label in zip(DENSITIES, DENSITY_LABELS):
            rep = DATA_DIR / "fig1_density" / f"k{kernel}_4096_{label}.ncu-rep"
            if rep.exists():
                t = extract_e2e_time_us(rep)
                rows.append({"kernel": kernel, "density": density, "time_us": t})
                print(f"K{kernel} {density}%: {t:.0f} µs")
    return pd.DataFrame(rows)


def load_cusparse_data():
    """Load cuSPARSE total (conversion + SpMM) times."""
    csv_path = DATA_DIR / "cusparse_4096.csv"
    df = pd.read_csv(csv_path)
    df["density"] = df["density"] * 100
    df = df.rename(columns={"total_us": "time_us"})
    df["kernel"] = "cuSPARSE"
    return df[["kernel", "density", "time_us"]]


def plot():
    print("Loading NCU data...")
    ncu_df = load_ncu_data()

    print("\nLoading cuSPARSE data...")
    cusparse_df = load_cusparse_data()

    # cuBLAS baseline (constant across density)
    cublas_time = ncu_df[ncu_df["kernel"] == 15]["time_us"].mean()
    print(f"\ncuBLAS baseline: {cublas_time:.0f} µs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.5))

    styles = {
        29:         {"label": "ESMM (ours)",   "marker": "D", "color": COLORS["ESMM"],        "lw": 2.0, "zorder": 5},
        20:         {"label": "AB-Cached-64",  "marker": "s", "color": COLORS["AB-Cached-64"],"lw": 1.5, "zorder": 4},
        "cuSPARSE": {"label": "cuSPARSE",     "marker": "^", "color": COLORS["cuSPARSE"],    "lw": 1.5, "zorder": 3},
    }

    # --- Panel (a): Speedup over cuBLAS ---
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=2,
                label="cuBLAS (dense)", zorder=1)

    for kernel, style in styles.items():
        if kernel == "cuSPARSE":
            kdata = cusparse_df.sort_values("density")
        else:
            kdata = ncu_df[ncu_df["kernel"] == kernel].sort_values("density")
        if kdata.empty:
            continue
        speedup = cublas_time / kdata["time_us"].values
        ax1.plot(kdata["density"].values, speedup,
                 marker=style["marker"], linewidth=style["lw"], markersize=8,
                 label=style["label"], color=style["color"], zorder=style["zorder"])

    # Annotate K29 at 25% density
    k29 = ncu_df[ncu_df["kernel"] == 29]
    k29_25 = k29[k29["density"] == 25.0]
    if not k29_25.empty:
        sp = cublas_time / k29_25["time_us"].values[0]
        ax1.annotate(f"{sp:.2f}×", xy=(25.0, sp), xytext=(10, 10),
                     textcoords="offset points", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    ax1.set_xlabel("Matrix Density (%)")
    ax1.set_ylabel("Speedup vs. cuBLAS")
    ax1.set_title("(a) Speedup vs. Density")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(5, 105)
    ax1.set_ylim(0, max(3.0, ax1.get_ylim()[1]))

    # --- Panel (b): Absolute runtime ---
    ax2.axhline(y=cublas_time / 1000, color="gray", linestyle="--", linewidth=2,
                label=f"cuBLAS ({cublas_time/1000:.1f} ms)", zorder=1)

    for kernel, style in styles.items():
        if kernel == "cuSPARSE":
            kdata = cusparse_df.sort_values("density")
        else:
            kdata = ncu_df[ncu_df["kernel"] == kernel].sort_values("density")
        if kdata.empty:
            continue
        ax2.plot(kdata["density"].values, kdata["time_us"].values / 1000,
                 marker=style["marker"], linewidth=style["lw"], markersize=8,
                 label=style["label"], color=style["color"], zorder=style["zorder"])

    ax2.set_xlabel("Matrix Density (%)")
    ax2.set_ylabel("Runtime (ms)")
    ax2.set_title("(b) Absolute Runtime vs. Density")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(5, 105)
    ax2.set_yscale("log")

    fig.suptitle("Performance vs. Density (4096\u00d74096, blockwise sparsity)", y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig1_performance_vs_density"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
