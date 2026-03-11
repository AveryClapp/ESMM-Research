#!/usr/bin/env python3
"""
Figure 3: End-to-End Breakdown + Size Scaling
(a) Stacked bar: K29 preprocessing vs compute at each density (4096×4096)
(b) K29 vs cuBLAS speedup across matrix sizes at 25% density
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


def extract_times(ncu_rep_path):
    """Extract (preprocess_us, compute_us) from NCU report."""
    cmd = ["sudo", "/usr/local/cuda-12.1/bin/ncu", "--import", str(ncu_rep_path),
           "--csv", "--page", "details"]
    env = {"PATH": "/usr/bin:/usr/local/bin", "LD_LIBRARY_PATH": "/usr/local/lib64"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    preprocess_us = 0.0
    compute_us = 0.0

    for line in result.stdout.split("\n"):
        if '"Duration"' not in line:
            continue

        is_preprocess = "preprocess" in line.lower()

        if '"msecond"' in line:
            val = re.search(r'"msecond","([^"]+)"', line)
            if val:
                t = float(val.group(1).replace(",", "")) * 1000
                if is_preprocess:
                    preprocess_us += t
                else:
                    compute_us = t  # last non-preprocess msecond wins
        elif '"usecond"' in line:
            val = re.search(r'"usecond","([^"]+)"', line)
            if val:
                t = float(val.group(1).replace(",", ""))
                if is_preprocess:
                    preprocess_us += t
                else:
                    compute_us = t  # last non-preprocess usecond wins

    return preprocess_us, compute_us


def extract_total_time_us(ncu_rep_path):
    """Extract total time (sum of all kernel durations)."""
    pre, comp = extract_times(ncu_rep_path)
    return pre + comp


def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.2))

    # --- Panel (a): Preprocessing vs Compute breakdown for K29 ---
    print("Panel (a): K29 breakdown by density...")
    densities = [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 100.0]
    density_labels = ["12_5pct", "25_0pct", "37_5pct", "50_0pct", "62_5pct", "75_0pct", "100_0pct"]

    preprocess_times = []
    compute_times = []
    for density, label in zip(densities, density_labels):
        rep = DATA_DIR / "fig1_density" / f"k29_4096_{label}.ncu-rep"
        pre, comp = extract_times(rep)
        preprocess_times.append(pre / 1000)  # ms
        compute_times.append(comp / 1000)    # ms
        pct = pre / (pre + comp) * 100
        print(f"  {density}%: preprocess={pre:.0f}µs, compute={comp:.0f}µs, overhead={pct:.1f}%")

    x = np.arange(len(densities))
    ax1.bar(x, compute_times, label="Compute", color=COLORS["ESMM"], edgecolor="white", linewidth=0.5)
    ax1.bar(x, preprocess_times, bottom=compute_times, label="Preprocessing",
            color=COLORS["AB-Cached-64"], edgecolor="white", linewidth=0.5)

    # Add overhead % labels
    for i, (pre, comp) in enumerate(zip(preprocess_times, compute_times)):
        total = pre + comp
        pct = pre / total * 100
        ax1.text(i, total + 0.15, f"{pct:.0f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#1f77b4")

    ax1.set_xlabel("Matrix Density (%)")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("(a) Preprocessing Overhead (ESMM, 4096\u00d74096)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d:.0f}%" if d == int(d) else f"{d:.1f}%" for d in densities], fontsize=9)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Panel (b): Size scaling at 25% density ---
    print("\nPanel (b): Size scaling at 25% density...")
    sizes = [2048, 4096]

    cublas_times = []
    k29_times = []

    for size in sizes:
        if size == 4096:
            cublas_rep = DATA_DIR / "fig1_density" / "k15_4096_25_0pct.ncu-rep"
            k29_rep = DATA_DIR / "fig1_density" / "k29_4096_25_0pct.ncu-rep"
        else:
            cublas_rep = DATA_DIR / "fig3b_scaling" / f"k15_{size}_25_0pct.ncu-rep"
            k29_rep = DATA_DIR / "fig3b_scaling" / f"k29_{size}_25_0pct.ncu-rep"

        ct = extract_total_time_us(cublas_rep) / 1000  # ms
        kt = extract_total_time_us(k29_rep) / 1000     # ms
        cublas_times.append(ct)
        k29_times.append(kt)
        speedup = ct / kt if kt > 0 else 0
        print(f"  {size}: cuBLAS={ct:.2f}ms, K29={kt:.2f}ms, speedup={speedup:.2f}×")

    speedups = [c / k if k > 0 else 0 for c, k in zip(cublas_times, k29_times)]

    ax2.plot(sizes, speedups, marker="D", linewidth=2.0, markersize=5,
             label="ESMM (ours)", color=COLORS["ESMM"])
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=2,
                label="cuBLAS (dense)", zorder=1)

    # Annotate each point
    for size, sp in zip(sizes, speedups):
        ax2.annotate(f"{sp:.2f}×", xy=(size, sp), xytext=(0, 10),
                     textcoords="offset points", fontsize=9, fontweight="bold",
                     ha="center")

    ax2.set_xlabel("Matrix Size (N\u00d7N)")
    ax2.set_ylabel("Speedup vs. cuBLAS")
    ax2.set_title("(b) Size Scaling (25% density, blockwise)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("End-to-End Analysis", y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig3_breakdown_scaling"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
