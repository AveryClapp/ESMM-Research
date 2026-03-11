#!/usr/bin/env python3
"""
Figure 4: Roofline Model
K29 at different density levels + cuBLAS plotted against the A10G hardware roofline.
Shows K29 transitions from memory-bound (low density) toward compute-bound (high density).
"""

import subprocess
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W1

apply_style()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmarks" / "paper_data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A10G hardware specs
PEAK_GFLOPS = 31240      # FP32 peak GFLOPS
PEAK_BW_GBS = 600.0      # Memory bandwidth GB/s
RIDGE_POINT = PEAK_GFLOPS / PEAK_BW_GBS  # ~52 FLOP/byte


def extract_throughput_pct(ncu_rep_path):
    """Extract DRAM throughput % and Compute throughput % for the LAST (compute) kernel."""
    cmd = ["sudo", "/usr/local/cuda-12.1/bin/ncu", "--import", str(ncu_rep_path),
           "--csv", "--page", "details"]
    env = {"PATH": "/usr/bin:/usr/local/bin", "LD_LIBRARY_PATH": "/usr/local/lib64"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    dram_pct = None
    compute_pct = None

    for line in result.stdout.split("\n"):
        # Only use the last occurrence (compute kernel, not preprocessing)
        if '"Memory Throughput"' in line and '"%"' in line:
            val = re.search(r'"%","([^"]+)"', line)
            if val:
                dram_pct = float(val.group(1).replace(",", ""))
        if '"Compute (SM) Throughput"' in line and '"%"' in line:
            val = re.search(r'"%","([^"]+)"', line)
            if val:
                compute_pct = float(val.group(1).replace(",", ""))

    return dram_pct, compute_pct


def plot():
    print("Loading roofline data...")

    # Collect K29 data points at various densities
    densities = [12.5, 25.0, 50.0, 100.0]
    density_labels = ["12_5pct", "25_0pct", "50_0pct", "100_0pct"]

    points = []

    # K29 at each density
    for density, label in zip(densities, density_labels):
        rep = DATA_DIR / "fig1_density" / f"k29_4096_{label}.ncu-rep"
        dram, compute = extract_throughput_pct(rep)
        if dram and compute:
            achieved_gflops = PEAK_GFLOPS * compute / 100
            achieved_bw = PEAK_BW_GBS * dram / 100
            ai = achieved_gflops / achieved_bw
            points.append({
                "label": f"ESMM @ {density:.0f}%" if density == int(density) else f"ESMM @ {density:.1f}%",
                "ai": ai, "gflops": achieved_gflops,
                "color": COLORS["ESMM"], "density": density
            })
            print(f"  K29 {density}%: AI={ai:.1f}, GFLOPS={achieved_gflops:.0f}, DRAM={dram:.1f}%, Compute={compute:.1f}%")

    # cuBLAS
    rep = DATA_DIR / "fig1_density" / "k15_4096_100_0pct.ncu-rep"
    dram, compute = extract_throughput_pct(rep)
    if dram and compute:
        achieved_gflops = PEAK_GFLOPS * compute / 100
        achieved_bw = PEAK_BW_GBS * dram / 100
        ai = achieved_gflops / achieved_bw
        points.append({
            "label": "cuBLAS", "ai": ai, "gflops": achieved_gflops,
            "color": "gray", "density": 100
        })
        print(f"  cuBLAS: AI={ai:.1f}, GFLOPS={achieved_gflops:.0f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(W1, 3.0))

    # Draw roofline ceiling
    ai_range = np.logspace(0, 3, 500)
    roofline = np.minimum(PEAK_GFLOPS, PEAK_BW_GBS * ai_range)
    ax.plot(ai_range, roofline, "k-", linewidth=2, zorder=1)
    ax.fill_between(ai_range, roofline, alpha=0.05, color="gray")

    # Label ceilings
    ax.text(3, PEAK_BW_GBS * 3 * 0.75, f"Memory BW\n({PEAK_BW_GBS:.0f} GB/s)",
            fontsize=9, rotation=38, color="gray", fontweight="bold")
    ax.text(200, PEAK_GFLOPS * 0.85, f"Peak Compute ({PEAK_GFLOPS/1000:.1f} TFLOPS)",
            fontsize=9, color="gray", fontweight="bold", ha="center")

    # Ridge point
    ax.axvline(x=RIDGE_POINT, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(RIDGE_POINT, PEAK_GFLOPS * 0.05, f"Ridge\n({RIDGE_POINT:.0f})",
            ha="center", fontsize=8, color="gray")

    # Plot K29 points with arrow showing density progression
    k29_points = [p for p in points if "ESMM" in p["label"]]
    k29_points.sort(key=lambda p: p["density"])

    # Draw connecting arrow
    for i in range(len(k29_points) - 1):
        ax.annotate("", xy=(k29_points[i+1]["ai"], k29_points[i+1]["gflops"]),
                     xytext=(k29_points[i]["ai"], k29_points[i]["gflops"]),
                     arrowprops=dict(arrowstyle="-", color=COLORS["ESMM"], alpha=0.4, lw=1.5))

    # Plot points
    for p in points:
        marker = "D" if "ESMM" in p["label"] else "s"
        size = 120 if "ESMM" in p["label"] else 100
        ax.scatter(p["ai"], p["gflops"], s=size, marker=marker,
                   color=p["color"], edgecolors="black", linewidths=0.8, zorder=5)
        # Label
        offset = (8, -15) if "cuBLAS" in p["label"] else (8, 8)
        ax.annotate(p["label"], xy=(p["ai"], p["gflops"]),
                    xytext=offset, textcoords="offset points",
                    fontsize=9, fontweight="bold", color=p["color"])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Achieved Performance (GFLOPS)")
    ax.set_title("Roofline Model (NVIDIA Ampere, 4096\u00d74096)")
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(4, 512)
    ax.set_ylim(1000, PEAK_GFLOPS * 1.5)

    # Legend
    k29_patch = mpatches.Patch(color=COLORS["ESMM"], label="ESMM (ours)")
    cublas_patch = mpatches.Patch(color="gray", label="cuBLAS")
    ax.legend(handles=[k29_patch, cublas_patch], fontsize=10, loc="lower right")

    plt.tight_layout()

    out = OUTPUT_DIR / "fig4_roofline"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out}.pdf / .png")


if __name__ == "__main__":
    plot()
