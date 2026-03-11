#!/usr/bin/env python3
"""
Figure 3: ESMM compute and end-to-end speedup over cuBLAS vs. block density.

Uses NCU extraction from paper_data/fig1_density (7 density points: 12.5–100%).
Falls back to 4-point hardcoded Table 1 data if NCU extraction fails.
"""

import subprocess, re, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from paper_style import apply as apply_style, COLORS, W1

apply_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmarks" / "paper_data" / "fig1_density"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESS_US = 374.0  # µs, fixed across all sparsity levels

# 7 density points available in fig1_density
NCU_DENSITIES   = [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 100.0]
NCU_LABELS      = ["12_5pct", "25_0pct", "37_5pct", "50_0pct", "62_5pct", "75_0pct", "100_0pct"]

# Fallback: exact Table 1 values (4096×4096, from NCU)
TABLE1_DENSITIES        = [100.0, 50.0, 25.0, 12.5]
TABLE1_CUBLAS_MS        = 7.19
TABLE1_ESMM_COMPUTE_MS  = [12.10, 6.12, 4.04, 2.72]
TABLE1_ESMM_E2E_MS      = [12.47, 6.49, 4.41, 3.09]


def extract_kernel_times_us(ncu_rep_path):
    """Return (preprocess_us, compute_us) from an NCU report. Returns (0, None) on failure."""
    cmd = [
        "sudo", "/usr/local/cuda-12.1/bin/ncu",
        "--import", str(ncu_rep_path),
        "--csv", "--page", "details",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/usr/local/bin"}, timeout=30)
    if result.returncode != 0:
        return 0.0, None

    preprocess_us = 0.0
    last_compute_us = None

    for line in result.stdout.split("\n"):
        if '"Duration"' not in line:
            continue
        is_pre = "preprocess" in line.lower()

        if '"msecond"' in line:
            m = re.search(r'"msecond","([^"]+)"', line)
            if m:
                t_us = float(m.group(1).replace(",", "")) * 1000
                if is_pre:
                    preprocess_us += t_us
                else:
                    last_compute_us = t_us

        elif '"usecond"' in line:
            m = re.search(r'"usecond","([^"]+)"', line)
            if m:
                t_us = float(m.group(1).replace(",", ""))
                if is_pre:
                    preprocess_us += t_us
                else:
                    last_compute_us = t_us

    return preprocess_us, last_compute_us


def load_ncu_data():
    """Try to load K15 and K29 times from NCU files. Returns None on any failure."""
    k15_times, k29_compute, k29_e2e = [], [], []

    for density, label in zip(NCU_DENSITIES, NCU_LABELS):
        rep15 = DATA_DIR / f"k15_4096_{label}.ncu-rep"
        rep29 = DATA_DIR / f"k29_4096_{label}.ncu-rep"

        if not rep15.exists() or not rep29.exists():
            print(f"  Missing NCU file at {density}%, falling back to table data")
            return None

        try:
            _, t15 = extract_kernel_times_us(rep15)
            _, t29 = extract_kernel_times_us(rep29)
        except Exception as e:
            print(f"  NCU extraction failed ({e}), falling back to table data")
            return None

        if t15 is None or t29 is None:
            print(f"  NCU parse failed at {density}%, falling back to table data")
            return None

        k15_times.append(t15)
        k29_compute.append(t29)
        k29_e2e.append(t29 + PREPROCESS_US)
        print(f"  K15 {density}%: {t15/1000:.2f}ms  K29: {t29/1000:.2f}ms  e2e: {(t29+PREPROCESS_US)/1000:.2f}ms")

    cublas_us = np.mean(k15_times)
    compute_speedup = [cublas_us / t for t in k29_compute]
    e2e_speedup     = [cublas_us / t for t in k29_e2e]

    return {
        "densities": NCU_DENSITIES,
        "compute_speedup": compute_speedup,
        "e2e_speedup": e2e_speedup,
        "cublas_ms": cublas_us / 1000,
    }


def table_fallback():
    """Return hardcoded Table 1 speedup data (4 points)."""
    compute_speedup = [TABLE1_CUBLAS_MS / t for t in TABLE1_ESMM_COMPUTE_MS]
    e2e_speedup     = [TABLE1_CUBLAS_MS / t for t in TABLE1_ESMM_E2E_MS]
    return {
        "densities": TABLE1_DENSITIES,
        "compute_speedup": compute_speedup,
        "e2e_speedup": e2e_speedup,
        "cublas_ms": TABLE1_CUBLAS_MS,
    }


def plot(data):
    densities = data["densities"]
    compute_su = data["compute_speedup"]
    e2e_su     = data["e2e_speedup"]
    cublas_ms  = data["cublas_ms"]

    fig, ax = plt.subplots(figsize=(W1, 2.8))

    # cuBLAS parity reference
    ax.axhline(y=1.0, color="#555555", linestyle="--", linewidth=1.5,
               label="cuBLAS parity (1.0×)", zorder=1)

    # Breakeven annotation near 50%
    ax.axvline(x=50, color="#888888", linestyle=":", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text(51, 0.35, "Break-even\n~50%", fontsize=8, color="#666666", va="bottom")

    # Speedup lines
    ax.plot(densities, compute_su, "D-", color=COLORS["ESMM"], linewidth=2.0,
            markersize=5, label="Compute speedup", zorder=4, clip_on=False)
    ax.plot(densities, e2e_su, "o--", color="#ff7f0e", linewidth=1.5,
            markersize=5, label=f"End-to-end (+{PREPROCESS_US:.0f} \u03bcs preprocess)",
            zorder=3, clip_on=False)

    # Annotate peaks at lowest density
    d_min = min(densities)
    idx_min = densities.index(d_min)
    ax.annotate(f"{compute_su[idx_min]:.2f}×",
                xy=(d_min, compute_su[idx_min]),
                xytext=(10, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#d62728")
    ax.annotate(f"{e2e_su[idx_min]:.2f}×",
                xy=(d_min, e2e_su[idx_min]),
                xytext=(10, -14), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#ff7f0e")

    ax.set_xlabel("Block Density (%)")
    ax.set_ylabel("Speedup over cuBLAS")
    ax.set_title("ESMM Speedup vs. Block Density (4096\u00d74096)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="both")
    ax.set_xlim(min(densities) - 5, 108)
    ax.set_ylim(0.0, 3.2)

    # X ticks at data points
    ax.set_xticks(sorted(set([12.5, 25, 37.5, 50, 62.5, 75, 100]) & set(densities + [100.0])))
    ax.xaxis.set_tick_params(labelsize=9)
    ax.yaxis.set_tick_params(labelsize=9)

    plt.tight_layout()

    out = OUTPUT_DIR / "fig3_speedup_curve"
    plt.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}.pdf / .png")


def main():
    print("Attempting NCU extraction from paper_data/fig1_density...")
    data = load_ncu_data()

    if data is None:
        print("Using hardcoded Table 1 values (4-point fallback).")
        data = table_fallback()
        source = "\n(Table 1 values)"
    else:
        source = "\n(NCU extracted)"

    plot(data)


if __name__ == "__main__":
    main()
