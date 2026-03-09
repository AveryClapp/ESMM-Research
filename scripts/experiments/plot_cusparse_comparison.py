#!/usr/bin/env python3
"""
cuSPARSE Comparison Plot
Two-panel figure comparing cuSPARSE SpMM vs AB-Fused (K25) vs cuBLAS at N=4096.
Left panel: absolute time (ms) grouped bar chart.
Right panel: speedup vs cuBLAS line chart.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_TO_DENSITY = {
    "10000000": 12.5,
    "11000000": 25.0,
    "11110000": 50.0,
}

DENSITIES = [0.125, 0.25, 0.5]
DENSITY_LABELS = ["12.5%", "25%", "50%"]


def load_cusparse(n=4096):
    f = PROJECT_ROOT / "results" / "cusparse_benchmark" / "cusparse_results.csv"
    if not f.exists():
        print(f"ERROR: {f} not found")
        sys.exit(1)
    df = pd.read_csv(f)
    df = df[df['size'] == n].copy()
    return df


def load_k25_k15(n=4096):
    f = (PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity"
         / "esmm_kernels" / "summary.csv")
    if not f.exists():
        print(f"ERROR: {f} not found")
        sys.exit(1)
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel'])
    df['kernel'] = df['kernel'].astype(int)
    df['density_pct'] = df['pattern'].astype(str).str.strip().map(PATTERN_TO_DENSITY)
    df = df.dropna(subset=['density_pct'])
    df['density'] = df['density_pct'] / 100.0
    df = df[df['size'] == n].copy()
    return df


def get_series(cusparse_df, esmm_df):
    """
    Returns dicts keyed by density fraction (0.125, 0.25, 0.5) for each series.
    Times in microseconds internally; converted to ms for plotting.
    """
    spmm_us = {}
    total_us = {}
    for d in DENSITIES:
        row = cusparse_df[np.isclose(cusparse_df['density'], d, atol=0.01)]
        if row.empty:
            print(f"WARNING: cuSPARSE row missing for density={d}")
            spmm_us[d] = np.nan
            total_us[d] = np.nan
        else:
            spmm_us[d] = float(row['spmm_us'].values[0])
            total_us[d] = float(row['total_us'].values[0])

    k25_us = {}
    k25_data = esmm_df[esmm_df['kernel'] == 25]
    for d in DENSITIES:
        row = k25_data[np.isclose(k25_data['density'], d, atol=0.01)]
        if row.empty:
            print(f"WARNING: K25 row missing for density={d}")
            k25_us[d] = np.nan
        else:
            k25_us[d] = float(row['kernel_time_us'].mean())

    k15_data = esmm_df[esmm_df['kernel'] == 15]
    if k15_data.empty:
        print("WARNING: K15 (cuBLAS) not found, using fallback 7210 µs")
        cublas_us = 7210.0
    else:
        cublas_us = float(k15_data['kernel_time_us'].mean())

    return spmm_us, total_us, k25_us, cublas_us


def print_summary(spmm_us, total_us, k25_us, cublas_us):
    print("\n" + "=" * 72)
    print("  cuSPARSE vs AB-Fused (K25) Summary Table — 4096×4096")
    print("=" * 72)
    header = f"{'Density':>8} | {'cuSP SpMM (ms)':>16} | {'cuSP Total (ms)':>16} | {'K25 (ms)':>10} | {'cuBLAS (ms)':>12}"
    print(header)
    print("-" * 72)
    for d, label in zip(DENSITIES, DENSITY_LABELS):
        spmm_ms = spmm_us[d] / 1000.0 if not np.isnan(spmm_us[d]) else float('nan')
        tot_ms = total_us[d] / 1000.0 if not np.isnan(total_us[d]) else float('nan')
        k25_ms = k25_us[d] / 1000.0 if not np.isnan(k25_us[d]) else float('nan')
        cub_ms = cublas_us / 1000.0
        print(f"{label:>8} | {spmm_ms:>16.2f} | {tot_ms:>16.2f} | {k25_ms:>10.2f} | {cub_ms:>12.2f}")
    print("=" * 72)
    print(f"\ncuBLAS time: {cublas_us:.1f} µs ({cublas_us/1000.0:.2f} ms)")


def plot_comparison(spmm_us, total_us, k25_us, cublas_us):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'cuSPARSE vs AB-Fused: Sparse Matrix Multiply (4096×4096)',
        fontsize=14, fontweight='bold', y=1.01
    )

    # ------------------------------------------------------------------ #
    #  Left panel: absolute time (ms) grouped bar chart
    # ------------------------------------------------------------------ #
    ax = axes[0]

    n_groups = len(DENSITIES)
    n_bars = 3  # spmm, total, k25
    bar_width = 0.22
    group_gap = 0.08
    group_width = n_bars * bar_width + group_gap
    x = np.arange(n_groups) * group_width

    spmm_ms_vals = [spmm_us[d] / 1000.0 for d in DENSITIES]
    total_ms_vals = [total_us[d] / 1000.0 for d in DENSITIES]
    k25_ms_vals = [k25_us[d] / 1000.0 for d in DENSITIES]
    cublas_ms = cublas_us / 1000.0

    offsets = [-bar_width, 0, bar_width]

    bars_spmm = ax.bar(x + offsets[0], spmm_ms_vals, bar_width,
                       label='cuSPARSE SpMM only', color='steelblue', zorder=3)
    bars_total = ax.bar(x + offsets[1], total_ms_vals, bar_width,
                        label='cuSPARSE total', color='cornflowerblue', zorder=3)
    bars_k25 = ax.bar(x + offsets[2], k25_ms_vals, bar_width,
                      label='AB-Fused \u2605 (K25)', color='red', zorder=3)

    # cuBLAS horizontal dashed line
    ax.axhline(y=cublas_ms, color='gray', linestyle='--', linewidth=2,
               label=f'cuBLAS ({cublas_ms:.1f} ms)', zorder=2)

    # Value labels above bars
    def add_bar_labels(ax, bars, vals):
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + ax.get_ylim()[1] * 0.01,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

    # Set ylim first so label offset calculation is correct
    all_vals = spmm_ms_vals + total_ms_vals + k25_ms_vals
    max_val = max(v for v in all_vals if not np.isnan(v))
    ax.set_ylim(0, max_val * 1.18)

    add_bar_labels(ax, bars_spmm, spmm_ms_vals)
    add_bar_labels(ax, bars_total, total_ms_vals)
    add_bar_labels(ax, bars_k25, k25_ms_vals)

    ax.set_xticks(x)
    ax.set_xticklabels(DENSITY_LABELS, fontsize=12)
    ax.set_xlabel('Matrix Density', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Runtime', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    # ------------------------------------------------------------------ #
    #  Right panel: speedup vs cuBLAS line chart
    # ------------------------------------------------------------------ #
    ax2 = axes[1]

    spmm_speedup = [cublas_ms / v for v in spmm_ms_vals]
    total_speedup = [cublas_ms / v for v in total_ms_vals]
    k25_speedup = [cublas_ms / v for v in k25_ms_vals]

    density_vals = [d * 100.0 for d in DENSITIES]  # for x-axis labels

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                label='cuBLAS (1.0×)', zorder=2)

    ax2.plot(density_vals, spmm_speedup, marker='o', linewidth=2.5, markersize=9,
             label='cuSPARSE SpMM only', color='steelblue', zorder=3)
    ax2.plot(density_vals, total_speedup, marker='s', linewidth=2.5, markersize=9,
             label='cuSPARSE total', color='cornflowerblue', zorder=3)
    ax2.plot(density_vals, k25_speedup, marker='D', linewidth=3.0, markersize=10,
             label='AB-Fused \u2605 (K25)', color='red', zorder=4)

    # Annotate K25 speedup values
    for d_val, sp in zip(density_vals, k25_speedup):
        ax2.annotate(f'{sp:.2f}\u00d7',
                     xy=(d_val, sp),
                     xytext=(6, 6), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='red')

    ax2.set_xlabel('Matrix Density (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax2.set_xticks(density_vals)
    ax2.set_xticklabels(DENSITY_LABELS, fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, zorder=0)

    all_speedups = spmm_speedup + total_speedup + k25_speedup
    max_sp = np.nanmax(all_speedups)
    min_sp = np.nanmin(all_speedups)
    ax2.set_ylim(min(0, min_sp * 0.85), max_sp * 1.2)

    plt.tight_layout()

    out_pdf = OUTPUT_DIR / "figure_cusparse_comparison.pdf"
    out_png = OUTPUT_DIR / "figure_cusparse_comparison.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf, out_png


def main():
    print("Loading cuSPARSE results...")
    cusparse_df = load_cusparse(n=4096)
    print(f"  {len(cusparse_df)} cuSPARSE rows at N=4096")

    print("Loading ESMM kernel results (K25, K15)...")
    esmm_df = load_k25_k15(n=4096)
    print(f"  Kernels present: {sorted(esmm_df['kernel'].unique())}")

    spmm_us, total_us, k25_us, cublas_us = get_series(cusparse_df, esmm_df)

    print_summary(spmm_us, total_us, k25_us, cublas_us)

    print("\nGenerating plot...")
    out_pdf, out_png = plot_comparison(spmm_us, total_us, k25_us, cublas_us)

    pdf_bytes = os.path.getsize(out_pdf)
    png_bytes = os.path.getsize(out_png)
    print(f"\nFile sizes: PDF={pdf_bytes:,} bytes, PNG={png_bytes:,} bytes")


if __name__ == "__main__":
    main()
