#!/usr/bin/env python3
"""
Figure 5: Matrix Size Scaling
Shows how AB-Fused (K25) and cuBLAS scale across matrix sizes at 50% density.
cuBLAS baseline measured via NCU (K15) for methodology consistency.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "figure5_matrix_scaling"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_gflops(size, time_us):
    """GFLOPS for square matrix multiply: 2*N^3 FLOPs"""
    flops = 2.0 * size ** 3
    return (flops / time_us) / 1e6  # µs → s, FLOPs → GFLOPS

def load_data():
    """Load all kernel rows from summary.csv (PREPROCESS rows excluded)."""
    summary_file = DATA_DIR / "esmm_kernels" / "summary.csv"
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel'])
    df['kernel'] = df['kernel'].astype(int)

    result = df.groupby(['kernel', 'size'])['kernel_time_us'].mean().reset_index()
    result['gflops'] = result.apply(
        lambda r: calculate_gflops(r['size'], r['kernel_time_us']), axis=1)
    return result

def get_cublas_df(df, esmm_sizes):
    """Extract cuBLAS data from K15 rows, or fall back to O(N^3) scaling."""
    k15 = df[df['kernel'] == 15].copy()
    if not k15.empty:
        k15 = k15[['size', 'kernel_time_us', 'gflops']].copy()
        k15['kernel'] = 'cuBLAS'
        print(f"cuBLAS from K15 (NCU-profiled):")
        for _, row in k15.iterrows():
            print(f"  {int(row['size'])}×{int(row['size'])}: {row['kernel_time_us']:.1f} µs")
        return k15[['kernel', 'size', 'kernel_time_us', 'gflops']]

    # Fallback: O(N^3) scaling from known 4096 measurement
    print("WARNING: K15 not in data, scaling cuBLAS O(N^3) from 7210 µs at 4096")
    ref_time, ref_size = 7210.0, 4096
    sizes = sorted(esmm_sizes)
    times = [ref_time * (s / ref_size) ** 3 for s in sizes]
    result = pd.DataFrame({'kernel': 'cuBLAS', 'size': sizes, 'kernel_time_us': times})
    result['gflops'] = result.apply(
        lambda r: calculate_gflops(r['size'], r['kernel_time_us']), axis=1)
    return result

def plot_figure5():
    print("Loading data...")
    df = load_data()
    print(f"Data: {df.shape[0]} rows, kernels: {sorted(df['kernel'].unique())}")

    esmm_sizes = df[df['kernel'] != 15]['size'].unique()
    cublas_df = get_cublas_df(df, esmm_sizes)

    # Only K25 from ESMM kernels
    esmm_df = df[df['kernel'] == 25].copy()
    all_df = pd.concat([esmm_df, cublas_df], ignore_index=True)

    cublas_times = cublas_df.set_index('size')['kernel_time_us']

    def calc_speedup(row):
        if row['kernel'] == 'cuBLAS':
            return 1.0
        size = row['size']
        if size in cublas_times.index:
            return cublas_times[size] / row['kernel_time_us']
        return np.nan

    all_df['speedup'] = all_df.apply(calc_speedup, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kernels_to_plot = {
        'cuBLAS': {'label': 'cuBLAS (dense)',  'marker': 's', 'color': 'gray',  'lw': 2.0},
        25:       {'label': 'AB-Fused ★',      'marker': 'D', 'color': 'red',   'lw': 3.0},
    }

    sizes = sorted(all_df['size'].unique())

    for kernel, style in kernels_to_plot.items():
        kdata = all_df[all_df['kernel'] == kernel].sort_values('size')
        if not kdata.empty:
            ax1.plot(kdata['size'], kdata['speedup'],
                     marker=style['marker'], linewidth=style['lw'], markersize=9,
                     label=style['label'], color=style['color'])
            ax2.plot(kdata['size'], kdata['gflops'],
                     marker=style['marker'], linewidth=style['lw'], markersize=9,
                     label=style['label'], color=style['color'])

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup Across Matrix Sizes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f"{s:,}" for s in sizes], rotation=20)

    ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f"{s:,}" for s in sizes], rotation=20)

    fig.suptitle('Figure 5: Matrix Size Scaling (AB-Fused vs cuBLAS, 25% density)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "figure5_matrix_scaling.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out}")

    print("\n===== Performance Summary =====")
    print(f"{'Size':>6} | {'cuBLAS (ms)':>12} | {'AB-Fused (ms)':>14} | {'Speedup':>8}")
    print("-" * 50)
    for size in sizes:
        cb = all_df[(all_df['kernel'] == 'cuBLAS') & (all_df['size'] == size)]
        k25 = all_df[(all_df['kernel'] == 25) & (all_df['size'] == size)]
        if not cb.empty and not k25.empty:
            print(f"{size:>6} | {cb['kernel_time_us'].values[0]/1000:>11.2f} | "
                  f"{k25['kernel_time_us'].values[0]/1000:>13.2f} | "
                  f"{k25['speedup'].values[0]:>7.2f}×")

    k25_speedups = all_df[all_df['kernel'] == 25]['speedup'].dropna()
    if len(k25_speedups) > 0:
        print(f"\nAB-Fused avg speedup: {k25_speedups.mean():.2f}× "
              f"(range: {k25_speedups.min():.2f}× – {k25_speedups.max():.2f}×)")

if __name__ == "__main__":
    plot_figure5()
