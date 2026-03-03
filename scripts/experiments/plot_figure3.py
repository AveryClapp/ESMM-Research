#!/usr/bin/env python3
"""
Figure 3: Fusion vs Granularity Tradeoff
Compares AB-Separate (K20), AB-Fine (K21), AB-Fused (K25).
Shows that fusion matters more than granularity.
cuBLAS baseline loaded from K15 (NCU-profiled) in Figure 1 data for consistency.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "figure3_granularity_tradeoff"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_TO_DENSITY = {
    "00000000": 0.0, "10000000": 12.5, "11000000": 25.0, "11100000": 37.5,
    "11110000": 50.0, "11111000": 62.5, "11111100": 75.0, "11111110": 87.5,
    "11111111": 100.0,
}

def pattern_to_density(pattern):
    if pd.isna(pattern):
        return np.nan
    pattern = str(pattern).strip()
    if pattern in PATTERN_TO_DENSITY:
        return PATTERN_TO_DENSITY[pattern]
    try:
        return (pattern.count('1') / 8.0) * 100.0
    except Exception:
        return np.nan

def load_data():
    summary_file = DATA_DIR / "fusion_vs_granularity" / "summary.csv"
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()

    df['pattern'] = df['pattern'].astype(str).str.strip()
    df['density'] = df['pattern'].apply(pattern_to_density)
    df = df.dropna(subset=['density'])

    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel'])
    df['kernel'] = df['kernel'].astype(int)

    result = df.groupby(['kernel', 'density'])['kernel_time_us'].mean().reset_index()
    return result

def plot_figure3():
    print("Loading data...")
    df = load_data()
    print(f"Data: {df.shape[0]} rows, kernels: {sorted(df['kernel'].unique())}")

    # Use cuBLAS time from K15 (NCU-profiled) in Figure 1 data for methodology consistency
    fig1_summary = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity" / "esmm_kernels" / "summary.csv"
    cublas_time = 7210.0  # fallback
    if fig1_summary.exists():
        cdf = pd.read_csv(fig1_summary, comment='#')
        cdf['kernel'] = cdf['kernel'].astype(str).str.strip()
        k15 = cdf[cdf['kernel'] == '15']
        if not k15.empty:
            cublas_time = float(k15['kernel_time_us'].mean())
            print(f"cuBLAS time from K15 (NCU-profiled): {cublas_time:.1f} µs")
        else:
            print(f"WARNING: K15 not found in Figure 1 data, using fallback {cublas_time} µs")
    else:
        print(f"WARNING: Figure 1 summary not found, using fallback {cublas_time} µs")

    print(f"cuBLAS reference: {cublas_time:.1f} µs")

    df['speedup'] = cublas_time / df['kernel_time_us']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    k20_data = df[df['kernel'] == 20].sort_values('density')
    k21_data = df[df['kernel'] == 21].sort_values('density')
    k25_data = df[df['kernel'] == 25].sort_values('density')

    # Left: speedup vs density
    if not k20_data.empty:
        ax1.plot(k20_data['density'], k20_data['speedup'],
                 marker='s', linewidth=2.5, markersize=9,
                 label='AB-Separate', color='purple')
    if not k21_data.empty:
        ax1.plot(k21_data['density'], k21_data['speedup'],
                 marker='o', linewidth=2.5, markersize=9,
                 label='AB-Fine', color='blue')
    if not k25_data.empty:
        ax1.plot(k25_data['density'], k25_data['speedup'],
                 marker='D', linewidth=3.0, markersize=9,
                 label='AB-Fused ★', color='red')

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label='cuBLAS baseline')
    ax1.set_xlabel('Matrix Density (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup vs Density', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)

    if not k25_data.empty:
        k25_avg = k25_data['speedup'].mean()
        k21_avg = k21_data['speedup'].mean() if not k21_data.empty else None
        if k21_avg:
            improvement = (k25_avg / k21_avg - 1) * 100
            print(f"\nK25 avg speedup: {k25_avg:.2f}×, K21: {k21_avg:.2f}× → K25 is {improvement:.1f}% faster")

    # Right: bar chart at 50% density
    density_50 = 50.0
    k20_t = k20_data[k20_data['density'] == density_50]['kernel_time_us'].values
    k21_t = k21_data[k21_data['density'] == density_50]['kernel_time_us'].values
    k25_t = k25_data[k25_data['density'] == density_50]['kernel_time_us'].values

    if len(k20_t) > 0 and len(k21_t) > 0 and len(k25_t) > 0:
        labels = ['AB-Separate\n(64-row)', 'AB-Fine\n(8×32)', 'AB-Fused ★\n(64×32)']
        times_ms = [k20_t[0] / 1000, k21_t[0] / 1000, k25_t[0] / 1000]
        colors = ['purple', 'blue', 'red']

        bars = ax2.bar(labels, times_ms, color=colors, alpha=0.75,
                       edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Kernel Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Absolute Time at 50% Density\n(4096x4096)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, t_ms in zip(bars, times_ms):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                     f'{t_ms:.1f} ms',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        speedup_vs_k21 = k21_t[0] / k25_t[0]
        ax2.text(2, times_ms[2] * 0.5, f'{speedup_vs_k21:.2f}x\nfaster\nvs AB-Fine',
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

        # cuBLAS reference line
        ax2.axhline(y=cublas_time / 1000, color='gray', linestyle='--',
                    linewidth=2, label=f'cuBLAS ({cublas_time/1000:.1f} ms)')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data at 50% density',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=12)

    fig.suptitle('Figure 3: Fusion > Granularity (AB-Separate / AB-Fine / AB-Fused, 4096×4096)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "figure3_granularity_tradeoff.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out}")

    print("\n===== Key Findings =====")
    for kn, kdata in [(20, k20_data), (21, k21_data), (25, k25_data)]:
        if not kdata.empty:
            peak = kdata['speedup'].max()
            peak_d = kdata.loc[kdata['speedup'].idxmax(), 'density']
            tag = " ★ MAIN" if kn == 25 else ""
            print(f"K{kn}: peak {peak:.2f}x at {peak_d:.0f}% density{tag}")

    print("\nKey insight: Fusion (K25) beats fine granularity (K21) despite coarser patterns")

if __name__ == "__main__":
    plot_figure3()
