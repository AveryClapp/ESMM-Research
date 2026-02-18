#!/usr/bin/env python3
"""
Roofline Model: K25 vs hardware limits (A10G)
Analytical arithmetic intensity + timing from existing summary CSVs.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A10G hardware limits
PEAK_GFLOPS    = 31_240.0   # FP32 GFLOPS
PEAK_BW_GBs    = 600.0      # GB/s
RIDGE_POINT    = PEAK_GFLOPS / PEAK_BW_GBs  # ≈ 52.1 FLOP/byte

PATTERN_TO_DENSITY = {
    "00000000": 0.0, "10000000": 12.5, "11000000": 25.0, "11100000": 37.5,
    "11110000": 50.0, "11111000": 62.5, "11111100": 75.0, "11111110": 87.5,
    "11111111": 100.0,
}

def pattern_to_density(pattern):
    p = str(pattern).strip()
    if p in PATTERN_TO_DENSITY:
        return PATTERN_TO_DENSITY[p]
    return (p.count('1') / 8.0) * 100.0

def arithmetic_intensity(n, density_pct):
    """AI = N * density / 6  [FLOP/byte]"""
    return n * (density_pct / 100.0) / 6.0

def effective_gflops(n, density_pct, time_us):
    """Effective GFLOPS: only count FLOPs actually performed."""
    flops = 2.0 * n**3 * (density_pct / 100.0)
    return (flops / time_us) / 1e3  # µs → s, FLOPs → GFLOPS

def load_fig1_data():
    """K25 + K15 at N=4096, varying density."""
    f = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity" / "esmm_kernels" / "summary.csv"
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel']).copy()
    df['kernel'] = df['kernel'].astype(int)
    df['density'] = df['pattern'].astype(str).str.strip().apply(pattern_to_density)
    return df

def load_fig5_data():
    """K25 + K15 at 50% density, varying N."""
    f = PROJECT_ROOT / "results" / "figure5_matrix_scaling" / "esmm_kernels" / "summary.csv"
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel']).copy()
    df['kernel'] = df['kernel'].astype(int)
    df['density'] = 50.0  # fig5 is fixed at 50% density (pattern 11110000)
    return df

def plot_roofline():
    df1 = load_fig1_data()
    df5 = load_fig5_data()

    # K25 points: densities at N=4096 (from fig1)
    k25_fig1 = df1[(df1['kernel'] == 25) & (df1['density'] > 0)].copy()
    k25_fig1 = k25_fig1.groupby('density')['kernel_time_us'].mean().reset_index()
    k25_fig1['n'] = 4096
    k25_fig1['ai']    = k25_fig1.apply(lambda r: arithmetic_intensity(r['n'], r['density']), axis=1)
    k25_fig1['gflops'] = k25_fig1.apply(lambda r: effective_gflops(r['n'], r['density'], r['kernel_time_us']), axis=1)

    # K25 points: sizes at 50% density (from fig5)
    k25_fig5 = df5[df5['kernel'] == 25].copy()
    k25_fig5 = k25_fig5.groupby('size')['kernel_time_us'].mean().reset_index()
    k25_fig5['density'] = 50.0
    k25_fig5['ai']    = k25_fig5.apply(lambda r: arithmetic_intensity(r['size'], r['density']), axis=1)
    k25_fig5['gflops'] = k25_fig5.apply(lambda r: effective_gflops(r['size'], r['density'], r['kernel_time_us']), axis=1)
    # Remove N=4096 duplicate (already in k25_fig1)
    k25_fig5 = k25_fig5[k25_fig5['size'] != 4096]

    # cuBLAS (K15) points from fig5 (dense: density=100%)
    k15 = df5[df5['kernel'] == 15].copy()
    k15 = k15.groupby('size')['kernel_time_us'].mean().reset_index()
    k15['density'] = 100.0
    k15['ai']    = k15.apply(lambda r: arithmetic_intensity(r['size'], r['density']), axis=1)
    k15['gflops'] = k15.apply(lambda r: effective_gflops(r['size'], r['density'], r['kernel_time_us']), axis=1)

    # Roofline curve
    ai_range = np.logspace(-1, 4, 500)
    roofline  = np.minimum(PEAK_GFLOPS, PEAK_BW_GBs * ai_range)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline ceilings
    ax.plot(ai_range, roofline, 'k-', linewidth=2.5, label='Roofline (A10G)', zorder=2)
    ax.axhline(PEAK_GFLOPS, color='black', linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(RIDGE_POINT, color='black', linestyle='--', linewidth=1, alpha=0.4,
               label=f'Ridge point ({RIDGE_POINT:.0f} FLOP/byte)')

    # Shade memory-bound / compute-bound regions
    ax.fill_betweenx([0, PEAK_GFLOPS], 0.1, RIDGE_POINT, alpha=0.05, color='blue')
    ax.fill_betweenx([0, PEAK_GFLOPS], RIDGE_POINT, 10000, alpha=0.05, color='red')

    # K25 points: varying density at N=4096
    sc1 = ax.scatter(k25_fig1['ai'], k25_fig1['gflops'],
                     c=k25_fig1['density'], cmap='RdYlGn_r',
                     s=120, marker='D', zorder=5,
                     vmin=0, vmax=100, label='AB-Fused (N=4096, vary density)')
    for _, row in k25_fig1.iterrows():
        ax.annotate(f"{row['density']:.0f}%",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # K25 points: varying size at 50% density
    ax.scatter(k25_fig5['ai'], k25_fig5['gflops'],
               color='red', s=100, marker='D', alpha=0.6, zorder=5,
               label='AB-Fused (50% density, vary N)')
    for _, row in k25_fig5.iterrows():
        ax.annotate(f"N={int(row['size'])}",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, -12), textcoords='offset points', fontsize=8)

    # cuBLAS reference points
    ax.scatter(k15['ai'], k15['gflops'],
               color='gray', s=100, marker='s', zorder=5,
               label='cuBLAS (dense, vary N)')
    for _, row in k15.iterrows():
        ax.annotate(f"N={int(row['size'])}",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color='gray')

    # Colorbar for density
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Matrix Density (%)', fontsize=11)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(10, PEAK_GFLOPS * 2)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Effective Performance (GFLOPS)', fontsize=13, fontweight='bold')
    ax.set_title('Roofline Model: AB-Fused (K25) on A10G\n(Analytical AI, NCU-timed execution)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.2)

    # Annotations for regions
    ax.text(2, 200, 'Memory-bound', fontsize=10, color='blue', alpha=0.6, style='italic')
    ax.text(200, 200, 'Compute-bound', fontsize=10, color='red', alpha=0.6, style='italic')

    plt.tight_layout()
    out = OUTPUT_DIR / "roofline.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")

    # Summary table
    print("\n===== Roofline Summary =====")
    print(f"Peak FP32:    {PEAK_GFLOPS:,.0f} GFLOPS")
    print(f"Peak BW:      {PEAK_BW_GBs:.0f} GB/s")
    print(f"Ridge point:  {RIDGE_POINT:.1f} FLOP/byte")
    print(f"\n{'Config':30s} | {'AI':>10} | {'GFLOPS':>10} | {'% of Peak':>10}")
    print("-" * 68)
    for _, row in k25_fig1.sort_values('density').iterrows():
        pct = row['gflops'] / PEAK_GFLOPS * 100
        print(f"N=4096, density={row['density']:5.1f}% | {row['ai']:>10.1f} | {row['gflops']:>10.1f} | {pct:>9.1f}%")

if __name__ == "__main__":
    plot_roofline()
