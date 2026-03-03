#!/usr/bin/env python3
"""
Figure 1: Performance vs Sparsity
Main result showing speedup over cuBLAS for AB-Separate (K20), AB-Fine (K21), AB-Fused (K25).
cuBLAS baseline is measured via NCU (K15) for methodology consistency.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN_TO_DENSITY = {
    "00000000": 0.0,
    "10000000": 12.5,
    "11000000": 25.0,
    "11100000": 37.5,
    "11110000": 50.0,
    "11111000": 62.5,
    "11111100": 75.0,
    "11111110": 87.5,
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
    """Load all kernel rows from summary.csv (PREPROCESS rows excluded)."""
    summary_file = DATA_DIR / "esmm_kernels" / "summary.csv"
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
    return df

def get_cublas_time(df):
    """Return cuBLAS time from K15 (NCU-profiled) rows, or fallback."""
    k15 = df[df['kernel'] == 15]
    if not k15.empty:
        t = float(k15['kernel_time_us'].mean())
        print(f"cuBLAS time from K15 (NCU-profiled): {t:.1f} µs")
        return t
    print("WARNING: K15 not in data, using fallback 7210 µs")
    return 7210.0

def plot_figure1():
    print("Loading data...")
    df = load_data()
    cublas_time = get_cublas_time(df)

    # Only K20, K21, K25 in the figure (K17 dropped, K15 used only for reference)
    esmm_df = df[df['kernel'].isin([20, 21, 25])].copy()
    print(f"ESMM kernels: {sorted(esmm_df['kernel'].unique())}")
    print(f"cuBLAS reference: {cublas_time:.1f} µs")

    esmm_df = esmm_df.groupby(['kernel', 'density'])['kernel_time_us'].mean().reset_index()
    esmm_df['speedup'] = cublas_time / esmm_df['kernel_time_us']

    fig, (ax_speedup, ax_abs) = plt.subplots(1, 2, figsize=(14, 5))

    kernels = {
        20: {'label': 'AB-Separate',  'marker': 's', 'color': 'purple'},
        21: {'label': 'AB-Fine',      'marker': 'o', 'color': 'blue'},
        25: {'label': 'AB-Fused ★',   'marker': 'D', 'color': 'red'},
    }

    # ---- Left subplot: speedup vs density ----
    ax_speedup.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                       label='cuBLAS (dense baseline)', zorder=1)

    for kernel_num, style in kernels.items():
        kdata = esmm_df[esmm_df['kernel'] == kernel_num].sort_values('density')
        if not kdata.empty:
            lw = 3.0 if kernel_num == 25 else 2.5
            ax_speedup.plot(kdata['density'], kdata['speedup'],
                            marker=style['marker'], linewidth=lw, markersize=9,
                            label=style['label'], color=style['color'])

    ax_speedup.set_xlabel('Matrix Density (%)', fontsize=14, fontweight='bold')
    ax_speedup.set_ylabel('Speedup vs cuBLAS', fontsize=14, fontweight='bold')
    ax_speedup.set_title('Speedup vs Density', fontsize=13, fontweight='bold')
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax_speedup.set_xlim(-5, 105)

    max_speedup = esmm_df['speedup'].max()
    ax_speedup.set_ylim(0, max(max_speedup * 1.15, 2.5))

    # Annotate AB-Fused at 25% density (regime where sparsity exploitation clearly pays off)
    k25 = esmm_df[esmm_df['kernel'] == 25]
    k25_25 = k25[k25['density'] == 25.0]
    if not k25_25.empty:
        row = k25_25.iloc[0]
        ax_speedup.annotate(f"{row['speedup']:.2f}× at 25%",
                            xy=(row['density'], row['speedup']),
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # ---- Right subplot: absolute runtime vs density ----
    ax_abs.axhline(y=cublas_time / 1000, color='gray', linestyle='--', linewidth=2,
                   label=f"cuBLAS ({cublas_time/1000:.1f} ms)", zorder=1)

    for kernel_num, style in kernels.items():
        kdata = esmm_df[esmm_df['kernel'] == kernel_num].sort_values('density')
        if not kdata.empty:
            lw = 3.0 if kernel_num == 25 else 2.5
            ax_abs.plot(kdata['density'], kdata['kernel_time_us'] / 1000,
                        marker=style['marker'], linewidth=lw, markersize=9,
                        label=style['label'], color=style['color'])

    ax_abs.set_xlabel('Matrix Density (%)', fontsize=14, fontweight='bold')
    ax_abs.set_ylabel('Kernel Time (ms)', fontsize=14, fontweight='bold')
    ax_abs.set_title('Absolute Runtime vs Density', fontsize=13, fontweight='bold')
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax_abs.set_xlim(-5, 105)

    fig.suptitle('Figure 1: Performance vs Sparsity (4096×4096, blockwise)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    out = OUTPUT_DIR / "figure1_performance_vs_sparsity.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out}")

    print("\n===== Summary =====")
    print(f"cuBLAS baseline: {cublas_time:.1f} µs")
    for kn, name in [(20, 'AB-Separate'), (21, 'AB-Fine'), (25, 'AB-Fused ★')]:
        kdata = esmm_df[esmm_df['kernel'] == kn]
        for d in [12.5, 25.0, 37.5, 50.0]:
            kd = kdata[kdata['density'] == d]
            s = kd['speedup'].values[0] if not kd.empty else float('nan')
            tag = " ★ MAIN" if kn == 25 else ""
            print(f"{name}: speedup at {d:.1f}% density = {s:.2f}×{tag}")

if __name__ == "__main__":
    plot_figure1()
