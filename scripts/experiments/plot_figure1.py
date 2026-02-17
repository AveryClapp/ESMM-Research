#!/usr/bin/env python3
"""
Figure 1: Performance vs Sparsity
Main result showing speedup over cuBLAS for K17, K24, K28, plus cuSPARSE baseline
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sparsity pattern to percentage mapping
PATTERN_TO_SPARSITY = {
    "00000000": 0.0,    # 0% density = 100% sparsity
    "10000000": 12.5,   # 12.5% density
    "11000000": 25.0,
    "11100000": 37.5,
    "11110000": 50.0,
    "11111000": 62.5,
    "11111100": 75.0,
    "11111110": 87.5,
    "11111111": 100.0,  # 100% density
}

def pattern_to_density(pattern):
    """Convert 8-bit pattern to density percentage"""
    if pattern in PATTERN_TO_SPARSITY:
        return PATTERN_TO_SPARSITY[pattern]
    # Fallback: count 1s
    ones = pattern.count('1')
    return (ones / 8.0) * 100.0

def load_esmm_data():
    """Load K17, K24, K28 data from benchmark results"""
    summary_file = DATA_DIR / "esmm_kernels" / "summary.csv"

    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found")
        print("Please run: bash scripts/experiments/01_collect_figure1_data.sh")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')

    # Filter out preprocessing kernels
    df = df[df['kernel'] != 'PREPROCESS'].copy()

    # Extract kernel number from kernel_name (e.g., "esmm_ab_8x32" -> 28)
    # Simplify: just use the 'kernel' column which should have the number

    # Add density column
    df['density'] = df['pattern'].apply(pattern_to_density)

    # Group by kernel and density, take mean of kernel_time_us
    result = df.groupby(['kernel', 'density'])['kernel_time_us'].mean().reset_index()

    return result

def load_cublas_data():
    """Load cuBLAS baseline (constant across sparsity levels)"""
    # Try dedicated cuBLAS script output first
    cublas_file = DATA_DIR / "cublas_baseline.csv"

    if cublas_file.exists():
        df = pd.read_csv(cublas_file, comment='#')
        # Assume format: size, time_us
        time_us = df[df['size'] == 4096]['time_us'].mean()
    else:
        # Fallback: use K10 dense reference
        summary_file = DATA_DIR / "cublas_reference" / "summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file, comment='#')
            df = df[df['kernel'] != 'PREPROCESS']
            time_us = df['kernel_time_us'].mean()
        else:
            print("WARNING: cuBLAS baseline not found, using placeholder value")
            time_us = 5000.0  # Placeholder

    return time_us

def load_cusparse_data():
    """Load cuSPARSE baseline (varies with sparsity)"""
    cusparse_file = DATA_DIR / "cusparse_baseline.csv"

    if not cusparse_file.exists():
        print("WARNING: cuSPARSE baseline not found, skipping")
        return None

    df = pd.read_csv(cusparse_file, comment='#')
    # Assume format: sparsity (0-1), time_us (including conversion)
    df['density'] = (1.0 - df['sparsity']) * 100.0
    return df[['density', 'time_us']]

def plot_figure1():
    """Generate Figure 1: Performance vs Sparsity"""

    print("Loading data...")
    esmm_df = load_esmm_data()
    cublas_time = load_cublas_data()
    cusparse_df = load_cusparse_data()

    print(f"\nESMM data shape: {esmm_df.shape}")
    print(f"cuBLAS time: {cublas_time:.1f} µs")

    # Calculate speedups
    esmm_df['speedup'] = cublas_time / esmm_df['kernel_time_us']

    if cusparse_df is not None:
        cusparse_df['speedup'] = cublas_time / cusparse_df['time_us']

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot cuBLAS baseline (horizontal line at 1.0)
    density_range = np.linspace(0, 100, 100)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='cuBLAS (dense)', zorder=1)

    # Plot cuSPARSE if available
    if cusparse_df is not None:
        cusparse_sorted = cusparse_df.sort_values('density')
        ax.plot(cusparse_sorted['density'], cusparse_sorted['speedup'],
                marker='s', linewidth=2, markersize=8, label='cuSPARSE (CSR)', color='orange')

    # Plot K17, K20, K21, K25
    kernels = {
        17: {'label': 'K17 (B-sparse only)', 'marker': '^', 'color': 'green'},
        20: {'label': 'K20 (A+B, separate preprocess)', 'marker': 's', 'color': 'purple'},
        21: {'label': 'K21 (A+B, 8×32, separate)', 'marker': 'o', 'color': 'blue'},
        25: {'label': 'K25 (A+B, fused) ⭐', 'marker': 'D', 'color': 'red'},
    }

    for kernel_num, style in kernels.items():
        kernel_data = esmm_df[esmm_df['kernel'] == kernel_num].sort_values('density')
        if not kernel_data.empty:
            ax.plot(kernel_data['density'], kernel_data['speedup'],
                    marker=style['marker'], linewidth=2.5, markersize=9,
                    label=style['label'], color=style['color'])

    # Formatting
    ax.set_xlabel('Matrix Density (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs cuBLAS', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1: Performance vs Sparsity (4096×4096)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, max(esmm_df['speedup'].max() * 1.1, 2.0))

    # Add annotations for peak speedups (K25 and K21)
    for kernel_num in [21, 25]:
        kernel_data = esmm_df[esmm_df['kernel'] == kernel_num]
        if not kernel_data.empty:
            max_speedup = kernel_data['speedup'].max()
            max_density = kernel_data.loc[kernel_data['speedup'].idxmax(), 'density']
            color = 'yellow' if kernel_num == 25 else 'lightblue'
            ax.annotate(f'{max_speedup:.2f}×',
                       xy=(max_density, max_speedup),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / "figure1_performance_vs_sparsity.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    # Also save as PNG for preview
    png_file = output_file.with_suffix('.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_file}")

    plt.close()

    # Print summary statistics
    print("\n===== Summary Statistics =====")
    print(f"cuBLAS baseline: {cublas_time:.1f} µs (1.00× by definition)")

    for kernel_num in [17, 20, 21, 25]:
        kernel_data = esmm_df[esmm_df['kernel'] == kernel_num]
        if not kernel_data.empty:
            max_speedup = kernel_data['speedup'].max()
            max_density = kernel_data.loc[kernel_data['speedup'].idxmax(), 'density']
            marker = " ⭐ MAIN KERNEL" if kernel_num == 25 else ""
            print(f"K{kernel_num}: Peak speedup {max_speedup:.2f}× at {max_density:.0f}% density{marker}")

if __name__ == "__main__":
    plot_figure1()
