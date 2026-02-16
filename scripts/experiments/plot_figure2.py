#!/usr/bin/env python3
"""
Figure 2: Preprocessing Overhead Breakdown
Shows preprocessing + GEMM time across matrix sizes, and overhead percentage
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
DATA_DIR = PROJECT_ROOT / "results" / "figure2_preprocessing_overhead"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load K25 preprocessing and GEMM times across matrix sizes"""
    summary_file = DATA_DIR / "k25_scaling" / "summary.csv"

    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found")
        print("Please run: bash scripts/experiments/02_collect_figure2_data.sh")
        sys.exit(1)

    df = pd.read_csv(summary_file)

    # Separate preprocessing and main kernel
    preprocess_df = df[df['kernel'] == 'PREPROCESS'].copy()
    gemm_df = df[df['kernel'] != 'PREPROCESS'].copy()

    # Group by size
    preprocess_times = preprocess_df.groupby('size')['kernel_time_us'].mean()
    gemm_times = gemm_df.groupby('size')['kernel_time_us'].mean()

    # Combine into single dataframe
    result = pd.DataFrame({
        'size': preprocess_times.index,
        'preprocess_us': preprocess_times.values,
        'gemm_us': gemm_times.values
    })

    result['total_us'] = result['preprocess_us'] + result['gemm_us']
    result['overhead_pct'] = (result['preprocess_us'] / result['total_us']) * 100.0

    return result

def plot_figure2():
    """Generate Figure 2: Preprocessing Overhead Breakdown"""

    print("Loading data...")
    df = load_data()

    print(f"\nData shape: {df.shape}")
    print("\nOverhead progression:")
    for _, row in df.iterrows():
        print(f"  Size {row['size']:5d}: {row['overhead_pct']:5.2f}% "
              f"(preprocess: {row['preprocess_us']:7.1f} µs, gemm: {row['gemm_us']:8.1f} µs)")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = df['size'].values
    x_pos = np.arange(len(sizes))

    # Left subplot: Stacked bar chart
    ax1.bar(x_pos, df['preprocess_us'] / 1000, label='Preprocessing', color='orange', alpha=0.8)
    ax1.bar(x_pos, df['gemm_us'] / 1000, bottom=df['preprocess_us'] / 1000,
            label='GEMM', color='steelblue', alpha=0.8)

    ax1.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Breakdown', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sizes)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add percentage annotations on bars
    for i, row in df.iterrows():
        total_ms = row['total_us'] / 1000
        pct = row['overhead_pct']
        ax1.text(i, total_ms * 0.5, f'{pct:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Right subplot: Overhead percentage line plot
    ax2.plot(x_pos, df['overhead_pct'], marker='o', linewidth=2.5, markersize=10,
             color='red', label='Overhead %')
    ax2.axhline(y=5.0, color='gray', linestyle='--', linewidth=1.5, label='5% threshold')

    ax2.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Preprocessing Overhead (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overhead Percentage', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sizes)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(df['overhead_pct'].max() * 1.1, 25))

    # Add annotations for specific points
    for i, row in df.iterrows():
        ax2.annotate(f"{row['overhead_pct']:.1f}%",
                    xy=(i, row['overhead_pct']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Figure 2: Preprocessing Overhead Analysis (K25, 50% density)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / "figure2_preprocessing_overhead.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    png_file = output_file.with_suffix('.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_file}")

    plt.close()

    # Print key findings
    print("\n===== Key Findings =====")
    smallest = df.iloc[0]
    largest = df.iloc[-1]
    print(f"Overhead at {smallest['size']}×{smallest['size']}: {smallest['overhead_pct']:.2f}%")
    print(f"Overhead at {largest['size']}×{largest['size']}: {largest['overhead_pct']:.2f}%")
    print(f"Reduction: {smallest['overhead_pct'] / largest['overhead_pct']:.1f}× lower at largest size")

if __name__ == "__main__":
    plot_figure2()
