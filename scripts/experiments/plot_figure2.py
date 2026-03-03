#!/usr/bin/env python3
"""
Figure 2: Preprocessing Overhead Breakdown
Shows preprocessing + GEMM time across matrix sizes, and overhead percentage
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "figure2_preprocessing_overhead"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    summary_file = DATA_DIR / "k25_scaling" / "summary.csv"
    if not summary_file.exists():
        print(f"ERROR: {summary_file} not found")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()

    preprocess_df = df[df['kernel'] == 'PREPROCESS'].copy()
    gemm_df = df[df['kernel'] != 'PREPROCESS'].copy()

    preprocess_times = preprocess_df.groupby('size')['kernel_time_us'].mean()
    gemm_times = gemm_df.groupby('size')['kernel_time_us'].mean()

    result = pd.DataFrame({
        'size': preprocess_times.index.astype(int),
        'preprocess_us': preprocess_times.values,
        'gemm_us': gemm_times.values,
    })
    result = result.sort_values('size').reset_index(drop=True)

    result['total_us'] = result['preprocess_us'] + result['gemm_us']
    result['overhead_pct'] = (result['preprocess_us'] / result['total_us']) * 100.0
    return result

def plot_figure2():
    print("Loading data...")
    df = load_data()

    print(f"\nData: {df.shape[0]} sizes")
    print("\nOverhead progression:")
    for _, row in df.iterrows():
        print(f"  Size {int(row['size']):5d}: {row['overhead_pct']:5.2f}% "
              f"(preprocess: {row['preprocess_us']:7.1f} µs, gemm: {row['gemm_us']:8.1f} µs)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = df['size'].values.astype(int)
    x_pos = np.arange(len(sizes))
    size_labels = [f"{s:,}" for s in sizes]

    # Left: stacked bar
    b1 = ax1.bar(x_pos, df['preprocess_us'] / 1000, label='Preprocessing',
                 color='orange', alpha=0.85, edgecolor='black', linewidth=0.8)
    b2 = ax1.bar(x_pos, df['gemm_us'] / 1000, bottom=df['preprocess_us'] / 1000,
                 label='GEMM Compute', color='steelblue', alpha=0.85,
                 edgecolor='black', linewidth=0.8)

    ax1.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Kernel Time Breakdown', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(size_labels)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Percentage annotations in GEMM bar (show overhead %)
    for i, row in df.iterrows():
        pct = row['overhead_pct']
        total_ms = row['total_us'] / 1000
        # Label in orange segment (preprocessing)
        preprocess_ms = row['preprocess_us'] / 1000
        ax1.text(i, preprocess_ms / 2, f'{pct:.1f}%',
                 ha='center', va='center', fontsize=9, fontweight='bold', color='black')

    # Right: overhead % line
    ax2.plot(x_pos, df['overhead_pct'], marker='o', linewidth=2.5, markersize=10,
             color='darkorange', label='Preprocessing Overhead')
    ax2.axhline(y=5.0, color='gray', linestyle='--', linewidth=1.5,
                label='5% threshold', alpha=0.8)
    ax2.axhline(y=1.0, color='green', linestyle=':', linewidth=1.5,
                label='1% threshold', alpha=0.8)

    ax2.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Preprocessing Overhead (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overhead vs Matrix Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(size_labels)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(df['overhead_pct'].max() * 1.2, 10))

    for i, row in df.iterrows():
        ax2.annotate(f"{row['overhead_pct']:.1f}%",
                     xy=(i, row['overhead_pct']),
                     xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Figure 2: K25 Preprocessing Overhead (50% density)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "figure2_preprocessing_overhead.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out}")

    print("\n===== Key Findings =====")
    smallest = df.iloc[0]
    largest = df.iloc[-1]
    print(f"Overhead at {int(smallest['size'])}×{int(smallest['size'])}: {smallest['overhead_pct']:.2f}%")
    print(f"Overhead at {int(largest['size'])}×{int(largest['size'])}: {largest['overhead_pct']:.2f}%")
    ratio = smallest['overhead_pct'] / largest['overhead_pct']
    print(f"Overhead reduces {ratio:.1f}× from smallest to largest size")

if __name__ == "__main__":
    plot_figure2()
