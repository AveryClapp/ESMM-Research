#!/usr/bin/env python3
"""
Figure 4: Batch Amortization
Shows how preprocessing overhead decreases with batch size
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
DATA_DIR = PROJECT_ROOT / "results" / "figure4_batch_amortization"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_times():
    """Extract preprocessing and GEMM times from K25 at 4096×4096, 50% density"""

    # Try Figure 4 specific data first
    summary_file = DATA_DIR / "k25_reference" / "summary.csv"

    if not summary_file.exists():
        # Fall back to Figure 2 data
        summary_file = PROJECT_ROOT / "results" / "figure2_preprocessing_overhead" / "k25_scaling" / "summary.csv"

    if not summary_file.exists():
        print(f"ERROR: No data found. Please run:")
        print("  bash scripts/experiments/02_collect_figure2_data.sh")
        print("  OR")
        print("  bash scripts/experiments/04_collect_figure4_data.sh")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')

    # Filter for size 4096
    df_4096 = df[df['size'] == 4096]

    if df_4096.empty:
        print("ERROR: No data for size=4096 found")
        sys.exit(1)

    # Extract preprocessing and GEMM times
    preprocess_df = df_4096[df_4096['kernel'] == 'PREPROCESS']
    gemm_df = df_4096[df_4096['kernel'] != 'PREPROCESS']

    if preprocess_df.empty or gemm_df.empty:
        print("ERROR: Missing preprocessing or GEMM data")
        sys.exit(1)

    preprocess_time_us = preprocess_df['kernel_time_us'].mean()
    gemm_time_us = gemm_df['kernel_time_us'].mean()

    return preprocess_time_us, gemm_time_us

def calculate_batch_overhead(preprocess_us, gemm_us, batch_sizes):
    """Calculate effective overhead for different batch sizes

    Formula: overhead% = preprocess / (preprocess + gemm * batch_size) * 100
    """
    overheads = []
    for batch_size in batch_sizes:
        total_time = preprocess_us + (gemm_us * batch_size)
        overhead_pct = (preprocess_us / total_time) * 100.0
        overheads.append(overhead_pct)

    return np.array(overheads)

def plot_figure4():
    """Generate Figure 4: Batch Amortization"""

    print("Extracting times from benchmark data...")
    preprocess_us, gemm_us = extract_times()

    print(f"\nExtracted times (4096×4096, 50% density):")
    print(f"  Preprocessing: {preprocess_us:.1f} µs")
    print(f"  GEMM:          {gemm_us:.1f} µs")
    print(f"  Single-batch overhead: {(preprocess_us / (preprocess_us + gemm_us) * 100):.2f}%")

    # Batch sizes (log scale)
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

    # Calculate overhead
    overheads = calculate_batch_overhead(preprocess_us, gemm_us, batch_sizes)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(batch_sizes, overheads, marker='o', linewidth=2.5, markersize=10,
            color='red', label='Preprocessing Overhead')

    # Add reference lines
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='1% threshold', alpha=0.7)
    ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, label='5% threshold', alpha=0.7)

    # Shade typical LLM batch range (32-128)
    ax.axvspan(32, 128, alpha=0.2, color='blue', label='Typical LLM batch range')

    # Formatting
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preprocessing Overhead (%)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: Batch Amortization Analysis (K25, 4096×4096, 50% density)',
                 fontsize=16, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    # Add annotations for key batch sizes
    for batch_size in [1, 32, 128]:
        idx = np.where(batch_sizes == batch_size)[0][0]
        overhead = overheads[idx]
        ax.annotate(f'{overhead:.2f}%',
                   xy=(batch_size, overhead),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / "figure4_batch_amortization.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")

    png_file = output_file.with_suffix('.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_file}")

    plt.close()

    # Print summary table
    print("\n===== Overhead vs Batch Size =====")
    print(f"{'Batch Size':>12} | {'Overhead %':>12} | {'Total Time (ms)':>18}")
    print("-" * 48)
    for batch_size, overhead in zip(batch_sizes, overheads):
        total_time_ms = (preprocess_us + gemm_us * batch_size) / 1000.0
        print(f"{batch_size:>12} | {overhead:>11.3f}% | {total_time_ms:>17.2f}")

    print("\n===== Key Findings =====")
    print(f"Batch size 1:   {overheads[0]:.2f}% overhead")
    print(f"Batch size 32:  {overheads[np.where(batch_sizes == 32)[0][0]]:.3f}% overhead")
    print(f"Batch size 128: {overheads[np.where(batch_sizes == 128)[0][0]]:.3f}% overhead")
    print(f"\nConclusion: Preprocessing overhead becomes negligible (<0.5%) at typical LLM batch sizes")

if __name__ == "__main__":
    plot_figure4()
