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
    """Extract A preprocessing, B preprocessing, and GEMM times from K25 at 4096×4096.

    PREPROCESS rows are separated by kernel_name:
      - rows containing 'preprocess_a' → A preprocessing (non-amortizable: activations change each call)
      - rows containing 'preprocess_b' → B preprocessing (amortizable: weights are static)
    """
    summary_file = DATA_DIR / "k25_reference" / "summary.csv"

    if not summary_file.exists():
        summary_file = PROJECT_ROOT / "results" / "figure2_preprocessing_overhead" / "k25_scaling" / "summary.csv"

    if not summary_file.exists():
        print("ERROR: No data found. Please run:")
        print("  bash scripts/experiments/02_collect_figure2_data.sh")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')

    df_4096 = df[df['size'] == 4096]
    if df_4096.empty:
        print("ERROR: No data for size=4096 found")
        sys.exit(1)

    preprocess_df = df_4096[df_4096['kernel'] == 'PREPROCESS'].copy()
    gemm_df = df_4096[df_4096['kernel'] != 'PREPROCESS'].copy()

    if preprocess_df.empty or gemm_df.empty:
        print("ERROR: Missing preprocessing or GEMM data")
        sys.exit(1)

    # Split by kernel_name to separate A (non-amortizable) from B (amortizable)
    a_rows = preprocess_df[preprocess_df['kernel_name'].str.contains('preprocess_a', na=False)]
    b_rows = preprocess_df[preprocess_df['kernel_name'].str.contains('preprocess_b', na=False)]

    if a_rows.empty or b_rows.empty:
        # Fallback: split combined preprocess time using measured ratio from profiling
        combined = preprocess_df['kernel_time_us'].sum()
        a_preprocess_us = combined * 0.551  # A is ~55% of total (167/(167+136))
        b_preprocess_us = combined * 0.449  # B is ~45% of total
        print("WARNING: Could not split A/B preprocessing by kernel_name; using ratio split")
    else:
        a_preprocess_us = a_rows['kernel_time_us'].mean()
        b_preprocess_us = b_rows['kernel_time_us'].mean()

    gemm_time_us = gemm_df['kernel_time_us'].mean()
    return a_preprocess_us, b_preprocess_us, gemm_time_us

def calculate_batch_overhead(a_preprocess_us, b_preprocess_us, gemm_us, batch_sizes):
    """Calculate effective preprocessing overhead for different batch sizes.

    Only B preprocessing (weight matrix) is amortizable across batch items.
    A preprocessing (activation matrix) is required every forward pass.

    Formula:
      effective_preprocess = a_preprocess + b_preprocess / batch_size
      overhead% = effective_preprocess / (effective_preprocess + gemm) * 100
    """
    overheads = []
    for batch_size in batch_sizes:
        effective_preprocess = a_preprocess_us + b_preprocess_us / batch_size
        total_time = effective_preprocess + gemm_us
        overhead_pct = (effective_preprocess / total_time) * 100.0
        overheads.append(overhead_pct)
    return np.array(overheads)

def plot_figure4():
    """Generate Figure 4: Batch Amortization"""

    print("Extracting times from benchmark data...")
    a_preprocess_us, b_preprocess_us, gemm_us = extract_times()
    combined_us = a_preprocess_us + b_preprocess_us

    print(f"\nExtracted times (4096×4096, 50% density):")
    print(f"  A preprocessing (non-amortizable): {a_preprocess_us:.1f} µs")
    print(f"  B preprocessing (amortizable):     {b_preprocess_us:.1f} µs")
    print(f"  GEMM:                              {gemm_us:.1f} µs")
    print(f"  Single-call overhead: {(combined_us / (combined_us + gemm_us) * 100):.2f}%")
    print(f"  Note: B-preprocessing (weights) amortizes across calls; A-preprocessing does not.")

    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    overheads = calculate_batch_overhead(a_preprocess_us, b_preprocess_us, gemm_us, batch_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(batch_sizes, overheads, marker='o', linewidth=2.5, markersize=10,
            color='red', label='Effective Preprocessing Overhead\n(B amortized, A per-call)')

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='1% threshold', alpha=0.7)
    ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, label='5% threshold', alpha=0.7)

    # Asymptote: A-only overhead at large batch (B fully amortized)
    a_only_pct = (a_preprocess_us / (a_preprocess_us + gemm_us)) * 100.0
    ax.axhline(y=a_only_pct, color='blue', linestyle=':', linewidth=1.5,
               label=f'A-only floor ({a_only_pct:.2f}%, large batch limit)', alpha=0.7)

    ax.axvspan(32, 128, alpha=0.2, color='blue', label='Typical LLM batch range')

    ax.set_xlabel('Batch Size (# inferences, same weights)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Effective Preprocessing Overhead (%)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: Batch Amortization (K25, 4096×4096, 50% density)\n'
                 'B-preprocessing (weight matrix) amortized across batch; '
                 'A-preprocessing (activations) per-call',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    for batch_size in [1, 32, 128]:
        idx = np.where(batch_sizes == batch_size)[0][0]
        overhead = overheads[idx]
        ax.annotate(f'{overhead:.2f}%',
                   xy=(batch_size, overhead),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    output_file = OUTPUT_DIR / "figure4_batch_amortization.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file.with_suffix('.png')}")
    plt.close()

    print("\n===== Overhead vs Batch Size (corrected model) =====")
    print(f"{'Batch':>8} | {'Overhead %':>11} | {'A contribution':>15} | {'B contribution':>15}")
    print("-" * 60)
    for batch_size, overhead in zip(batch_sizes, overheads):
        eff_pre = a_preprocess_us + b_preprocess_us / batch_size
        a_contrib = (a_preprocess_us / (eff_pre + gemm_us)) * 100.0
        b_contrib = ((b_preprocess_us / batch_size) / (eff_pre + gemm_us)) * 100.0
        print(f"{batch_size:>8} | {overhead:>10.3f}% | {a_contrib:>14.3f}% | {b_contrib:>14.3f}%")

    print(f"\nAt large batch: overhead converges to {a_only_pct:.3f}% (A-preprocessing only)")

if __name__ == "__main__":
    plot_figure4()
