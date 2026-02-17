#!/usr/bin/env python3
"""
cuBLAS Baseline Runner
Runs cuBLAS GEMM at specified matrix sizes and outputs timing results
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path

def run_cublas_baseline(sizes, output_file):
    """
    Run cuBLAS baseline measurements

    TODO: This is a stub. Implement one of these options:
    1. Call cublasSgemm directly via ctypes/pybind11
    2. Use existing K10 kernel which internally calls cuBLAS
    3. Create a minimal CUDA program that just calls cublasSgemm

    For now, we use K10 (dense ESMM) as a proxy for cuBLAS performance.
    """

    print("Running cuBLAS baseline (Kernel 15)")

    # Use benchmark.py with K15 kernel (actual cuBLAS)
    project_root = Path(__file__).parent.parent
    benchmark_script = project_root / "scripts" / "benchmark.py"

    results = []

    for size in sizes:
        print(f"\nRunning cuBLAS (K15) at size {size}...")

        # Run benchmark
        cmd = [
            "python3", str(benchmark_script),
            "--kernel", "15",
            "--sizes", str(size),
            "--sparsity", "11111111",  # Dense
            "--cold-start",
            "--parallel", "1",
            "-o", "/tmp/cublas_baseline"
        ]

        subprocess.run(cmd, check=True)

        # Read results
        summary_file = Path("/tmp/cublas_baseline/summary.csv")
        if summary_file.exists():
            df = pd.read_csv(summary_file, comment='#')
            df = df[df['kernel'] != 'PREPROCESS']
            time_us = df['kernel_time_us'].mean()
            results.append({'size': size, 'time_us': time_us})
            print(f"  Size {size}: {time_us:.1f} µs")

    # Save results
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cuBLAS baseline benchmarks")
    parser.add_argument("--size", type=int, help="Single matrix size (if not using --sizes)")
    parser.add_argument("--sizes", type=str, help="Comma-separated matrix sizes (e.g., '1024,2048,4096')")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")

    args = parser.parse_args()

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(',')]
    elif args.size:
        sizes = [args.size]
    else:
        print("ERROR: Must specify --size or --sizes")
        exit(1)

    run_cublas_baseline(sizes, args.output)
