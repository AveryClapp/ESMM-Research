"""
Fixed K11 Double Buffered Kernel Autotuning
Uses external .cu file instead of embedded kernel code
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from kernel_tuner import tune_kernel


def setup_k11_autotuning():
    """
    Setup autotuning for K11 kernel using external .cu file
    """

    # Test sizes for autotuning
    test_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    # Tunable parameters
    tune_params = {
        "NUM_THREADS": [128, 256],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8, 16, 32],
        "TM": [1],  # Keep fixed for now
        "TN": [8],
        "WM": [32, 64, 128],
        "WN": [32, 64, 128],
        "WNITER": [1, 2, 4],
    }

    # Constraints to ensure valid configurations
    restrictions = [
        "BN % WN == 0",
        "BM % WM == 0",
        "(BN / WN) * (BM / WM) == NUM_THREADS / 32",
        "(WM * WN) % (32 * TM * TN * WNITER) == 0",
        "WN % WNITER == 0",
        "(NUM_THREADS / 2 * 4) % BK == 0",
        "(NUM_THREADS / 2 * 4) % BN == 0",
        "BN % (16 * TN) == 0",
        "BM % (16 * TM) == 0",
        "(BM * BK) % (4 * NUM_THREADS / 2) == 0",
        "(BN * BK) % (4 * NUM_THREADS / 2) == 0",
        # WMITER calculation constraint
        "WM % ((WM * WN) / (32 * TM * TN * WNITER)) == 0",
        # Ensure WSUBN is divisible by TN for thread indexing
        "(WN / WNITER) % TN == 0",
    ]

    return tune_params, restrictions, test_sizes


def run_autotuning():
    """
    Run autotuning using external .cu file
    """
    tune_params, restrictions, test_sizes = setup_k11_autotuning()

    os.makedirs("tuner_results", exist_ok=True)
    results = {}

    for M, N, K in test_sizes:
        print(f"\n=== Autotuning for matrix size {M}x{N}x{K} ===")

        # Create test matrices
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        # Arguments for the kernel
        args = [np.int32(M), np.int32(N), np.int32(K), A, B, C]

        # Grid size calculation function
        def grid_dimensions(config):
            bm = config.get("BM", 128)
            bn = config.get("BN", 128)
            return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)

        # Reference answer for verification
        reference = np.dot(A, B).astype(np.float32)

        try:
            result = tune_kernel(
                "esmm",  # Kernel name
                "esmm_tune.cu",  # External .cu file
                grid_dimensions,  # Grid size function
                args,  # Kernel arguments
                tune_params,  # Parameters to tune
                restrictions=restrictions,  # Constraints
                # atol=1e-4,  # Tolerance for verification
                iterations=3,
                verbose=True,  # Print progress
                block_size_names=["NUM_THREADS"],
                # answer=reference,
                compiler_options=[
                    "-I.",
                    "-I/home/ec2-user/cuda_work/MMMResearch",
                    "-w",  # Suppress warnings
                ],
                lang="CUDA",
                cache=f"tuner_results/buffered_cache_{M}x{N}x{K}.json",
            )

            if result and len(result) > 1:
                results[f"{M}x{N}x{K}"] = {
                    "best_config": result[0],
                    "best_time": result[1],
                    "all_results": result[2] if len(result) > 2 else None,
                }

                # Calculate GFLOPS
                best_time_s = result[1] / 1000
                gflops = (2 * M * N * K) / (best_time_s * 1e9)

                print(f"Best performance: {result[1]:.6f} ms ({gflops:.2f} GFLOPS)")
                print(f"Best config: {result[0]}")
            else:
                print(f"No valid results found for {M}x{N}x{K}")

        except Exception as e:
            print(f"Error tuning {M}x{N}x{K}: {str(e)}")
            continue

    return results


def analyze_results(results):
    """
    Analyze autotuning results
    """
    if not results:
        print("No results to analyze!")
        return {}

    analysis = {"performance_summary": {}, "trends": {}}

    for size_key, result_data in results.items():
        if "best_time" not in result_data:
            continue

        M, N, K = map(int, size_key.split("x"))

        # Calculate performance metrics
        flops = 2 * M * N * K
        time_s = result_data["best_time"] / 1000
        gflops = flops / (time_s * 1e9)

        analysis["performance_summary"][size_key] = {
            "gflops": gflops,
            "time_ms": result_data["best_time"],
            "config": result_data["best_config"],
        }

    return analysis


if __name__ == "__main__":
    print("Starting Kernel Autotuning...")
    print("=" * 70)

    # Check if the kernel file exists
    kernel_file = "esmm_tune.cu"
    if not os.path.exists(kernel_file):
        print(f"Error: Kernel file '{kernel_file}' not found!")
        print("Make sure the C-compatible kernel file is in the current directory.")
        exit(1)

    results = run_autotuning()

    if not results:
        print("No results obtained from autotuning!")
        exit(1)

    # Analyze and save results
    analysis = analyze_results(results)

    with open("autotuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("performance_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 70)
    print("AUTOTUNING COMPLETE!")
    print("=" * 70)

    print("\nPerformance Summary:")
    for size_key, perf_data in analysis["performance_summary"].items():
        print(
            f"  {size_key}: {perf_data['gflops']:.2f} GFLOPS ({perf_data['time_ms']:.3f} ms)"
        )

    print(f"\nFiles generated:")
    print(f"  - autotuning_results.json: Raw autotuning results")
    print(f"  - performance_analysis.json: Performance analysis")

    # Find best overall configuration
    best_gflops = 0
    best_config = None
    best_size = None

    for size_key, perf_data in analysis["performance_summary"].items():
        if perf_data["gflops"] > best_gflops:
            best_gflops = perf_data["gflops"]
            best_config = perf_data["config"]
            best_size = size_key

    if best_config:
        print(f"\nBest overall configuration:")
        print(f"  Performance: {best_gflops:.2f} GFLOPS")
        print(f"  Matrix Size: {best_size}")
        print(
            f"  Config: BM={best_config.get('BM')}, BN={best_config.get('BN')}, BK={best_config.get('BK')}"
        )
        print(
            f"          WM={best_config.get('WM')}, WN={best_config.get('WN')}, WNITER={best_config.get('WNITER')}"
        )
        print(
            f"          TM={best_config.get('TM')}, TN={best_config.get('TN')}, NUM_THREADS={best_config.get('NUM_THREADS')}"
        )
