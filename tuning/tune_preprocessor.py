#!/usr/bin/env python3
"""
Autotuner for row-level preprocessing kernel
Tests different BK and NUM_THREADS configurations to find optimal performance
"""

import json
import os
import sys
import numpy as np
from kernel_tuner import tune_kernel


def get_test_matrix(M, K, sparsity=0.5):
    """Generate a random sparse matrix for testing"""
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    # Apply sparsity
    mask = np.random.rand(M, K) < sparsity
    A[mask] = 0.0
    return A


def get_tuning_params():
    """
    Tuning parameters for preprocessing kernel:
    - BK: 8 or 16 (K-block size for bitmask)
    - NUM_THREADS: Thread block size (must be multiple of 32)
    """
    return {
        "TUNE_BK": [8, 16],
        "TUNE_NUM_THREADS": [128, 256, 512],
    }


def calculate_grid_dim(M, BK, NUM_THREADS):
    """Calculate grid dimensions based on configuration"""
    WARP_SIZE = 32
    ROWS_PER_WARP = WARP_SIZE // BK
    NUM_WARPS = NUM_THREADS // WARP_SIZE
    ROWS_PER_BLOCK = ROWS_PER_WARP * NUM_WARPS
    grid_dim = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    return grid_dim


def get_restrictions():
    """
    Valid configuration restrictions
    """
    return [
        # Thread count must be multiple of warp size
        "TUNE_NUM_THREADS % 32 == 0",
        # BK must divide evenly into warp size for efficient processing
        "32 % TUNE_BK == 0",
        # Minimum thread count for efficiency
        "TUNE_NUM_THREADS >= 128",
        # Maximum thread count (hardware limit)
        "TUNE_NUM_THREADS <= 1024",
    ]


def tune_preprocessor(M=4096, K=4096):
    """Run the autotuner"""
    print(f"Tuning preprocessor for M={M}, K={K}")
    print("=" * 60)

    # Generate test data
    A = get_test_matrix(M, K, sparsity=0.5)

    # Calculate maximum output size (for BK=8, largest case)
    max_numKBlocks = K // 8
    A_LIST = np.zeros(M * max_numKBlocks, dtype=np.uint8)

    # Kernel arguments
    args = [np.int32(M), np.int32(M), np.int32(K), A, A_LIST]

    # Tuning parameters
    tune_params = get_tuning_params()

    # Calculate a safe grid size (will use max needed)
    max_grid = calculate_grid_dim(M, 8, 128)  # Worst case: BK=8, fewest threads

    # Run the tuner (using default pycuda backend)
    results, env = tune_kernel(
        "preprocess_A_rowlevel_tune",
        "preprocess_tuner.cu",
        (max_grid, 1, 1),
        args,
        tune_params,
        restrictions=get_restrictions(),
        block_size_names=["TUNE_NUM_THREADS"],
        verbose=True,
        compiler_options=["-O3", "-use_fast_math"],
        iterations=32,  # Number of timing iterations per config
    )

    # Save results
    output_file = "tuner_results/preprocessor_results.json"
    os.makedirs("tuner_results", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print best configuration
    if results:
        best = min(results, key=lambda x: x['time'])
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION:")
        print("=" * 60)
        print(f"BK: {best['TUNE_BK']}")
        print(f"NUM_THREADS: {best['TUNE_NUM_THREADS']}")
        print(f"Time: {best['time']:.4f} ms")
        print(f"Bandwidth: {(M * K * 4 / best['time'] / 1e6):.2f} GB/s")

        # Calculate grid dimensions
        grid_dim = calculate_grid_dim(M, best['TUNE_BK'], best['TUNE_NUM_THREADS'])
        print(f"Grid dimensions: ({grid_dim}, 1, 1)")
        print(f"Block dimensions: ({best['TUNE_NUM_THREADS']}, 1, 1)")


if __name__ == "__main__":
    # Change to preprocessors directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Parse command line arguments for matrix size
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    tune_preprocessor(M, K)
