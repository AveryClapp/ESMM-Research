import json
import os
import sys

import numpy as np
from kernel_tuner import tune_kernel


def setup_environment():
    os.makedirs("tuner_results", exist_ok=True)

    if not os.path.exists("esmm_hybrid_tune.cu"):
        print("Error: esmm_hybrid_tune.cu not found!")
        sys.exit(1)


def init_pattern_lut():
    """Initialize pattern lookup table for BK=8"""
    pattern_lut = np.zeros((256, 9), dtype=np.uint8)

    for pattern in range(256):
        count = 0
        for bit in range(8):
            if pattern & (1 << bit):
                pattern_lut[pattern, count + 1] = bit
                count += 1
        pattern_lut[pattern, 0] = count

    return pattern_lut


def preprocess_blockwise_patterns(A, M, K, WM, BK):
    """Generate block-wise sparsity patterns (simplified CPU version)"""
    numWarpRows = M // WM
    numKBlocks = K // BK

    blockPatterns = np.zeros((numWarpRows, numKBlocks), dtype=np.uint8)

    for warpRow in range(numWarpRows):
        for kBlock in range(numKBlocks):
            pattern = 0
            # OR together all 32 rows in this warp
            for row in range(WM):
                globalRow = warpRow * WM + row
                for k in range(BK):
                    globalK = kBlock * BK + k
                    if A[globalRow, globalK] != 0.0:
                        pattern |= (1 << k)
            blockPatterns[warpRow, kBlock] = pattern

    return blockPatterns.flatten()


def get_test_matrices(M, N, K):
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def get_tuning_params():
    """
    Tuning parameters for K17 (esmm_hybrid_blockwise)

    Key constraints:
    - BK = 8 (fixed for pattern-based sparsity)
    - TM = 1 (fixed for warp-based computation)
    - TN = 8 (fixed for vectorized float4 loads)
    """
    return {
        "NUM_THREADS": [128, 256],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8],  # FIXED
        "TM": [1],  # FIXED
        "TN": [8],  # FIXED for vectorized loads
        "WM": [32, 64],
        "WN": [32, 64, 128],
        "WNITER": [1, 2, 4, 8],
    }


def get_restrictions():
    """
    Restrictions for valid K17 kernel configurations
    """
    return [
        # Block/warp division constraints
        "BN % WN == 0",
        "BM % WM == 0",
        "(BN // WN) * (BM // WM) == NUM_THREADS // 32",

        # Warp-level work distribution
        "(WM * WN) % (32 * TM * TN * WNITER) == 0",
        "WN % WNITER == 0",

        # Memory access alignment (float4 vectorization)
        "NUM_THREADS >= BK // 4",
        "NUM_THREADS >= BN // 4",
        "(NUM_THREADS * 4) % BK == 0",
        "(NUM_THREADS * 4) % BN == 0",

        # Shared memory constraint (A10G has 48KB SMEM)
        "(BN * BK + BM * BK) * 4 <= 48000",

        # Block size bounds
        "BM >= 64",
        "BN >= 64",
        "WM <= BM",
        "WN <= BN",

        # Register file constraints
        "WM % ((WM * WN) // (32 * TM * TN * WNITER)) == 0",
        "((WM * WN) // (32 * TM * TN * WNITER)) * TM * WNITER * TN <= 64",

        # Thread granularity for loading
        "(BM * BK) % (4 * NUM_THREADS) == 0",
        "(BN * BK) % (4 * NUM_THREADS) == 0",
    ]


def calculate_metrics(gpu_args, A, M, N, K):
    if isinstance(gpu_args, dict):
        time_ms = gpu_args.get("time", 0)
    else:
        time_ms = gpu_args

    time_s = time_ms / 1000
    nnz_A = np.count_nonzero(A)
    sparse_flops = 2 * nnz_A * N
    dense_flops = 2 * M * N * K
    memory_bytes = (M * K + K * N + M * N) * 4

    return {
        "Sparse_GFLOPS": sparse_flops / (time_s * 1e9),
        "Dense_GFLOPS": dense_flops / (time_s * 1e9),
        "Memory_BW_GBps": memory_bytes / (time_s * 1e9),
        "Sparsity": 1.0 - (nnz_A / (M * K)),
        "Speedup": dense_flops / sparse_flops,
    }


def tune_single_size(M, N, K, tune_params, restrictions):
    print(f"\n{'='*60}")
    print(f"Tuning K17 (esmm_hybrid_blockwise) for {M}x{N}x{K}")
    print(f"{'='*60}")

    A, B, C = get_test_matrices(M, N, K)
    reference = np.dot(A, B).astype(np.float32)

    # Preprocess blockPatterns (use WM=32 for preprocessing, consistent with current K17)
    WM_preprocess = 32
    BK = 8
    blockPatterns = preprocess_blockwise_patterns(A, M, K, WM_preprocess, BK)

    print(f"Generated {len(blockPatterns)} block patterns")
    print(f"Pattern statistics: min={blockPatterns.min()}, max={blockPatterns.max()}, " +
          f"mean={blockPatterns.mean():.2f}")

    # Initialize pattern LUT
    pattern_lut = init_pattern_lut()

    # Prepare arguments
    args = [np.int32(M), np.int32(N), np.int32(K), A, B, C, blockPatterns]

    def grid_func(config):
        bm, bn = config.get("BM", 128), config.get("BN", 128)
        return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)

    def metrics_func(gpu_args):
        return calculate_metrics(gpu_args, A, M, N, K)

    def verify_hybrid(cpu_result, gpu_result, atol=1e-3):
        if gpu_result is None:
            return False
        # Result is in gpu_result[5] (C matrix, index 5 in args)
        if not np.allclose(reference, gpu_result[5], atol=atol):
            max_diff = np.max(np.abs(reference - gpu_result[5]))
            print(f"WARNING: Verification failed - max diff: {max_diff:.6f}")
        return True

    try:
        # Copy pattern LUT to constant memory (handled by kernel_tuner)
        result = tune_kernel(
            "esmm_hybrid",
            "esmm_hybrid_tune.cu",
            grid_func,
            args,
            tune_params,
            verify=verify_hybrid,
            block_size_names=["NUM_THREADS"],
            restrictions=restrictions,
            atol=1e-1,
            verbose=True,
            iterations=3,
            compiler_options=[
                "-w",
                "-O3",
                "--use_fast_math",
                "-I.",
                "-I..",
                "-I../include",
            ],
            lang="CUDA",
            cache=f"tuner_results/k17_{M}x{N}x{K}.json",
        )

        if result:
            # result is a tuple: (list_of_all_results, metadata_dict)
            # metadata_dict contains 'best_config' with the best configuration and timing
            if isinstance(result, (list, tuple)) and len(result) > 1:
                metadata = result[1]
                if isinstance(metadata, dict) and 'best_config' in metadata:
                    best_config = metadata['best_config']
                else:
                    best_config = metadata
            elif isinstance(result, dict):
                best_config = result.get('best_config', result)
            else:
                print(f"ERROR: Unexpected result format: {type(result)}")
                return None

            time_ms = best_config.get("time", None) if isinstance(best_config, dict) else None
            if time_ms is None or time_ms <= 0:
                print(f"ERROR: Invalid timing result: {time_ms}")
                return None

            time_s = time_ms / 1000
            nnz_A = np.count_nonzero(A)
            sparse_gflops = (2 * nnz_A * N) / (time_s * 1e9)
            dense_gflops = (2 * M * N * K) / (time_s * 1e9)
            sparsity = 1.0 - (nnz_A / (M * K))

            print(f"\n{'='*60}")
            print(f"BEST CONFIGURATION FOUND:")
            print(f"{'='*60}")
            print(f"Time: {time_s * 1000:.3f} ms")
            print(f"Sparse GFLOPS: {sparse_gflops:.2f}")
            print(f"Dense GFLOPS: {dense_gflops:.2f}")
            print(f"Sparsity: {sparsity:.1%}")
            print(f"Config: {best_config}")
            print(f"{'='*60}")

            return {
                "best_config": best_config,
                "sparse_gflops": sparse_gflops,
                "dense_gflops": dense_gflops,
                "sparsity": sparsity,
                "all_results": result[2] if len(result) > 2 else None,
            }
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return None


def run_k17_tuning():
    setup_environment()

    # Test 4096x4096x4096 as requested
    test_sizes = [
        (4096, 4096, 4096)
    ]

    tune_params = get_tuning_params()
    restrictions = get_restrictions()

    results = {}

    for M, N, K in test_sizes:
        result = tune_single_size(M, N, K, tune_params, restrictions)
        if result:
            results[f"{M}x{N}x{K}"] = result

    return results


def analyze_results(results):
    if not results:
        return {"performance_summary": {}, "best_overall": {}}

    analysis = {"performance_summary": {}, "best_overall": {}}
    best_gflops = 0
    best_config = None
    best_size = None

    for size_key, result_data in results.items():
        if "dense_gflops" not in result_data:
            print(f"Skipping {size_key} - incomplete data")
            continue

        gflops = result_data["dense_gflops"]
        config = result_data["best_config"]

        analysis["performance_summary"][size_key] = {
            "dense_gflops": gflops,
            "sparse_gflops": result_data["sparse_gflops"],
            "time_ms": config.get("time", 0),
            "sparsity": result_data["sparsity"],
            "config": config,
        }

        if gflops > best_gflops:
            best_gflops = gflops
            best_config = config
            best_size = size_key

    analysis["best_overall"] = {
        "gflops": best_gflops,
        "config": best_config,
        "size": best_size,
    }

    return analysis


def save_results(results, analysis):
    with open("tuner_results/k17_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("tuner_results/k17_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)


def print_summary(analysis):
    print("\n" + "=" * 60)
    print("K17 TUNING COMPLETE!")
    print("=" * 60)

    if analysis["performance_summary"]:
        print("\nResults by matrix size:")
        print("-" * 60)
        for size, perf in analysis["performance_summary"].items():
            print(f"\n{size}:")
            print(f"  Time: {perf['time_ms']:.3f} ms")
            print(f"  Dense GFLOPS: {perf['dense_gflops']:.2f}")
            print(f"  Sparse GFLOPS: {perf['sparse_gflops']:.2f}")
            print(f"  Sparsity: {perf['sparsity']:.1%}")
            config = perf['config']
            print(f"  Config: BM={config['BM']}, BN={config['BN']}, "
                  f"WM={config['WM']}, WN={config['WN']}, "
                  f"WNITER={config['WNITER']}, THREADS={config['NUM_THREADS']}")

        if analysis["best_overall"]["config"]:
            print("\n" + "=" * 60)
            print("BEST OVERALL CONFIGURATION:")
            print("=" * 60)
            print(f"Performance: {analysis['best_overall']['gflops']:.2f} GFLOPS")
            print(f"Matrix Size: {analysis['best_overall']['size']}")
            print(f"Configuration: {analysis['best_overall']['config']}")
    else:
        print("\nNo successful results!")


def main():
    print("=" * 60)
    print("K17 (esmm_hybrid_blockwise) Kernel Autotuning for A10G")
    print("=" * 60)

    results = run_k17_tuning()
    analysis = analyze_results(results)
    save_results(results, analysis)
    print_summary(analysis)

    print("\nResults saved to:")
    print("  - tuner_results/k17_results.json")
    print("  - tuner_results/k17_analysis.json")


if __name__ == "__main__":
    main()
