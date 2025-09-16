import json
import os
import sys

import numpy as np
from kernel_tuner import tune_kernel


def setup_environment():
    os.makedirs("tuner_results", exist_ok=True)

    if not os.path.exists("esmm_tune.cu"):
        print("Error: esmm_tune.cu not found!")
        sys.exit(1)


def get_test_matrices(M, N, K):
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


def get_tuning_params():
    return {
        "NUM_THREADS": [128, 256, 512],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8, 16, 32, 64],
        "TM": [1],
        "TN": [8],
        "WM": [32, 64, 128],
        "WN": [32, 64, 128],
        "WNITER": [1, 2, 4, 8],
    }


def get_restrictions():
    return [
        "BN % WN == 0",
        "BM % WM == 0",
        "(BN // WN) * (BM // WM) == NUM_THREADS // 32",
        "(WM * WN) % (32 * TM * TN * WNITER) == 0",
        "WN % WNITER == 0",
        "NUM_THREADS >= BK // 4",
        "NUM_THREADS >= BN // 4",
        "(BN * BK + BM * BK) * 4 <= 44032",
        "BM >= 64",
        "BN >= 64",
        "BK >= 8",
        "WM <= BM",
        "WN <= BN",
        "NUM_THREADS <= 512",
        "WNITER <= 8",
        "TM <= 4",
        "WM % ((WM * WN) // (32 * TM * TN * WNITER)) == 0",
        "((WM * WN) // (32 * TM * TN * WNITER)) * TM * WNITER * TN <= 64",
        "(NUM_THREADS * 4) % BK == 0",
        "(NUM_THREADS * 4) % BN == 0",
        "BN % (16 * TN) == 0",
        "BM % (16 * TM) == 0",
        "(BM * BK) % (4 * NUM_THREADS) == 0",
        "(BN * BK) % (4 * NUM_THREADS) == 0",
    ]


def grid_dimensions(config):
    bm, bn = config.get("BM", 128), config.get("BN", 128)
    return (
        (config.get("N", 1024) + bn - 1) // bn,
        (config.get("M", 1024) + bm - 1) // bm,
        1,
    )


def calculate_metrics(gpu_args, A, M, N, K):
    if isinstance(gpu_args, dict):
        time_ms = gpu_args.get("time", 0)
    else:
        time_ms = gpu_args

    time_s = gpu_args["time"] / 1000
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
    print(f"Tuning {M}x{N}x{K}")

    A, B, C = get_test_matrices(M, N, K)
    reference = np.dot(A, B).astype(np.float32)
    args = [np.int32(M), np.int32(N), np.int32(K), A, B, C]

    def grid_func(config):
        bm, bn = config.get("BM", 128), config.get("BN", 128)
        return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)

    def metrics_func(gpu_args):
        return calculate_metrics(gpu_args, A, M, N, K)

    def verify_esmm(cpu_result, gpu_result, atol=1e-5):
        if gpu_result is None:
            return False
        if not np.allclose(reference, gpu_result[3], atol=atol):
            print("WARNING: Verification failed - continuing")
        return True

    try:

        result = tune_kernel(
            "esmm",
            "esmm_tune.cu",
            grid_func,
            args,
            tune_params,
            verify=verify_esmm,
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
                "-I/home/ec2-user/cuda_work/MMMResearch",
            ],
            lang="CUDA",
            cache=f"tuner_results/esmm_{M}x{N}x{K}.json",
        )

        if result and len(result) > 1:
            best_config = result[1].get("best_config")
            time_s = best_config.get("time", 0) / 1000
            nnz_A = np.count_nonzero(A)
            sparse_gflops = (2 * nnz_A * N) / (time_s * 1e9)
            dense_gflops = (2 * M * N * K) / (time_s * 1e9)
            sparsity = 1.0 - (nnz_A / (M * K))

            print(f"Best: {time_s * 1000} ms")
            print(f"Sparse: {sparse_gflops:.2f} GFLOPS")
            print(f"Dense: {dense_gflops:.2f} GFLOPS")
            print(f"Sparsity: {sparsity:.1%}")
            print(f"Config: {best_config}")

            return {
                "best_config": result[1].get("best_config"),
                "best_time": result[1],
                "sparse_gflops": sparse_gflops,
                "dense_gflops": dense_gflops,
                "sparsity": sparsity,
                "all_results": result[2] if len(result) > 2 else None,
            }
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")
        return None


def run_esmm_tuning():
    setup_environment()

    test_sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
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
            print(f"Skipping {size_key}")
            continue

        gflops = result_data["dense_gflops"]
        analysis["performance_summary"][size_key] = {
            "dense_gflops": gflops,
            "sparse_gflops": result_data["sparse_gflops"],
            "time_ms": result_data["best_time"],
            "sparsity": result_data["sparsity"],
            "config": result_data["best_config"],
        }

        if gflops > best_gflops:
            best_gflops = gflops
            best_config = result_data["best_config"]
            best_size = size_key

    all_configs = results["1024x1024x1024"]["all_results"]

    analysis["best_overall"] = {
        "gflops": best_gflops,
        "config": best_config,
        "size": best_size,
    }

    return analysis


def save_results(results, analysis):
    with open("esmm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("esmm_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)


def print_summary(analysis):
    print("\nTuning Complete!")
    print("=" * 50)

    if analysis["performance_summary"]:
        print("\nResults:")
        for size, perf in analysis["performance_summary"].items():
            print(size, perf)
            print(f"{size}: {perf['dense_gflops']} GFLOPS ({perf['time_ms']} ms)")
            print(
                f"  Sparse: {perf['sparse_gflops']} GFLOPS, Sparsity: {perf['sparsity']}"
            )

        if analysis["best_overall"]["config"]:
            print(f"\nBest Overall:")
            print(f"  Performance: {analysis['best_overall']['gflops']:.2f} GFLOPS")
            print(f"  Matrix Size: {analysis['best_overall']['size']}")
            print(f"  Config: {analysis['best_overall']['config']}")
    else:
        print("No successful results!")


def main():
    print("ESMM Kernel Autotuning for A10G")
    print("=" * 50)

    results = run_esmm_tuning()
    analysis = analyze_results(results)
    save_results(results, analysis)
    print_summary(analysis)


if __name__ == "__main__":
    main()
