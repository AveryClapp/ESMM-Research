"""
Kernel Tuner setup for autotuning K11 double buffered GEMM kernel
Based on the CUDA MMM implementation from your project knowledge
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from kernel_tuner import tune_kernel


def setup_k11_autotuning():
    """
    Complete autotuning setup for your K11 double buffered kernel
    """

    kernel_code = """
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

#define WARPSIZE 32

__forceinline__ __device__ void multiply_dense(int wSubRowIdx, int wSubColIdx,
                                int WNITER, float regM_val, float* regN,
                                        float* threadResults) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
    threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
    threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
    threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
    threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
    threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
    threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
    threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
    threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

namespace db {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
      &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // transpose A while storing it
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
      &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
      reinterpret_cast<float4 *>(
        &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN, float *threadResults, 
                                const float *As, const float *Bs, const int warpRow, 
                                const int warpCol, const uint threadRowInWarp, 
                                const int threadColInWarp) {
  for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        regM[wSubRowIdx] =
          As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
          threadRowInWarp];
    }
    for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (int i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
          Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
          threadColInWarp * TN + i];
      }
    }

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        //calculate per-thread results
        multiply_dense(wSubRowIdx, wSubColIdx, WNITER, regM[wSubRowIdx], regN, threadResults);
      }
    }
  }
}
} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_buffered(const int M, const int N, const int K, float *A, float *B, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);

  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int threadIdxInWarp = threadIdx.x % WARPSIZE;
  const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // Allocate space for the current blocktile in SMEM
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  // Divide threads into two groups: loaders and computers
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  /**
   * Calculate the indices that this thread will load into SMEM.
   * This is half of what we used for other kernels since we are dividing into two groups
   */
  const int innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const int innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr int rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const int innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const int innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr int rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // Load block 0
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As[0], Bs[0], innerRowA, innerColA, innerRowB, innerColB);
  }

  __syncthreads();

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // Process block 0
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
        TN>(regM, regN, threadResults, As[0], Bs[0], warpRow,
            warpCol, threadRowInWarp, threadColInWarp);

      // Process block 1 (loaded by other side)
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
          TM, TN>(regM, regN, threadResults, As[1],
                  Bs[1], warpRow, warpCol,
                  threadRowInWarp, threadColInWarp);
      }

      // Load block 2 into the first half of As & Bs (this will be block 0 next iteration)
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
          N, K, A + 2 * BK, B + 2 * BK * N, As[0], Bs[0], innerRowA, innerColA,
          innerRowB, innerColB);
      }
    } else {
      // Load block 1 into the second half of As & Bs
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
          N, K, A + BK, B + BK * N, As[1], Bs[1], innerRowA,
          innerColA, innerRowB, innerColB);
      }

      // Process the rest of block 0
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
        TN>(regM, regN, threadResults, As[0], Bs[0], warpRow,
            warpCol, threadRowInWarp, threadColInWarp);

      // Process the rest of block 1
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
          TM, TN>(regM, regN, threadResults, As[1],
                  Bs[1], warpRow, warpCol,
                  threadRowInWarp, threadColInWarp);
      }
    }

    A += 2 * BK;
    B += 2 * BK * N;
    __syncthreads();
  }

  for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp;
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
            wSubColIdx * TN + resIdxN;
          tmp.x = threadResults[i + 0];
          tmp.y = threadResults[i + 1];
          tmp.z = threadResults[i + 2];
          tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
            threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}
    """

    # Test matrix dimensions - start with sizes where you have performance gaps vs cuBLAS
    test_sizes = [
        (512, 512, 512),  # Small matrices where cuBLAS dominates
        (1024, 1024, 1024),  # Medium matrices
        (2048, 2048, 2048),  # Large matrices where you're competitive
        (4096, 4096, 4096),  # Very large matrices
    ]

    # Comprehensive parameter space based on your existing autotuning
    tune_params = {
        # Block tiling parameters - expand from your current best configs
        "NUM_THREADS": [128, 256, 512],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8, 16, 32],
        # Thread tiling - optimize for register usage
        "TM": [1],  # Fixed to 8 based on multiply_dense implementation
        "TN": [8],  # Fixed to 8 based on multiply_dense implementation
        # Warp tiling parameters - critical for double buffering efficiency
        "WN": [32, 64, 128],
        "WM": [32, 64, 128],
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

    return kernel_code, tune_params, restrictions, test_sizes


def run_comprehensive_autotuning():
    """
    Run systematic autotuning across multiple matrix sizes and optimization strategies
    """
    kernel_code, tune_params, restrictions, test_sizes = setup_k11_autotuning()

    os.makedirs("tuner_results", exist_ok=True)

    results = {}

    for M, N, K in test_sizes:
        print(f"\n=== Autotuning for matrix size {M}x{N}x{K} ===")

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        args = [np.int32(M), np.int32(N), np.int32(K), A, B, C]

        def grid_dimensions(config):
            bm, bn = config.get("BM", 128), config.get("BN", 128)
            return ((N + bn - 1) // bn, (M + bm - 1) // bm, 1)

        def verify_output(params, answer, atol=1e-4):
            reference = np.dot(A, B).astype(np.float32)
            if answer is None:
                return False
            return np.allclose(answer, reference, atol=atol)

        def custom_metrics(gpu_args):
            flops = 2 * M * N * K  # Fused multiply-add ops
            memory_bytes = (M * K + K * N + M * N) * 4  # float32

            return {
                "GFLOPS": flops / (gpu_args["time"] * 1e9),
                "GB_per_s": memory_bytes / (gpu_args["time"] * 1e9),
                "arithmetic_intensity": flops / memory_bytes,
            }

        try:
            result = tune_kernel(
                "esmm_buffered",
                kernel_code,
                grid_dimensions,
                args,
                tune_params,
                restrictions=restrictions,
                answer=np.dot(A, B).astype(np.float32),  # Reference for verification
                atol=1e-4,
                verbose=True,
                iterations=3,  # Average over multiple runs
                compiler_options=["-w"],  # Suppress all warnings
                lang="CUDA",
                compiler="/usr/local/cuda-12.1/bin/nvcc",
                cache=f"tuner_results/buffered_cache_{M}x{N}x{K}.json",  # Cache results per size
                metrics=custom_metrics,
            )

            if result and len(result) > 1:
                results[f"{M}x{N}x{K}"] = {
                    "best_config": result[0],
                    "best_time": result[1],
                    "all_results": result[2] if len(result) > 2 else None,
                }

                # Calculate performance metrics for best result
                best_time_s = result[1] / 1000  # Convert ms to seconds
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
    Analyze autotuning results and extract insights
    """
    analysis = {"performance_summary": {}, "optimal_configs": {}, "trends": {}}

    for size_key, result_data in results.items():
        M, N, K = map(int, size_key.split("x"))

        if "best_time" in result_data:
            # Calculate theoretical peak performance
            flops = 2 * M * N * K
            time_s = result_data["best_time"] / 1000
            gflops = flops / (time_s * 1e9)

            analysis["performance_summary"][size_key] = {
                "gflops": gflops,
                "time_ms": result_data["best_time"],
                "config": result_data["best_config"],
            }

            # Extract key config parameters for trend analysis
            config = result_data["best_config"]
            for param in ["BM", "BN", "BK", "WM", "WN", "WNITER"]:
                if param not in analysis["trends"]:
                    analysis["trends"][param] = []
                analysis["trends"][param].append((M, config.get(param)))

    return analysis


def generate_production_templates(results):
    """
    Generate optimized kernel templates for different use cases
    """
    templates = {
        "small_matrices": {},  # For transformer attention, small batch inference
        "large_matrices": {},  # For training, large batch inference
        "general_purpose": {},  # Best overall configuration
    }

    best_gflops = 0

    for size_key, result_data in results.items():
        if "best_config" not in result_data or "best_time" not in result_data:
            continue

        M, N, K = map(int, size_key.split("x"))

        # Calculate GFLOPS for this configuration
        flops = 2 * M * N * K
        time_s = result_data["best_time"] / 1000
        gflops = flops / (time_s * 1e9)

        config = result_data["best_config"].copy()
        config["performance_gflops"] = gflops
        config["matrix_size"] = f"{M}x{N}x{K}"

        if M <= 1024:
            if not templates["small_matrices"] or gflops > templates[
                "small_matrices"
            ].get("performance_gflops", 0):
                templates["small_matrices"] = config
        else:
            if not templates["large_matrices"] or gflops > templates[
                "large_matrices"
            ].get("performance_gflops", 0):
                templates["large_matrices"] = config

        # Track best overall
        if gflops > best_gflops:
            best_gflops = gflops
            templates["general_purpose"] = config

    return templates


if __name__ == "__main__":
    print("Starting K11 Double Buffered Kernel Autotuning...")
    print("=" * 70)

    results = run_comprehensive_autotuning()

    if not results:
        print("No results obtained from autotuning!")
        exit(1)

    analysis = analyze_results(results)

    templates = generate_production_templates(results)

    with open("k11_autotuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("k11_performance_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    with open("k11_optimized_templates.json", "w") as f:
        json.dump(templates, f, indent=2)

    print("\n" + "=" * 70)
    print("AUTOTUNING COMPLETE!")
    print("=" * 70)

    print("\nPerformance Summary:")
    for size_key, perf_data in analysis["performance_summary"].items():
        print(
            f"  {size_key}: {perf_data['gflops']:.2f} GFLOPS ({perf_data['time_ms']:.3f} ms)"
        )

    print(f"\nFiles generated:")
    print(f"  - k11_autotuning_results.json: Raw autotuning results")
    print(f"  - k11_performance_analysis.json: Performance analysis and trends")
    print(f"  - k11_optimized_templates.json: Production-ready configurations")

    print(f"\nBest overall configuration:")
    if "general_purpose" in templates and templates["general_purpose"]:
        best_config = templates["general_purpose"]
        print(f"  Performance: {best_config['performance_gflops']:.2f} GFLOPS")
        print(f"  Matrix Size: {best_config['matrix_size']}")
        print(
            f"  Config: BM={best_config.get('BM')}, BN={best_config.get('BN')}, BK={best_config.get('BK')}"
        )
        print(
            f"          WM={best_config.get('WM')}, WN={best_config.get('WN')}, WNITER={best_config.get('WNITER')}"
        )
    else:
        print("  No valid configurations found!")
