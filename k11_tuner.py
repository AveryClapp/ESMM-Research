"""
Kernel Tuner setup for autotuning K11 double buffered GEMM kernel
Based on the CUDA MMM implementation from your project knowledge
"""

import json
from typing import Dict, List, Tuple

import numpy as np
from kernel_tuner import tune_kernel


def setup_k11_autotuning():
    """
    Complete autotuning setup for your K11 double buffered kernel
    """

    kernel_code = """
namespace db {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
      &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // transpose A while storing it
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
      &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
      reinterpret_cast<float4 *>(
        &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN, float *threadResults, 
                                const float *As, const float *Bs, const uint warpRow, 
                                const uint warpCol, const uint threadRowInWarp, 
                                const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        regM[wSubRowIdx] =
          As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
          threadRowInWarp];
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
          Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
          threadColInWarp * TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
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
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

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
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

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
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
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
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
        TM, TN>(regM, regN, threadResults, As[1],
                Bs[1], warpRow, warpCol,
                threadRowInWarp, threadColInWarp);
    }


    A += 2 * BK;
    B += 2 * BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
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
        "NUM_THREADS": [128, 256],
        "BM": [64, 128, 256],
        "BN": [64, 128, 256],
        "BK": [8, 16, 32],
        # Thread tiling - optimize for register usage
        "TM": [1],
        "TN": [8, 16, 32],
        # Warp tiling parameters - critical for double buffering efficiency
        "WN": [16, 32, 64],
        "WM": [16, 32, 64],
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
        # WM % WMITER == 0
        "WM % ((WM * WN) / (32 * TM * TM * WNITER)) == 0",
    ]

    return kernel_code, tune_params, restrictions, test_sizes


def run_comprehensive_autotuning():
    """
    Run systematic autotuning across multiple matrix sizes and optimization strategies
    """
    kernel_code, tune_params, restrictions, test_sizes = setup_k11_autotuning()

    results = {}

    for M, N, K in test_sizes:
        print(f"\n=== Autotuning for matrix size {M}x{N}x{K} ===")

        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        args = [A, B, C, np.int32(M), np.int32(N), np.int32(K)]

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
            results = tune_kernel(
                "esmm_buffereed",
                kernel_code,
                grid_dimensions,
                args,
                tune_params,
                restrictions=restrictions,
                answer=np.dot(A, B).astype(np.float32),  # Reference for verification
                atol=1e-4,
                verbose=True,
                iterations=3,  # Average over multiple runs
                cache=f"tuner_results/buffered_cache_{M}x{N}x{K}.json",  # Cache results per size
            )

            print(f"Performance: {result[1]:.6f} ms")

        except Exception as e:
            print(f"e")
            continue

    return results


def generate_production_templates(results):
    """
    Generate optimized kernel templates for different use cases
    """
    templates = {
        "small_matrices": {},  # For transformer attention, small batch inference
        "large_matrices": {},  # For training, large batch inference
        "energy_optimal": {},  # For edge deployment
        "throughput_optimal": {},  # For data center training
    }

    # Extract best configurations for each category
    for size, size_results in results.items():
        M, N, K = map(int, size.split("x"))

        if M <= 1024:
            category = "small_matrices"
        else:
            category = "large_matrices"

        # Find best throughput config
        if "maximize_GFLOPS" in size_results:
            templates[category] = size_results["maximize_GFLOPS"]["best_config"]

        # Find best energy config
        if "minimize_energy" in size_results:
            templates["energy_optimal"] = size_results["minimize_energy"]["best_config"]

    return templates


if __name__ == "__main__":
    print("Starting K11 Double Buffered Kernel Autotuning...")
    print("This will systematically optimize your double buffering implementation")

    # Run comprehensive autotuning
    results = run_comprehensive_autotuning()

    # Generate production-ready templates
    templates = generate_production_templates(results)

    # Save results for analysis
    with open("k11_autotuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("k11_optimized_templates.json", "w") as f:
        json.dump(templates, f, indent=2)

    print(
        "\nAutotuning complete! Check k11_autotuning_results.json for detailed results"
    )
    print("Optimized templates saved to k11_optimized_templates.json")
