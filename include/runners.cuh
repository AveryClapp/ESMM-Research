#pragma once
#include "utils.cuh"
#include "metadata.cuh"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_8.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_6.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_4.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_2.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_1.cu"
#include "../src/kernels/esmm_a_sparse.cu"
#include "../src/kernels/esmm_b_sparse_warp.cu"
#include "../src/preprocessors/ab_preprocessor.cu"
#include "../src/kernels/esmm_ab_sparse_optimized.cu"
#include "../src/kernels/esmm_ab_8x32.cu"
#include "../src/kernels/esmm_ab_simple_fused.cu"
#include "../src/kernels/esmm_ab_optimized_v2.cu"
#include "../src/kernels/esmm_ab_optimized_v2_baseline.cu"
#include "../src/kernels/esmm_ab_gmem_32.cu"
#include "../src/kernels/esmm_ab_optimized_v3.cu"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string_view>

// ============================================================================
// K14: A-Only Sparse (Offset List, Templated Dispatch)
// Dispatches to hand-unrolled kernels based on pattern density.
// ============================================================================
bool run_esmm_unrolled(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs,
                       std::string_view pattern, bool verify) {
  // BM=64, BN=64 verified correct; larger blocks can produce wrong results
  const uint NUM_THREADS = 128;
  const uint BN = 64;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;
  const uint WNITER = 1;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  auto sparsity_list = computeExpandedIndices(pattern);
  const int SIZE = sparsity_list.size();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    if (SIZE == 1) {
      esmm_unrolled_1<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    } else if (SIZE == 2) {
      esmm_unrolled_2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    } else if (SIZE == 4) {
      esmm_unrolled_4<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    } else if (SIZE == 6) {
      esmm_unrolled_6<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    } else if (SIZE == 8) {
      esmm_unrolled_8<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_unrolled(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs,
                       std::string_view pattern) {
  return run_esmm_unrolled(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, pattern, true);
}
bool run_esmm_unrolled_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                                float *d_C, int runs, std::string_view pattern) {
  return run_esmm_unrolled(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, pattern, false);
}

// ============================================================================
// K15: cuBLAS Baseline
// ============================================================================
void run_cuBlas(int rows, int cols, int inners, float *d_A, float *d_B,
                float *d_C, float *h_C, int runs) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners,
                &alpha, d_B, cols, d_A, inners, &beta, d_C, cols);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (h_C) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  }
  cublasDestroy(handle);
}

void run_cuBlas_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  run_cuBlas(rows, cols, inners, d_A, d_B, d_C, nullptr, runs);
}

// ============================================================================
// K16: A-Sparse Block-wise (Warp-Granularity Patterns)
// Preprocessing is inside the timed loop for end-to-end measurement.
// ============================================================================
bool run_esmm_a_sparse_blockwise(int rows, int cols, int inners, float *d_A, float *d_B,
                                  float *d_C, float *h_C, float *h_C_ref, int runs,
                                  bool verify) {
  const uint NUM_THREADS = 128;
  const uint BM = 64;
  const uint BN = 128;
  const uint BK = 8;
  const uint WM = 32;
  const uint WN = 64;
  const uint WNITER = 2;
  const uint TM = 1;
  const uint TN = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);
    esmm_a_sparse_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
    cudaDeviceSynchronize();
    free_block_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_a_sparse_blockwise(int rows, int cols, int inners, float *d_A, float *d_B,
                                  float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_a_sparse_blockwise(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_a_sparse_blockwise_no_check(int rows, int cols, int inners, float *d_A,
                                          float *d_B, float *d_C, int runs) {
  return run_esmm_a_sparse_blockwise(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K17: B-Sparse Warp-Granularity (32-col, Zero-Divergence)
// ============================================================================
bool run_esmm_b_sparse_warp(int rows, int cols, int inners, float *d_A, float *d_B,
                             float *d_C, float *h_C, float *h_C_ref, int runs,
                             bool verify) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    BMatrixPatternMetadata meta = analyze_b_sparsity_warp_granularity(d_B, inners, cols, BK, WN);
    esmm_b_sparse_warp<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
    cudaDeviceSynchronize();
    free_b_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_b_sparse_warp(int rows, int cols, int inners, float *d_A, float *d_B,
                             float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_b_sparse_warp(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_b_sparse_warp_no_check(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, int runs) {
  return run_esmm_b_sparse_warp(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K20: A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop)
// ============================================================================
bool run_esmm_ab_sparse_optimized(int rows, int cols, int inners, float *d_A, float *d_B,
                                   float *d_C, float *h_C, float *h_C_ref, int runs,
                                   bool verify) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    esmm_ab_sparse_optimized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_sparse_optimized(int rows, int cols, int inners, float *d_A, float *d_B,
                                   float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_sparse_optimized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_ab_sparse_optimized_no_check(int rows, int cols, int inners, float *d_A,
                                           float *d_B, float *d_C, int runs) {
  return run_esmm_ab_sparse_optimized(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K21: A+B Sparse - 8x32 Granularity
// ============================================================================
bool run_esmm_ab_8x32(int rows, int cols, int inners, float *d_A, float *d_B,
                      float *d_C, float *h_C, float *h_C_ref, int runs,
                      bool verify) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;
  const uint WNITER = 1;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, 8, WN>(d_A, d_B, rows, cols, inners);
    esmm_ab_8x32<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_8x32(int rows, int cols, int inners, float *d_A, float *d_B,
                      float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_8x32(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_ab_8x32_no_check(int rows, int cols, int inners, float *d_A,
                                float *d_B, float *d_C, int runs) {
  return run_esmm_ab_8x32(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K25: A+B Simple Fused (Main Contribution)
// Single-kernel-launch fused preprocessing + K20 compute kernel.
// ============================================================================
bool run_esmm_ab_simple_fused(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs,
                               bool verify) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  const int numMBlocks = rows / WM;
  const int numNBlocks = cols / WN;
  const int numKBlocks = inners / BK;

  uint8_t *d_a_patterns, *d_b_patterns;
  cudaMalloc(&d_a_patterns, numMBlocks * numKBlocks * sizeof(uint8_t));
  cudaMalloc(&d_b_patterns, numNBlocks * numKBlocks * sizeof(uint8_t));

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    preprocess_a_fused<BK, WM, 256><<<numMBlocks, 256>>>(rows, inners, d_A, d_a_patterns);
    preprocess_b_fused<BK, WN, 256><<<numNBlocks, 256>>>(inners, cols, d_B, d_b_patterns);
    esmm_ab_compute_inline<BM, BN, BK, WM, WN, WNITER, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_a_patterns, d_b_patterns, numKBlocks);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(d_a_patterns);
    cudaFree(d_b_patterns);
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = verifyResults(h_C, h_C_ref, rows * cols);
    cudaFree(d_a_patterns);
    cudaFree(d_b_patterns);
    return passed;
  }

  cudaFree(d_a_patterns);
  cudaFree(d_b_patterns);
  return true;
}

bool run_esmm_ab_simple_fused(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_simple_fused(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_ab_simple_fused_no_check(int rows, int cols, int inners, float *d_A,
                                       float *d_B, float *d_C, int runs) {
  return run_esmm_ab_simple_fused(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K26: A+B Sparse OPTIMIZED V2 (K20 + block-level skip + float4 inner loop)
// 32-row A granularity: matches the 32-row block structure in the A matrix,
// giving finer-grained skip decisions than K20's 64-row granularity.
// ============================================================================
bool run_esmm_ab_optimized_v2(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs,
                               bool verify) {
  const uint NUM_THREADS = 256;  // NUM_WARPS=8 (2 warp rows × 4 warp cols) × 32
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;            // 32-row A granularity (was 64 in K20)
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    // Dispatch to correct MAX_K_BLOCKS template for tight smem allocation
    if (meta.numKBlocks <= 128) {
      esmm_ab_optimized_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 128>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else if (meta.numKBlocks <= 256) {
      esmm_ab_optimized_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 256>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else if (meta.numKBlocks <= 512) {
      esmm_ab_optimized_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 512>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else {
      esmm_ab_optimized_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1024>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    }
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_optimized_v2(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_optimized_v2(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_ab_optimized_v2_no_check(int rows, int cols, int inners, float *d_A,
                                       float *d_B, float *d_C, int runs) {
  return run_esmm_ab_optimized_v2(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// K27: Ablation — K20 + 32-row granularity only (no block skip, no float4)
// ============================================================================
bool run_esmm_ab_optimized_v2_baseline(int rows, int cols, int inners, float *d_A, float *d_B,
                                       float *d_C, float *h_C, float *h_C_ref, int runs,
                                       bool verify) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    esmm_ab_optimized_v2_baseline<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_optimized_v2_baseline(int rows, int cols, int inners, float *d_A, float *d_B,
                                       float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_optimized_v2_baseline(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true);
}
bool run_esmm_ab_optimized_v2_baseline_no_check(int rows, int cols, int inners, float *d_A,
                                                float *d_B, float *d_C, int runs) {
  return run_esmm_ab_optimized_v2_baseline(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false);
}

// ============================================================================
// Skip Statistics Analysis (K28/K29) — computed analytically from patterns
// Three skip levels:
//   block: all warps in the CUDA block skip a K-tile (block_joint == 0)
//   warp:  one warp skips a K-tile (warp joint == 0, but block_joint != 0)
//   dot:   one K-column skipped inside the inner loop (bit 0 in joint)
// ============================================================================

static void compute_and_print_skip_stats(
    const uint8_t* d_a_patterns,
    const uint8_t* d_b_patterns,
    int numCBlocksM, int numCBlocksN, int numKBlocks,
    int NUM_WARPS_M, int NUM_WARPS_N, int BK,
    const char* label)
{
    const int numMBlocks = numCBlocksM * NUM_WARPS_M;
    const int numNBlocks = numCBlocksN * NUM_WARPS_N;
    const int NUM_WARPS  = NUM_WARPS_M * NUM_WARPS_N;

    std::vector<uint8_t> h_a(numMBlocks * numKBlocks);
    std::vector<uint8_t> h_b(numNBlocks * numKBlocks);
    cudaMemcpy(h_a.data(), d_a_patterns, h_a.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b.data(), d_b_patterns, h_b.size(), cudaMemcpyDeviceToHost);

    long long total_k       = (long long)numCBlocksM * numCBlocksN * numKBlocks;
    long long total_warp_k  = total_k * NUM_WARPS;
    long long total_dot     = total_warp_k * BK;

    long long block_skips = 0, warp_skips = 0, dot_skips = 0;

    for (int cRow = 0; cRow < numCBlocksM; ++cRow) {
        for (int cCol = 0; cCol < numCBlocksN; ++cCol) {
            for (int k = 0; k < numKBlocks; ++k) {
                uint8_t bj = 0;
                for (int wRow = 0; wRow < NUM_WARPS_M; ++wRow)
                    for (int wCol = 0; wCol < NUM_WARPS_N; ++wCol)
                        bj |= h_a[(cRow*NUM_WARPS_M+wRow)*numKBlocks+k]
                            & h_b[(cCol*NUM_WARPS_N+wCol)*numKBlocks+k];

                if (bj == 0) {
                    block_skips++;
                    warp_skips += NUM_WARPS;
                    dot_skips  += (long long)NUM_WARPS * BK;
                } else {
                    for (int wRow = 0; wRow < NUM_WARPS_M; ++wRow) {
                        for (int wCol = 0; wCol < NUM_WARPS_N; ++wCol) {
                            uint8_t j = h_a[(cRow*NUM_WARPS_M+wRow)*numKBlocks+k]
                                      & h_b[(cCol*NUM_WARPS_N+wCol)*numKBlocks+k];
                            if (j == 0) {
                                warp_skips++;
                                dot_skips += BK;
                            } else {
                                dot_skips += BK - __builtin_popcount(j);
                            }
                        }
                    }
                }
            }
        }
    }

    long long warp_beyond_block = warp_skips - block_skips * NUM_WARPS;
    long long block_k_hit = total_k - block_skips;

    printf("\n=== Skip Stats: %s ===\n", label);
    printf("Grid: %d x %d CUDA blocks, %d K-tiles, %d warps/block, BK=%d\n",
           numCBlocksM, numCBlocksN, numKBlocks, NUM_WARPS, BK);
    printf("Block-level  (all warps skip K-tile):  %lld / %lld  (%.1f%%)\n",
           block_skips, total_k, 100.0*block_skips/total_k);
    printf("Warp-level   (one warp  skips K-tile):  %lld / %lld  (%.1f%%)  "
           "[%lld beyond block skips, %.1f%% of non-block-skipped warp-K pairs]\n",
           warp_skips, total_warp_k, 100.0*warp_skips/total_warp_k,
           warp_beyond_block,
           block_k_hit>0 ? 100.0*warp_beyond_block/(block_k_hit*NUM_WARPS) : 0.0);
    printf("dotIdx-level (one col   skipped):        %lld / %lld  (%.1f%%)  "
           "[%lld executed, %.1f%%]\n\n",
           dot_skips, total_dot, 100.0*dot_skips/total_dot,
           total_dot-dot_skips, 100.0*(total_dot-dot_skips)/total_dot);
}

// ============================================================================
// K28: K25 with 32-row A granularity (gmem pattern reads, block+warp skip, float2 A-loads)
//      Templated MAX_K_BLOCKS for tight smem allocation (same dispatch as K29).
// ============================================================================
bool run_esmm_ab_gmem_32(int rows, int cols, int inners, float *d_A, float *d_B,
                          float *d_C, float *h_C, float *h_C_ref, int runs,
                          bool verify, bool print_skip_stats = false) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;
  const uint WNITER = 2;
  const uint TN = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  if (print_skip_stats) {
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    compute_and_print_skip_stats(
        meta.d_a_patterns, meta.d_b_patterns,
        CEIL_DIV(rows, BM), CEIL_DIV(cols, BN),
        meta.numKBlocks, BM/WM, BN/WN, BK, "K28");
    free_ab_pattern_metadata(meta);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    if (meta.numKBlocks <= 128) {
      esmm_ab_gmem_32<BM, BN, BK, WM, WN, WNITER, TN, NUM_THREADS, 128>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else if (meta.numKBlocks <= 256) {
      esmm_ab_gmem_32<BM, BN, BK, WM, WN, WNITER, TN, NUM_THREADS, 256>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else if (meta.numKBlocks <= 512) {
      esmm_ab_gmem_32<BM, BN, BK, WM, WN, WNITER, TN, NUM_THREADS, 512>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    } else {
      esmm_ab_gmem_32<BM, BN, BK, WM, WN, WNITER, TN, NUM_THREADS, 1024>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                  meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
    }
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_gmem_32(int rows, int cols, int inners, float *d_A, float *d_B,
                          float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_gmem_32(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true, false);
}
bool run_esmm_ab_gmem_32_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs,
                                   bool print_skip_stats = false) {
  return run_esmm_ab_gmem_32(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false, print_skip_stats);
}

// ============================================================================
// K29: K26 + templated MAX_K_BLOCKS + float2 A-loads (full thread utilization)
// ============================================================================

// Helper: dispatch to correct MAX_K_BLOCKS template instantiation
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
void dispatch_v3(dim3 gridDim, dim3 blockDim, int numKBlocks,
                 int rows, int cols, int inners,
                 float *d_A, float *d_B, float *d_C,
                 uint8_t *d_a_patterns, uint8_t *d_b_patterns) {
  if (numKBlocks <= 128) {
    esmm_ab_optimized_v3<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 128>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_a_patterns, d_b_patterns, numKBlocks);
  } else if (numKBlocks <= 256) {
    esmm_ab_optimized_v3<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 256>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_a_patterns, d_b_patterns, numKBlocks);
  } else if (numKBlocks <= 512) {
    esmm_ab_optimized_v3<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 512>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_a_patterns, d_b_patterns, numKBlocks);
  } else {
    esmm_ab_optimized_v3<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1024>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_a_patterns, d_b_patterns, numKBlocks);
  }
}

bool run_esmm_ab_optimized_v3(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs,
                               bool verify, bool print_skip_stats = false) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 32;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  if (print_skip_stats) {
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    compute_and_print_skip_stats(
        meta.d_a_patterns, meta.d_b_patterns,
        CEIL_DIV(rows, BM), CEIL_DIV(cols, BN),
        meta.numKBlocks, BM/WM, BN/WN, BK, "K29");
    free_ab_pattern_metadata(meta);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    cudaMemset(d_C, 0, rows * cols * sizeof(float));
    ABPatternMetadata meta = preprocess_ab<BK, WM, WN>(d_A, d_B, rows, cols, inners);
    dispatch_v3<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>(
        gridDim, blockDim, meta.numKBlocks,
        rows, cols, inners, d_A, d_B, d_C,
        meta.d_a_patterns, meta.d_b_patterns);
    cudaDeviceSynchronize();
    free_ab_pattern_metadata(meta);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  if (verify) {
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    return verifyResults(h_C, h_C_ref, rows * cols);
  }
  return true;
}

bool run_esmm_ab_optimized_v3(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs) {
  return run_esmm_ab_optimized_v3(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true, false);
}
bool run_esmm_ab_optimized_v3_no_check(int rows, int cols, int inners, float *d_A,
                                       float *d_B, float *d_C, int runs,
                                       bool print_skip_stats = false) {
  return run_esmm_ab_optimized_v3(rows, cols, inners, d_A, d_B, d_C, nullptr, nullptr, runs, false, print_skip_stats);
}
