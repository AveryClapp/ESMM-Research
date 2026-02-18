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
