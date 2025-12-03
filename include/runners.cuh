#pragma once
#include "utils.cuh"
#include "metadata.cuh"
#include "../old_kernels/1D_Blocktiling.cu"
#include "../old_kernels/2D_Blocktiling.cu"
#include "../old_kernels/basic.cu"
#include "../old_kernels/gmem_coalesce.cu"
#include "../old_kernels/smem_blocking.cu"
#include "../old_kernels/vectorized_blocktiling.cu"
#include "../old_kernels/warptiling.cu"
#include "../old_kernels/1d_warptiling.cu"
#include "../old_kernels/1d_warptiling_tm.cu"
#include "../old_kernels/esmm_warpskipping.cu"
#include "../old_kernels/esmm_buffered.cu"
#include "../old_kernels/1D_vec.cu"
#include "../src/kernels/esmm.cu"
#include "../src/kernels/esmm_offsets.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_8.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_6.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_4.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_2.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_1.cu"
#include "../src/kernels/esmm_a_sparse.cu"
#include "../src/kernels/esmm_b_sparse_warp.cu"
#include "../src/kernels/esmm_b_sparse_tn.cu"
#include "../src/kernels/esmm_b_fp_lut.cu"
#include "../src/kernels/esmm_b_transpose_k19.cu"
#include "../src/kernels/esmm_b_smem_transpose.cu"
#include "../src/kernels/esmm_btranspose.cu"
#include "../old_kernels/esmm_joint_precomputed.cu"
#include "../src/kernels/esmm_b_sparse_wn.cu"
#include "../src/kernels/esmm_ab_sparse.cu"
#include "../src/kernels/esmm_ab_turbo.cu"
#include "../src/kernels/esmm_ab_sparse_optimized.cu"
#include "../src/kernels/esmm_ab_8x32.cu"
#include "../src/kernels/esmm_ab_32x32.cu"
#include "../src/kernels/esmm_joint_skip_experiments.cu"
#include "../old_kernels/esmm_joint_1d.cu"
#include "../src/preprocessors/a_preprocessor_rowlevel.cu"
#include "../src/preprocessors/joint_preprocessor.cu"
#include "../src/preprocessors/a_preprocessor_hybrid.cu"
#include "../src/preprocessors/ab_preprocessor.cu"
#include "preprocess_params.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string_view>

// ============================================================================
// PERFORMANCE-ONLY FUNCTIONS (NO RESULT CHECKING)
// ============================================================================

bool run_naive(int rows, int cols, int inners, float *d_A, float *d_B,
               float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  
  for (int i = 0; i < runs; i++) {
    basic<<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_gmem_coalesce(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  
  for (int i = 0; i < runs; i++) {
    gmem_coalesce<32><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_smem_blocking(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  
  for (int i = 0; i < runs; i++) {
    smem_blocking<32><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_one_blocktiling(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BN * BM / TM);
  
  for (int i = 0; i < runs; i++) {
    one_blocktiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_two_blocktiling(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  
  for (int i = 0; i < runs; i++) {
    two_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

// ============================================================================
// FUNCTIONS WITH RESULT CHECKING
// ============================================================================

bool run_vectorized(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  
  for (int i = 0; i < runs; i++) {
    vectorized_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_1d_vec(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, float *h_C, float *h_C_ref, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 16;
  constexpr int TN = 1;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  
  for (int i = 0; i < runs; i++) {
    one_d_vec<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_warptiling(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 32;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    one_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_1d_warptiling(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 32;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    one_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_esmm(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 32;
  const uint K10_WNITER = 4;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_esmm_warpskipping(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 8;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_warpskipping<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_esmm_buffered(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 8;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_buffered<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_esmm_b_fp_lut_offsets(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs,
                    std::string_view pattern) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;  // 2× larger
  const uint BM = 64;  // 2× larger  
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  /* Build pattern byte from sparsity string */
  uint8_t h_pattern = computeExpandedIndicesBits(pattern);

  // Allocate and copy pattern to device
  uint8_t* d_pattern;
  cudaMalloc(&d_pattern, sizeof(uint8_t));
  cudaMemcpy(d_pattern, &h_pattern, sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Launch kernel with single pattern byte (SIZE parameter no longer meaningful, but kept for template compatibility)
  esmm_b_fp_lut<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
      <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, d_pattern);

  // Free device pattern memory
  cudaFree(d_pattern);

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_esmm_unrolled(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs, 
                    std::string_view pattern) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  /* Build list based on sparsity string */
  auto sparsity_list = computeExpandedIndices(pattern);
  const int SIZE = sparsity_list.size();

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

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

void run_cuBlas(int rows, int cols, int inners, float *d_A, float *d_B,
                 float *d_C, float *h_C, int runs) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < runs; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, 
                &alpha, d_B, cols, d_A, inners, &beta, d_C, cols);
  }
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
}



// ============================================================================
// PERFORMANCE-ONLY VERSIONS (NO RESULT CHECKING)
// ============================================================================

bool run_vectorized_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                             float *d_C, int runs) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  
  for (int i = 0; i < runs; i++) {
    vectorized_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_1d_vec_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int TM = 16;
  constexpr int TN = 1;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  
  for (int i = 0; i < runs; i++) {
    one_d_vec<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_warptiling_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                             float *d_C, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 32;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    one_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_1d_warptiling_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                                float *d_C, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 32;
  const uint K10_WN = 32;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    one_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_esmm_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                      float *d_C, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 32;
  const uint K10_WNITER = 4;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_esmm_warpskipping_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                                   float *d_C, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 8;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_warpskipping<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

bool run_esmm_buffered_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                                float *d_C, int runs) {
  const uint K10_NUM_THREADS = 256;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 8;
  const uint K10_TN = 8;
  const uint K10_TM = 1;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_buffered<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }
  cudaDeviceSynchronize();
  return true;
}

void run_cuBlas_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  
  for (int i = 0; i < runs; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, 
                &alpha, d_B, cols, d_A, inners, &beta, d_C, cols);
  }
  cudaDeviceSynchronize();
  cublasDestroy(handle);
}

bool run_esmm_offsets_no_check(int rows, int cols, int inners, float *d_A,
                          float *d_B, float *d_C, int runs,
                          std::string_view pattern) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  /* Build pattern byte from sparsity string */
  uint8_t h_pattern = computeExpandedIndicesBits(pattern);

  // Allocate and copy pattern to device
  uint8_t* d_pattern;
  cudaMalloc(&d_pattern, sizeof(uint8_t));
  cudaMemcpy(d_pattern, &h_pattern, sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Time kernel execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_b_fp_lut<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, d_pattern);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 13 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  cudaFree(d_pattern);
  return true;
}

bool run_esmm_unrolled_no_check(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, int runs, 
                    std::string_view pattern) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  /* Build list based on sparsity string */
  auto sparsity_list = computeExpandedIndices(pattern);
  const int SIZE = sparsity_list.size();

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

  cudaDeviceSynchronize();
  return true;
}

bool run_esmm_a_sparse_blockwise(int rows, int cols, int inners, float *d_A, float *d_B,
                     float *d_C, float *h_C, float *h_C_ref, int runs) {
  // BEST CONFIG from tuner: 5.488 ms (4096x4096, 50% sparse)
  const uint NUM_THREADS = 128;
  const uint BM = 64;
  const uint BN = 128;
  const uint BK = 8;
  const uint WM = 32;  // ✓ Matches preprocessor
  const uint WN = 64;
  const uint WNITER = 2;
  const uint TM = 1;
  const uint TN = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  for (int i = 0; i < runs; i++) {
    esmm_a_sparse_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_block_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_block_pattern_metadata(meta);
  return passed;
}



bool run_esmm_a_sparse_blockwise_no_check(int rows, int cols, int inners, float *d_A,
                               float *d_B, float *d_C, int runs) {
  // BEST CONFIG from tuner: 5.488 ms (4096x4096, 50% sparse)
  const uint NUM_THREADS = 128;
  const uint BM = 64;
  const uint BN = 128;
  const uint BK = 8;
  const uint WM = 32;  // ✓ Matches preprocessor
  const uint WN = 64;
  const uint WNITER = 2;
  const uint TM = 1;
  const uint TN = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_a_sparse_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel time: %.3f ms (avg: %.3f ms)\n", milliseconds, milliseconds / runs);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  free_block_pattern_metadata(meta);
  return true;
}

bool run_esmm_b_fp_lut(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs, std::string_view pattern) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Convert pattern string to uint8_t
  uint8_t pattern_byte = 0;
  for (size_t i = 0; i < pattern.size() && i < 8; i++) {
    if (pattern[i] == '1') {
      pattern_byte |= (1 << (7 - i));
    }
  }

  // Allocate device memory for pattern
  uint8_t* d_pattern;
  cudaMalloc(&d_pattern, sizeof(uint8_t));
  cudaMemcpy(d_pattern, &pattern_byte, sizeof(uint8_t), cudaMemcpyHostToDevice);

  for (int i = 0; i < runs; i++) {
    esmm_b_fp_lut<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 8>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, d_pattern);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(d_pattern);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  cudaFree(d_pattern);
  return passed;
}

bool run_esmm_b_fp_lut_no_check(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, int runs, std::string_view pattern) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Convert pattern string to uint8_t
  uint8_t pattern_byte = 0;
  for (size_t i = 0; i < pattern.size() && i < 8; i++) {
    if (pattern[i] == '1') {
      pattern_byte |= (1 << (7 - i));
    }
  }

  // Allocate device memory for pattern
  uint8_t* d_pattern;
  cudaMalloc(&d_pattern, sizeof(uint8_t));
  cudaMemcpy(d_pattern, &pattern_byte, sizeof(uint8_t), cudaMemcpyHostToDevice);

  for (int i = 0; i < runs; i++) {
    esmm_b_fp_lut<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 8>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, d_pattern);
  }
  cudaDeviceSynchronize();

  cudaFree(d_pattern);
  return true;
}

bool run_esmm_b_sparse_warp(int rows, int cols, int inners, float *d_A, float *d_B,
                            float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B matrix with warp-granularity patterns
  BMatrixPatternMetadata meta = analyze_b_sparsity_warp_granularity(d_B, inners, cols, BK, WN);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse_warp<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_b_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_b_pattern_metadata(meta);
  return passed;
}

bool run_esmm_b_sparse_warp_no_check(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B matrix with warp-granularity patterns
  BMatrixPatternMetadata meta = analyze_b_sparsity_warp_granularity(d_B, inners, cols, BK, WN);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse_warp<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaDeviceSynchronize();

  free_b_pattern_metadata(meta);
  return true;
}

// ============================================================================
// B-Sparse TN-Granularity (8-column patterns per thread group)
// ============================================================================

bool run_esmm_b_sparse_tn(int rows, int cols, int inners, float *d_A, float *d_B,
                          float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B matrix with TN-granularity patterns
  BMatrixPatternMetadata meta = analyze_b_sparsity_tn_granularity(d_B, inners, cols, BK, TN);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse_tn<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_b_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_b_pattern_metadata(meta);
  return passed;
}

bool run_esmm_b_sparse_tn_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B matrix with TN-granularity patterns
  BMatrixPatternMetadata meta = analyze_b_sparsity_tn_granularity(d_B, inners, cols, BK, TN);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse_tn<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaDeviceSynchronize();

  free_b_pattern_metadata(meta);
  return true;
}




// ============================================================================
// K19: B-Transpose Sparse (WN-granularity)
// ============================================================================

bool run_esmm_b_transpose_k19(int rows, int cols, int inners, float *d_A, float *d_B,
                               float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: transpose B and analyze patterns
  BTransposeMetadata meta = preprocess_b_transpose_simple(d_B, inners, cols, BK, WN);

  for (int i = 0; i < runs; i++) {
    esmm_b_transpose_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, meta.d_B_T, d_C,
                                meta.d_rowPatterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_b_transpose_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_b_transpose_metadata(meta);
  return passed;
}

bool run_esmm_b_transpose_k19_no_check(int rows, int cols, int inners, float *d_A,
                                        float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: transpose B and analyze patterns
  BTransposeMetadata meta = preprocess_b_transpose_simple(d_B, inners, cols, BK, WN);

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_b_transpose_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, meta.d_B_T, d_C,
                                meta.d_rowPatterns, meta.numKBlocks);
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
    free_b_transpose_metadata(meta);
    return false;
  }

  free_b_transpose_metadata(meta);
  return true;
}

// ============================================================================
// K20: B-Sparse with Shared Memory Transpose (WN-granularity)
// ============================================================================

bool run_esmm_b_smem_transpose(int rows, int cols, int inners, float *d_A, float *d_B,
                                float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze B column patterns
  BColumnPatternMetadata meta = analyze_b_column_patterns_gpu(d_B, inners, cols, WN, BK);

  for (int i = 0; i < runs; i++) {
    esmm_b_smem_transpose<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_colPatterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_b_column_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_b_column_pattern_metadata(meta);
  return passed;
}

bool run_esmm_b_smem_transpose_no_check(int rows, int cols, int inners, float *d_A,
                                         float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze B column patterns
  BColumnPatternMetadata meta = analyze_b_column_patterns_gpu(d_B, inners, cols, WN, BK);

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_b_smem_transpose<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_colPatterns, meta.numKBlocks);
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
    free_b_column_pattern_metadata(meta);
    return false;
  }

  free_b_column_pattern_metadata(meta);
  return true;
}

// ============================================================================
// B-Sparse Warp-Uniform Pattern (K21)
// ============================================================================

bool run_esmm_b_sparse(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze B patterns at WN granularity
  BPatternMetadata meta = preprocess_b_patterns<BK, WN>(d_B, inners, cols);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_b_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_b_pattern_metadata(meta);
  return passed;
}

bool run_esmm_b_sparse_no_check(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze B patterns at WN granularity
  BPatternMetadata meta = preprocess_b_patterns<BK, WN>(d_B, inners, cols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_b_sparse<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_patterns, meta.numKBlocks);
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
    free_b_pattern_metadata(meta);
    return false;
  }

  free_b_pattern_metadata(meta);
  return true;
}

// ============================================================================
// AB-Sparse Joint (K22)
// ============================================================================

bool run_esmm_ab_sparse(int rows, int cols, int inners, float *d_A, float *d_B,
                        float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns
  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  for (int i = 0; i < runs; i++) {
    esmm_ab_sparse_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_esmm_ab_sparse_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns
  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_ab_sparse_v2<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
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
    free_ab_pattern_metadata(meta);
    return false;
  }

  free_ab_pattern_metadata(meta);
  return true;
}

// ============================================================================
// AB-Turbo (K23): Precomputed joint patterns with warp shuffle
// ============================================================================

bool run_esmm_ab_turbo(int rows, int cols, int inners, float *d_A, float *d_B,
                        float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns, then compute joint patterns
  ABPatternMetadata ab_meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);
  JointPatternData joint_data = preprocess_joint_patterns<BK, WM, WN>(
      ab_meta.d_a_patterns, ab_meta.d_b_patterns, rows, cols, inners);

  for (int i = 0; i < runs; i++) {
    esmm_ab_turbo<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_data.d_joint_patterns, joint_data.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_joint_pattern_data(joint_data);
    free_ab_pattern_metadata(ab_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_joint_pattern_data(joint_data);
  free_ab_pattern_metadata(ab_meta);
  return passed;
}

bool run_esmm_ab_turbo_no_check(int rows, int cols, int inners, float *d_A,
                                  float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns, then compute joint patterns
  ABPatternMetadata ab_meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);
  JointPatternData joint_data = preprocess_joint_patterns<BK, WM, WN>(
      ab_meta.d_a_patterns, ab_meta.d_b_patterns, rows, cols, inners);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_ab_turbo<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_data.d_joint_patterns, joint_data.numKBlocks);
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
    free_joint_pattern_data(joint_data);
    free_ab_pattern_metadata(ab_meta);
    return false;
  }

  free_joint_pattern_data(joint_data);
  free_ab_pattern_metadata(ab_meta);
  return true;
}

// ============================================================================
// Kernel 24: ESMM A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop)
// ============================================================================

bool run_esmm_ab_sparse_optimized(int rows, int cols, int inners, float *d_A, float *d_B,
                                    float *d_C, float *h_C, float *h_C_ref, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns
  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  for (int i = 0; i < runs; i++) {
    esmm_ab_sparse_optimized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_esmm_ab_sparse_optimized_no_check(int rows, int cols, int inners, float *d_A,
                                             float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess: analyze both A and B patterns
  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_ab_sparse_optimized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
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
    free_ab_pattern_metadata(meta);
    return false;
  }

  free_ab_pattern_metadata(meta);
  return true;
}

// ============================================================================
// JOINT SKIP EXPERIMENTS (K25-K27)
// ============================================================================

// K25: BASELINE - Dense computation (no joint sparsity)
bool run_joint_baseline(int rows, int cols, int inners, float *d_A,
                        float *d_B, float *d_C, float *h_C, float *h_C_ref,
                        int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_joint_baseline<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);
  return passed;
}

bool run_joint_baseline_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_joint_baseline<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
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

  return true;
}

// K26: SKIP_ONLY - Check patterns and skip, but NO FMA
bool run_joint_skip_only(int rows, int cols, int inners, float *d_A,
                         float *d_B, float *d_C, float *h_C, float *h_C_ref,
                         int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  uint8_t* d_joint_patterns;
  size_t totalSize = (size_t)meta.numMBlocks * meta.numNBlocks * meta.numKBlocks;
  cudaMalloc(&d_joint_patterns, totalSize * sizeof(uint8_t));

  dim3 preprocBlock(256);
  dim3 preprocGrid(meta.numNBlocks, meta.numMBlocks, CEIL_DIV(meta.numKBlocks, 256));
  preprocess_joint_skip_kernel<<<preprocGrid, preprocBlock>>>(
      meta.d_a_patterns, meta.d_b_patterns, d_joint_patterns,
      meta.numMBlocks, meta.numNBlocks, meta.numKBlocks);
  cudaDeviceSynchronize();

  for (int i = 0; i < runs; i++) {
    esmm_joint_skip_only<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_joint_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(d_joint_patterns);
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  cudaFree(d_joint_patterns);
  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_joint_skip_only_no_check(int rows, int cols, int inners, float *d_A,
                                    float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  uint8_t* d_joint_patterns;
  size_t totalSize = (size_t)meta.numMBlocks * meta.numNBlocks * meta.numKBlocks;
  cudaMalloc(&d_joint_patterns, totalSize * sizeof(uint8_t));

  dim3 preprocBlock(256);
  dim3 preprocGrid(meta.numNBlocks, meta.numMBlocks, CEIL_DIV(meta.numKBlocks, 256));
  preprocess_joint_skip_kernel<<<preprocGrid, preprocBlock>>>(
      meta.d_a_patterns, meta.d_b_patterns, d_joint_patterns,
      meta.numMBlocks, meta.numNBlocks, meta.numKBlocks);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_joint_skip_only<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_joint_patterns, meta.numKBlocks);
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
    cudaFree(d_joint_patterns);
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaFree(d_joint_patterns);
  free_ab_pattern_metadata(meta);
  return true;
}

// K27: SKIP_FMA - Full joint sparse (check + skip + FMA)
bool run_joint_skip_fma(int rows, int cols, int inners, float *d_A,
                        float *d_B, float *d_C, float *h_C, float *h_C_ref,
                        int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  uint8_t* d_joint_patterns;
  size_t totalSize = (size_t)meta.numMBlocks * meta.numNBlocks * meta.numKBlocks;
  cudaMalloc(&d_joint_patterns, totalSize * sizeof(uint8_t));

  dim3 preprocBlock(256);
  dim3 preprocGrid(meta.numNBlocks, meta.numMBlocks, CEIL_DIV(meta.numKBlocks, 256));
  preprocess_joint_skip_kernel<<<preprocGrid, preprocBlock>>>(
      meta.d_a_patterns, meta.d_b_patterns, d_joint_patterns,
      meta.numMBlocks, meta.numNBlocks, meta.numKBlocks);
  cudaDeviceSynchronize();

  for (int i = 0; i < runs; i++) {
    esmm_joint_skip_fma<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_joint_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(d_joint_patterns);
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  cudaFree(d_joint_patterns);
  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_joint_skip_fma_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs) {
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
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  ABPatternMetadata meta = preprocess_ab_patterns<BK, WM, WN>(d_A, d_B, rows, cols, inners);

  uint8_t* d_joint_patterns;
  size_t totalSize = (size_t)meta.numMBlocks * meta.numNBlocks * meta.numKBlocks;
  cudaMalloc(&d_joint_patterns, totalSize * sizeof(uint8_t));

  dim3 preprocBlock(256);
  dim3 preprocGrid(meta.numNBlocks, meta.numMBlocks, CEIL_DIV(meta.numKBlocks, 256));
  preprocess_joint_skip_kernel<<<preprocGrid, preprocBlock>>>(
      meta.d_a_patterns, meta.d_b_patterns, d_joint_patterns,
      meta.numMBlocks, meta.numNBlocks, meta.numKBlocks);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_joint_skip_fma<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                d_joint_patterns, meta.numKBlocks);
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
    cudaFree(d_joint_patterns);
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaFree(d_joint_patterns);
  free_ab_pattern_metadata(meta);
  return true;
}

// ============================================================================
// K28: ESMM A+B Sparse - 8×32 GRANULARITY
// ============================================================================

bool run_esmm_ab_8x32(int rows, int cols, int inners, float *d_A, float *d_B,
                      float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 64;   // Changed from 32 for 8×32 granularity
  const uint WM = 32;   // Changed from 64 for 8×32 granularity
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess with 8×32 granularity (8-row × 32-column tiles)
  ABPatternMetadata meta = preprocess_ab_patterns_8x32<BK, 8, WN>(d_A, d_B, rows, cols, inners);

  for (int i = 0; i < runs; i++) {
    esmm_ab_8x32<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_esmm_ab_8x32_no_check(int rows, int cols, int inners, float *d_A,
                                float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 64;   // Changed from 32 for 8×32 granularity
  const uint WM = 32;   // Changed from 64 for 8×32 granularity
  const uint WNITER = 2;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess with 8×32 granularity (8-row × 32-column tiles)
  ABPatternMetadata meta = preprocess_ab_patterns_8x32<BK, 8, WN>(d_A, d_B, rows, cols, inners);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_ab_8x32<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
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
    free_ab_pattern_metadata(meta);
    return false;
  }

  free_ab_pattern_metadata(meta);
  return true;
}

// ============================================================================
// K29: ESMM A+B Sparse - 32×32 GRANULARITY
// ============================================================================

bool run_esmm_ab_32x32(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;   // Keep same as K24
  const uint WM = 64;   // Keep same as K24
  const uint WNITER = 2;  // Keep same as K24 → gives WMITER=2 (32×16 sub-tiles)
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess with 32×32 granularity (32-row × 32-column tiles)
  ABPatternMetadata meta = preprocess_ab_patterns_32x32<BK, 32, WN>(d_A, d_B, rows, cols, inners);

  for (int i = 0; i < runs; i++) {
    esmm_ab_32x32<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_ab_pattern_metadata(meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_ab_pattern_metadata(meta);
  return passed;
}

bool run_esmm_ab_32x32_no_check(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 64;
  const uint BK = 8;
  const uint WN = 32;   // Keep same as K24
  const uint WM = 64;   // Keep same as K24
  const uint WNITER = 2;  // Keep same as K24 → gives WMITER=2 (32×16 sub-tiles)
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess with 32×32 granularity (32-row × 32-column tiles)
  ABPatternMetadata meta = preprocess_ab_patterns_32x32<BK, 32, WN>(d_A, d_B, rows, cols, inners);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < runs; i++) {
    esmm_ab_32x32<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_a_patterns, meta.d_b_patterns, meta.numKBlocks);
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
    free_ab_pattern_metadata(meta);
    return false;
  }

  free_ab_pattern_metadata(meta);
  return true;
}
