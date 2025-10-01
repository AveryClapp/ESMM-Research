#pragma once
#include "utils.cuh"
#include "./old_kernels/1D_Blocktiling.cu"
#include "./old_kernels/2D_Blocktiling.cu"
#include "./old_kernels/basic.cu"
#include "./old_kernels/gmem_coalesce.cu"
#include "./old_kernels/smem_blocking.cu"
#include "./old_kernels/vectorized_blocktiling.cu"
#include "./old_kernels/warptiling.cu"
#include "./old_kernels/1d_warptiling.cu"
#include "./old_kernels/1d_warptiling_tm.cu"
#include "./esmm.cu"
#include "./esmm_warpskipping.cu"
#include "./old_kernels/esmm_buffered.cu"
#include "./old_kernels/1D_vec.cu"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

