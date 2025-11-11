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
#include "../src/kernels/esmm_pattern_specialized.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_8.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_6.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_4.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_2.cu"
#include "../src/kernels/unrolled_kernels/esmm_unrolled_1.cu"
#include "../src/kernels/esmm_hybrid.cu"
#include "../src/kernels/esmm_combined_opt.cu"
#include "../src/kernels/esmm_hybrid_combined.cu"
#include "../src/kernels/esmm_btranspose.cu"
#include "../src/kernels/esmm_offset_combined.cu"
#include "../src/kernels/esmm_btranspose_offset.cu"
#include "../src/kernels/esmm_joint_precomputed.cu"
#include "../src/kernels/esmm_joint_shared.cu"
#include "../src/preprocessors/a_preprocessor_rowlevel.cu"
#include "../src/preprocessors/a_preprocessor_hybrid.cu"
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

bool run_esmm_offsets(int rows, int cols, int inners, float *d_A, float *d_B,
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
  int* sparse_data;
  const int SIZE = sparsity_list.size();

  cudaMalloc(&sparse_data, SIZE * sizeof(int));
  cudaMemcpy(sparse_data, sparsity_list.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  if (SIZE == 1) {
    esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 2) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 2>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 3) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 3>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 4) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 4>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 5) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 5>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 6) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 6>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 7) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 7>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  } else if (SIZE == 8) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 8>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
  }

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
  int* sparse_data;
  const int SIZE = sparsity_list.size();

  cudaMalloc(&sparse_data, SIZE * sizeof(int));
  cudaMemcpy(sparse_data, sparsity_list.data(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

  // Time kernel execution
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    if (SIZE == 1) {
      esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
          <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 2) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 2>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 3) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 3>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 4) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 4>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 5) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 5>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 6) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 6>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 7) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 7>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    } else if (SIZE == 8) {
        esmm_offsets<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 8>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, sparse_data);
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 13 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  cudaFree(sparse_data);
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

// Pattern-specialized kernel with zero-overhead sparsity
bool run_esmm_pattern_specialized(int rows, int cols, int inners, float *d_A, float *d_B,
                                   float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;  // Pattern functions only support BK=8
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  // Run row-level preprocessing
  RowLevelMetadata meta = analyze_sparsity_pattern_rowlevel_gpu(d_A, rows, inners, BK);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_pattern_specialized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, meta.d_list);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\\n", cudaGetErrorString(error));
    free_rowlevel_metadata(meta);
    return false;
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool result = verifyResults(h_C, h_C_ref, rows * cols, 1e-3);

  printf("  Avg Time: %.3f ms | %.1f GFLOPS\\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_rowlevel_metadata(meta);
  return result;
}

bool run_esmm_pattern_specialized_no_check(int rows, int cols, int inners, float *d_A,
                                            float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;  // Pattern functions only support BK=8
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

  // Run row-level preprocessing
  RowLevelMetadata meta = analyze_sparsity_pattern_rowlevel_gpu(d_A, rows, inners, BK);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_pattern_specialized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C, meta.d_list);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\\n", cudaGetErrorString(error));
    free_rowlevel_metadata(meta);
    return false;
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;

  printf("  Avg Time: %.3f ms | %.1f GFLOPS\\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_rowlevel_metadata(meta);
  return true;
}


bool run_esmm_hybrid(int rows, int cols, int inners, float *d_A, float *d_B,
                     float *d_C, float *h_C, float *h_C_ref, int runs) {
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

  // GPU-based preprocessing - no host transfer needed!
  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Run kernel with block-wise patterns
  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
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

// K25: K17 with larger tiles for better L1 reuse
bool run_esmm_hybrid_large_no_check(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 256;  // 2× larger for better data reuse
  const uint BK = 8;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // GPU-based preprocessing - no host transfer needed!
  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Time kernel execution separately
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 25 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(meta);
  return true;
}

bool run_esmm_hybrid_no_check(int rows, int cols, int inners, float *d_A,
                               float *d_B, float *d_C, int runs) {
  // Tuned configuration from kernel_tuner (2048x2048x2048)
  // Achieved 310,689 GFLOPS on A10G
  const uint NUM_THREADS = 128;  // Tuned: was 256
  const uint BN = 128;            // Same
  const uint BM = 64;             // Tuned: was 128
  const uint BK = 8;              // Fixed
  const uint WN = 32;             // Tuned: was 64
  const uint WM = 64;             // Tuned: was 32
  const uint WNITER = 2;          // Tuned: was 4
  const uint TN = 8;              // Fixed
  const uint TM = 1;              // Fixed

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // GPU-based preprocessing - no host transfer needed!
  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Time kernel execution separately
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 17 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(meta);
  return true;
}

// ============================================================================
// COMBINED A+B SPARSE GEMM (Kernel 21)
// ============================================================================

bool run_esmm_combined(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, float *h_C, float *h_C_ref, int runs) {
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

  // Preprocess A matrix for row-level sparsity
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Preprocess B matrix for column-level sparsity
  BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, inners, cols, BK, TN);

  // Run combined kernel
  for (int i = 0; i < runs; i++) {
    esmm_combined_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns,
                                A_meta.numKBlocks, B_meta.numNBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_block_pattern_metadata(A_meta);
    free_b_pattern_metadata(B_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_block_pattern_metadata(A_meta);
  free_b_pattern_metadata(B_meta);
  return passed;
}

bool run_esmm_combined_no_check(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, int runs) {
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

  // Preprocess A matrix for row-level sparsity
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Preprocess B matrix for column-level sparsity
  BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, inners, cols, BK, TN);

  // Time kernel execution separately
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_combined_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns,
                                A_meta.numKBlocks, B_meta.numNBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 18 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(A_meta);
  free_b_pattern_metadata(B_meta);
  return true;
}

// ============================================================================
// B-TRANSPOSE SPARSE GEMM (Kernel 19)
// ============================================================================

bool run_esmm_btranspose(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;  // Flipped from 64
  const uint WM = 64;  // Flipped from 32
  const uint WNITER = 1;  // Adjusted for new tile shape
  const uint TN = 1;   // Flipped from 8
  const uint TM = 8;   // Flipped from 1

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B: transpose and analyze row-wise sparsity patterns
  BTPatternMetadata BT_meta = preprocess_b_transpose(d_B, inners, cols, WN, BK);

  // Run kernel with original B (no longer need B^T!)
  for (int i = 0; i < runs; i++) {
    esmm_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                BT_meta.d_blockPatterns, BT_meta.numKBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_bt_pattern_metadata(BT_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_bt_pattern_metadata(BT_meta);
  return passed;
}

bool run_esmm_btranspose_no_check(int rows, int cols, int inners, float *d_A,
                                   float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;  // Flipped from 64
  const uint WM = 64;  // Flipped from 32
  const uint WNITER = 1;  // Adjusted for new tile shape
  const uint TN = 1;   // Flipped from 8
  const uint TM = 8;   // Flipped from 1

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess B: transpose and analyze row-wise sparsity patterns
  BTPatternMetadata BT_meta = preprocess_b_transpose(d_B, inners, cols, WN, BK);

  // Time kernel execution separately (excluding preprocessing)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                BT_meta.d_blockPatterns, BT_meta.numKBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 20 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_bt_pattern_metadata(BT_meta);
  return true;
}




// ============================================================================
// OPTIMIZED COMBINED A+B SPARSE GEMM (Kernel 21)
// ============================================================================


bool run_esmm_combined_opt(int rows, int cols, int inners, float *d_A, float *d_B,
                            float *d_C, float *h_C, float *h_C_ref, int runs) {
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

  // Preprocess A matrix for row-level sparsity
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Preprocess B matrix for column-level sparsity
  BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, inners, cols, BK, TN);

  // Run optimized combined kernel
  for (int i = 0; i < runs; i++) {
    esmm_combined_blockwise_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns,
                                A_meta.numKBlocks, B_meta.numNBlocks);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_block_pattern_metadata(A_meta);
    free_b_pattern_metadata(B_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_block_pattern_metadata(A_meta);
  free_b_pattern_metadata(B_meta);
  return passed;
}

bool run_esmm_combined_opt_no_check(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, int runs) {
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

  // Preprocess A matrix for row-level sparsity
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Preprocess B matrix for column-level sparsity
  BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, inners, cols, BK, TN);

  // Time kernel execution separately
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_combined_blockwise_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns,
                                A_meta.numKBlocks, B_meta.numNBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 19 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(A_meta);
  free_b_pattern_metadata(B_meta);
  return true;
}

bool run_esmm_offset_combined_no_check(int rows, int cols, int inners, float *d_A,
                                        float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 64;
  const uint WM = 32;
  const uint WNITER = 8;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess A matrix for hybrid sparsity
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  // Preprocess B matrix for column-level sparsity
  BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, inners, cols, BK, TN);

  // Time kernel execution separately
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_offset_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 21 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(A_meta);
  free_b_pattern_metadata(B_meta);
  return true;
}

bool run_esmm_btranspose_offset(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, float *h_C,
                                 float *h_C_ref, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess A and B sparsity patterns
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);
  BTPatternMetadata B_meta = preprocess_b_transpose(d_B, inners, cols, WN, BK);

  // Run kernel
  for (int i = 0; i < runs; i++) {
    esmm_btranspose_offset<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_block_pattern_metadata(A_meta);
    free_bt_pattern_metadata(B_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_block_pattern_metadata(A_meta);
  free_bt_pattern_metadata(B_meta);
  return passed;
}

bool run_esmm_btranspose_offset_no_check(int rows, int cols, int inners, float *d_A,
                                          float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Preprocess A and B sparsity patterns
  BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);
  BTPatternMetadata B_meta = preprocess_b_transpose(d_B, inners, cols, WN, BK);

  // Time kernel execution separately (excluding preprocessing)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_btranspose_offset<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                A_meta.d_blockPatterns, B_meta.d_blockPatterns);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 22 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(A_meta);
  free_bt_pattern_metadata(B_meta);
  return true;
}

// ============================================================================
// Kernel 23: Joint A+B Sparsity with Precomputed Intersections
// ============================================================================

bool run_esmm_joint_precomputed(int rows, int cols, int inners, float *d_A,
                                 float *d_B, float *d_C, float *h_C,
                                 float *h_C_ref, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Precompute joint patterns
  JointPatternMetadata joint_meta = preprocess_joint_patterns(
      d_A, d_B, rows, cols, inners, BM, BN, BK, WM, WN);

  // Run kernel
  for (int i = 0; i < runs; i++) {
    esmm_joint_precomputed<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_meta.d_patterns,
                                joint_meta.numKBlocks,
                                joint_meta.numMWarps,
                                joint_meta.numNWarps);
  }

  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    free_joint_pattern_metadata(joint_meta);
    return false;
  }

  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_joint_pattern_metadata(joint_meta);
  return passed;
}

bool run_esmm_joint_precomputed_no_check(int rows, int cols, int inners, float *d_A,
                                          float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Precompute joint patterns
  JointPatternMetadata joint_meta = preprocess_joint_patterns(
      d_A, d_B, rows, cols, inners, BM, BN, BK, WM, WN);

  // Time kernel execution separately (excluding preprocessing)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_joint_precomputed<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_meta.d_patterns,
                                joint_meta.numKBlocks,
                                joint_meta.numMWarps,
                                joint_meta.numNWarps);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 23 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_joint_pattern_metadata(joint_meta);
  return true;
}

// ============================================================================
// Kernel 24: Joint A+B with Shared Memory Pattern Cache
// ============================================================================
bool run_esmm_joint_shared(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Precompute joint patterns
  JointPatternMetadata joint_meta = preprocess_joint_patterns_v2(
      d_A, d_B, rows, cols, inners, BM, BN, BK, WM, WN);

  // Time kernel execution separately (excluding preprocessing)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_joint_shared<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_meta.d_patterns,
                                joint_meta.numKBlocks,
                                joint_meta.numMWarps,
                                joint_meta.numNWarps);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 24 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  bool passed = verifyResults(h_C, h_C_ref, rows * cols);

  free_joint_pattern_metadata(joint_meta);
  return passed;
}

bool run_esmm_joint_shared_no_check(int rows, int cols, int inners, float *d_A,
                                     float *d_B, float *d_C, int runs) {
  const uint NUM_THREADS = 256;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 8;
  const uint WN = 32;
  const uint WM = 64;
  const uint WNITER = 1;
  const uint TN = 1;
  const uint TM = 8;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  // Precompute joint patterns
  JointPatternMetadata joint_meta = preprocess_joint_patterns_v2(
      d_A, d_B, rows, cols, inners, BM, BN, BK, WM, WN);

  // Time kernel execution separately (excluding preprocessing)
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_joint_shared<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                joint_meta.d_patterns,
                                joint_meta.numKBlocks,
                                joint_meta.numMWarps,
                                joint_meta.numNWarps);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 24 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_joint_pattern_metadata_v2(joint_meta);
  return true;
}

bool run_esmm_hybrid_square_no_check(int rows, int cols, int inners, float *d_A,
                                      float *d_B, float *d_C, int runs) {
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

  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C,
                                meta.d_blockPatterns, meta.numKBlocks);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double avg_time = elapsed.count() / runs;
  printf("  Kernel 26 Avg Time: %.3f ms | %.1f GFLOPS\n",
         avg_time,
         (2.0 * rows * cols * inners) / (avg_time * 1e6));

  free_block_pattern_metadata(meta);
  return true;
}

// K25 with verification
bool run_esmm_hybrid_large(int rows, int cols, int inners, float *d_A,
                           float *d_B, float *d_C, float *h_C, float *h_C_ref, int runs) {
  const uint NUM_THREADS = 512;
  const uint BN = 256;
  const uint BM = 256;
  const uint BK = 8;
  const uint WN = 128;
  const uint WM = 32;
  const uint WNITER = 4;
  const uint TN = 8;
  const uint TM = 1;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
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

// K26 with verification
bool run_esmm_hybrid_square(int rows, int cols, int inners, float *d_A,
                            float *d_B, float *d_C, float *h_C, float *h_C_ref, int runs) {
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

  BlockPatternMetadata meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);

  for (int i = 0; i < runs; i++) {
    esmm_hybrid_blockwise<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
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
