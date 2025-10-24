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
#include "./old_kernels/esmm_warpskipping.cu"
#include "./old_kernels/esmm_buffered.cu"
#include "./old_kernels/1D_vec.cu"
#include "./esmm.cu"
#include "./esmm_offsets.cu"
#include "./esmm_unrolled/esmm_unrolled_8.cu"
#include "./esmm_unrolled/esmm_unrolled_6.cu"
#include "./esmm_unrolled/esmm_unrolled_4.cu"
#include "./esmm_unrolled/esmm_unrolled_2.cu"
#include "./esmm_unrolled/esmm_unrolled_1.cu"
#include "./preprocessors/a_preprocessor.cu"
#include "./preprocessors/b_preprocessor.cu"
#include <chrono>
#include <cublas_v2.h>
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

bool run_a_preprocess(int rows, int cols, int inners, float *d_A,
                         float *d_ALIST, int *h_ALIST, int *h_ALIST_ref, int runs) {
    const uint NUM_THREADS = 256;
    const uint BN = 128;
    const uint BM = 128;
    const uint BK = 8;
    const uint WN = 64;
    const uint WM = 32;
    const uint WNITER = 4;
    const uint TN = 8;
    const uint TM = 1;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;

    const int denseListSize = (inners / BK) * (BK * WMITER + WMITER);
    const int numBlocks = CEIL_DIV(rows, BM) * CEIL_DIV(cols, BN);
    const int totalSize = numBlocks * denseListSize;
   
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

    float* h_A = (float*)malloc(rows * inners * sizeof(float));
    cudaMemcpy(h_A, d_A, rows * inners * sizeof(float), cudaMemcpyDeviceToHost);
    computeReferencePreprocessing(h_A, h_ALIST_ref, rows, inners, BM, BK, WMITER, WSUBM);
    free(h_A);

    cudaMemset(d_ALIST, 0, totalSize);

    for (int i = 0; i < runs; i++) {
        preprocess_A<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_ALIST);
    }
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(h_ALIST, d_ALIST, totalSize, cudaMemcpyDeviceToHost);
    return verifyPreprocessResults(h_ALIST, h_ALIST_ref, totalSize);

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

bool run_a_preprocess_no_check(int rows, int cols, int inners, float *d_A,
                         float *d_ALIST,int runs) {
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
  cudaMemset(d_ALIST, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    preprocess_A<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_ALIST);
  }
  cudaDeviceSynchronize();

  return true;

}


PreprocessResult preprocess_matrix_a(float* d_A, int rows, int cols, int inners) {
    const uint NUM_THREADS = 256;
    const uint BN = 128, BM = 128, BK = 8;
    const uint WN = 64, WM = 32, WNITER = 4;
    const uint TN = 8, TM = 1;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);

    const int denseListSize = (inners / BK) * (BK * WMITER + WMITER);
    const int numBlocks = CEIL_DIV(rows, BM) * CEIL_DIV(cols, BN);
    const int totalSize = numBlocks * denseListSize;

    PreprocessResult result;
    result.denseListSize = denseListSize;
    result.numBlocks = numBlocks;
    result.totalSize = totalSize;
    result.h_list = nullptr;

    // Allocate device memory
    cudaCheckError(cudaMalloc((void**)&result.d_list, totalSize));
    cudaCheckError(cudaMemset(result.d_list, 0, totalSize));

    // Run preprocessing kernel
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));

    preprocess_A<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, 1>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, result.d_list);

    cudaCheckError(cudaDeviceSynchronize());

    return result;
}

void free_preprocess_result(PreprocessResult& result) {
    if (result.d_list) cudaFree(result.d_list);
    if (result.h_list) free(result.h_list);
    result.d_list = nullptr;
    result.h_list = nullptr;
}

bool verify_preprocess_a(float* d_A, int rows, int cols, int inners, int runs) {
    const uint BM = 128, BK = 8, WMITER = 4, WSUBM = 32;
    
    printf("Computing CPU reference...\n");
    
    PreprocessResult result = preprocess_matrix_a(d_A, rows, inners, inners);
    
    result.h_list = (int*)malloc(result.totalSize);
    float* h_ALIST_ref = (int*)malloc(result.totalSize);
    
    cudaMemcpy(result.h_list, result.d_list, result.totalSize, cudaMemcpyDeviceToHost);
    
    float* h_A = (float*)malloc(rows * inners * sizeof(float));
    cudaMemcpy(h_A, d_A, rows * inners * sizeof(float), cudaMemcpyDeviceToHost);
    computeReferencePreprocessing(h_A, h_ALIST_ref, rows, inners, BM, BK, WMITER, WSUBM);
    free(h_A);
    
    if (runs > 1) {
        printf("Running GPU preprocessing %d times for timing...\n", runs);
        for (int i = 1; i < runs; i++) {
            cudaMemset(result.d_list, 0, result.totalSize);
            PreprocessResult temp = preprocess_matrix_a(d_A, rows, inners, inners);
            cudaFree(temp.d_list);
        }
    }
    
    bool passed = verifyPreprocessResults(result.h_list, h_ALIST_ref, result.totalSize);
    
    free(h_ALIST_ref);
    free_preprocess_result(result);
    
    return passed;
}
