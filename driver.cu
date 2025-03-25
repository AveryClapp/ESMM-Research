#include "./kernels/1D_Blocktiling.cu"
#include "./kernels/2D_Blocktiling.cu"
#include "./kernels/basic.cu"
#include "./kernels/gmem_coalesce.cu"
#include "./kernels/smem_blocking.cu"
#include "./kernels/vectorized_blocktiling.cu"
#include "./kernels/warptiling.cu"
#include "utils.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define cudaCheckError(ans)                                                    \
  {                                                                            \
    cudaAssert((ans), __FILE__, __LINE__);                                     \
  }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define SETUP \
	auto start = std::chrono::high_resolution_clock::now(); \
	auto end = std::chrono::high_resolution_clock::now(); \
	double total_time = 0.0f;
#define START start = std::chrono::high_resolution_clock::now();
#define END \
	end = std::chrono::high_resolution_clock::now(); \
	total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#define RESULTS(kernel) \
	std::cout << "Average Speed of Kernel " << kernel << " (" << runs << " runs): "\
	<< std::fixed << std::setprecision(4) \
	<< (total_time / runs) / 1000.0f << " ms" << std::endl;

// Move these parameters to file scope so they can be modified by the autotuner
const uint K10_NUM_THREADS = 128;
const uint K10_BN = 256;
const uint K10_BM = 256;
const uint K10_BK = 64;
const uint K10_WN = 64;
const uint K10_WM = 256;
const uint K10_WNITER = 8;
const uint K10_TN = 4;
const uint K10_TM = 16;

void run_naive(int rows, int cols, int inners, float *d_A, float *d_B,
               float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    basic<<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  RESULTS("Naive");
}

void run_gmem_coalesce(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    gmem_coalesce<32><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  RESULTS("GMEM Coalescing");
}

void run_smem_blocking(int rows, int cols, int inners, float *d_A, float *d_B,
                       float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    smem_blocking<32><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  RESULTS("SMEM Blocking");
}

void run_one_blocktiling(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BN * BM / TM);
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    one_blocktiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  RESULTS("1D Blocktiling")
}

void run_two_blocktiling(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, int runs) {
	
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  SETUP
    START
  for (int i = 0; i < runs; i++) {
	START
    two_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  RESULTS("2D Blocktiling")
}

void run_vectorized(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, int runs) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    vectorized_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END
	cudaDeviceSynchronize();
  }
  RESULTS("Vectorized Blocktiling")
}

void run_warptiling(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, int runs) {
  // Using global constants now
  dim3 blockDim(K10_NUM_THREADS);

  // Calculate NUM_WARPS based on K10_NUM_THREADS
  //const uint NUM_WARPS = K10_NUM_THREADS / 32;

  // Calculate WMITER here
  //const uint K10_WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);

  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
  SETUP
  for (int i = 0; i < runs; i++) {
	START
	warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
  	END
  	cudaDeviceSynchronize();
  }
  RESULTS("Warptiling")
}

void run_cuBlas(int rows, int cols, int inners, float *d_A, float *d_B,
                float *d_C, float *h_C, int runs) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, &alpha, d_B,
              cols, d_A, inners, &beta, d_C, cols);

  SETUP
  for (int i = 0; i < runs; i++) {
    START
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, &alpha,
                d_B, cols, d_A, inners, &beta, d_C, cols);
    END cudaDeviceSynchronize();
  }
  cudaCheckError(cudaMemcpy(h_C, d_C, rows * cols * sizeof(float),
                            cudaMemcpyDeviceToHost));
  cudaMemset(d_C, 0, rows * cols * sizeof(float));
  RESULTS("cuBLAS")
  cublasDestroy(handle);
}

int main(int argc, char *argv[]) {
  // Setup
  constexpr int rows = 1024;
  constexpr int cols = 1024;
  constexpr int inners = 1024;
  int kernel_choice = 10; // Default to warptiling
  int runs = 10;          // Default number of runs

  // Parse command line arguments
  if (argc > 1) {
    kernel_choice = atoi(argv[1]);
  }

  if (argc > 2) {
    runs = atoi(argv[2]);
  }
  kernel_choice = 10;
  runs = 1;
  // Allocate host matrices
  float *h_A = (float *)malloc(rows * cols * sizeof(float));
  float *h_B = (float *)malloc(rows * cols * sizeof(float));
  float *h_C = (float *)malloc(rows * cols * sizeof(float));
  float *h_C_cpu = (float *)malloc(rows * cols * sizeof(float));

  // Generate random data
  randomize_matrix(h_A, rows, cols);
  randomize_matrix(h_B, rows, cols);

  // Allocate device matrices
  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc(&d_A, rows * cols * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_B, rows * cols * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

  // Copy random data to device matrices
  cudaCheckError(cudaMemcpy(d_A, h_A, rows * cols * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, rows * cols * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Run CPU matrix multiplication for reference

  // Choose kernel based on input
  switch (kernel_choice) {
  case 1:
    run_naive(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 2:
    run_gmem_coalesce(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 3:
    run_smem_blocking(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 4:
    run_one_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 5:
    run_two_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 6:
    run_vectorized(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 10:
    run_warptiling(rows, cols, inners, d_A, d_B, d_C, runs);
    break;
  case 11:
  //  run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
    break;
  default:
    std::cout << "Invalid kernel choice. Using warptiling (10) by default."
              << std::endl;
    run_warptiling(rows, cols, inners, d_A, d_B, d_C, runs);
  }

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
