#include "./kernels/1D_Blocktiling.cu"
#include "./kernels/2D_Blocktiling.cu"
#include "./kernels/basic.cu"
#include "./kernels/gmem_coalesce.cu"
#include "./kernels/smem_blocking.cu"
#include "./kernels/vectorized_blocktiling.cu"
#include "./kernels/warptiling.cu"
#include "./kernels/1D_vec.cu"
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

#define SETUP                                                                  \
  auto start = std::chrono::high_resolution_clock::now();                      \
  auto end = std::chrono::high_resolution_clock::now();                        \
  double total_time = 0.0f;
#define START start = std::chrono::high_resolution_clock::now();
#define END                                                                    \
  end = std::chrono::high_resolution_clock::now();                             \
  total_time +=                                                                \
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)       \
          .count();
#define RESULTS(kernel)                                                        \
  std::cout << "Average Speed of Kernel " << kernel << " (" << runs            \
            << " runs): " << std::fixed << std::setprecision(4)                \
            << (total_time / runs) / 1000.0f << " ms" << std::endl;

const uint K10_NUM_THREADS = 128;
const uint K10_BN = 128;
const uint K10_BM = 128;
const uint K10_BK = 16;
const uint K10_WN = 64;
const uint K10_WM = 64;
const uint K10_WNITER = 4;
const uint K10_TN = 4;
const uint K10_TM = 8;


void run_naive(int rows, int cols, int inners, float *d_A, float *d_B,
               float *d_C, int runs) {
  dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
  dim3 blockDim(32, 32);
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    basic<<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END cudaDeviceSynchronize();
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
    END cudaDeviceSynchronize();
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
    END cudaDeviceSynchronize();
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
    END cudaDeviceSynchronize();
  }
  RESULTS("1D Blocktiling")
}

bool run_two_blocktiling(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, float *h_C, float*  h_C_ref, int runs) {

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 1;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    two_blocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END cudaDeviceSynchronize();
  }
  RESULTS("2D Blocktiling")
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
 return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_1d_vec(int rows, int cols, int inners, float *d_A, float *d_B,
                         float *d_C, float *h_C, float *h_C_ref, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 1;
  dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  dim3 blockDim(BM * BN / (TM * TN));
  SETUP
  for (int i = 0; i < runs; i++) {
    START
    one_d_vec<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
    END 
	cudaDeviceSynchronize();
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  RESULTS("1D Vectorized Blocktiling")
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);
}

bool run_vectorized(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 1;
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
  RESULTS("2D Vectorized Blocktiling")
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, rows * cols);

}

bool run_warptiling(int rows, int cols, int inners, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {
  // Setup cuda timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));

  // Initialize C to zeros
  cudaMemset(d_C, 0, rows * cols * sizeof(float));

  for (int i = 0; i < runs; i++) {
    cudaEventRecord(start);
    warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
  }
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	//return false;
	std::cout << "FAIL" << std::endl;
  } else {
	//bool success = verifyResults(h_C, h_C_ref, rows * cols);
	//if (!success) {
		//return false;
	//}
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << time << " ms" << std::endl;

	//return true;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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
  RESULTS("cuBLAS")
  cublasDestroy(handle);
}

int main(int argc, char *argv[]) {
  // Setup
  constexpr int rows = 1024;
  constexpr int cols = 1024;
  constexpr int inners = 1024;
  int kernel_choice = 10; // Default to warptiling
  int runs = 1;          // Default number of runs

  // Parse command line arguments
  if (argc > 1) {
    kernel_choice = atoi(argv[1]);
  }

  if (argc > 2) {
    runs = atoi(argv[2]);
  }

  // Allocate host matrices
  float *h_A = (float *)malloc(rows * inners * sizeof(float));
  float *h_B = (float *)malloc(inners * cols * sizeof(float));
  float *h_C = (float *)malloc(rows * cols * sizeof(float));
  float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

  // Generate random data
  randomize_matrix(h_A, rows, inners);
  randomize_matrix(h_B, inners, cols);

  // Set h_C to zeros
  memset(h_C, 0, rows * cols * sizeof(float));

  // Allocate device matrices
  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc(&d_A, rows * inners * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_B, inners * cols * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

  // Copy random data to device matrices
  cudaCheckError(cudaMemcpy(d_A, h_A, rows * inners * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, inners * cols * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Generate reference solution on CPU
  //matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);

  // Initialize d_C to zeros
  cudaCheckError(cudaMemset(d_C, 0, rows * cols * sizeof(float)));

  bool verificationResult = true;

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
    //run_one_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
	std::cout << run_two_blocktiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs) << std::endl;
    break;
  case 5:
	std::cout << run_1d_vec(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs) << std::endl;
    break;
  case 6:
	std::cout << run_vectorized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs) << std::endl;
    break;
  case 10:
	run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    break;
  case 11:
    run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
    break;
  case 12:
    run_naive(rows, cols, inners, d_A, d_B, d_C, runs);
    run_gmem_coalesce(rows, cols, inners, d_A, d_B, d_C, runs);
    run_smem_blocking(rows, cols, inners, d_A, d_B, d_C, runs);
    run_one_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
    run_two_blocktiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
	run_1d_vec(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    //run_vectorized(rows, cols, inners, d_A, d_B, d_C, runs);
    //run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    break;
  default:
    std::cout << "Invalid kernel choice. Using warptiling (10) by default."
              << std::endl;
    break;
  }

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return verificationResult ? 0 : 1;
}
