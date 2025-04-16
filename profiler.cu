#include "./kernels/1d_warptiling.cu"
#include "utils.cuh"
#include <chrono>
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
const uint K10_BM = 256;
const uint K10_BK = 64;
const uint K10_WN = 32;
const uint K10_WM = 256;
const uint K10_WNITER = 8;
const uint K10_TN = 8;
const uint K10_TM = 1;

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
  float time = 0.0f;
  float totalTime = 0.0f;

  for (int i = 0; i < runs; i++) {
    cudaEventRecord(start);
    one_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
	cudaEventRecord(stop);
    cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&time, start, stop);
	totalTime += time;
  }
  
  cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
	//return false;
	std::cout << "FAIL" << std::endl;
  } else {
	bool success = verifyResults(h_C, h_C_ref, rows * cols);
	if (!success) {
		std::cout << "FAIL" << std::endl;
		return false;
	}
	float avg_time = totalTime / runs;
	std::cout << time << " ms" << std::endl;
	return true;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
  // Setup
  constexpr int rows = 1024;
  constexpr int cols = 1024;
  constexpr int inners = 1024;
  int runs = 20;

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
  matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);

  // Initialize d_C to zeros
  cudaCheckError(cudaMemset(d_C, 0, rows * cols * sizeof(float)));
  run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 1;
}
