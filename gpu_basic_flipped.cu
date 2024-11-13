#include <cuda.h>
#include "utils.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                            \
  do {                                                              \
    cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
          cudaGetErrorString(error), error, __FILE__, __LINE__);    \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// Matrix dimensions
const int A_ROWS = 1024;
const int A_COLS = 32;
const int B_ROWS = 32;
const int B_COLS = 8;
const int C_ROWS = A_ROWS;
const int C_COLS = B_COLS;

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int A_COLS, int B_COLS, int C_COLS) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  float sum = 0.0f;
  for(int k = 0; k < A_COLS; ++k) {
    float a_val = A[row * A_COLS + k];     
    float b_val = B[k * B_COLS + col];    
    sum += a_val * b_val;                
  }
  C[row * C_COLS + col] = sum;
}

void matrixMultiplyCPU(float*A, float*B, float*C) {
  for (int row = 0; row < A_ROWS; row++) {
    for (int col = 0; col < B_COLS; col++) {
      float sum = 0.0f;
      for (int i = 0; i < A_COLS; i++) {
        sum += A[row * A_COLS + i] * B[i * B_COLS + col];
      }
      C[row * C_COLS + col] = sum;
    }
  }
}

bool verifyResults(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-5) {
  for (int i = 0; i < size; i++) {
    if (fabs(gpuResult[i] - cpuResult[i]) > tolerance) {
      return false;
    }
  }
  return true;
}
int main() {
  size_t size_A = A_ROWS * A_COLS * sizeof(float);
  size_t size_B = B_ROWS * B_COLS * sizeof(float);
  size_t size_C = C_ROWS * C_COLS * sizeof(float);

  float *h_A = (float*)malloc(size_A);
  float *h_B = (float*)malloc(size_B);
  float *h_C = (float*)malloc(size_C);
  float *h_C_cpu = (float *)malloc(size_C);
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrices.\n");
    exit(EXIT_FAILURE);
  }

  randomize_matrix(h_A, A_ROWS, A_COLS);
  randomize_matrix(h_B, B_ROWS, B_COLS);

  for(int i = 0; i < C_ROWS * C_COLS; i++) {
    h_C[i] = 0.0f;
    h_C_cpu[i] = 0.0f;
  }

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, size_C));

  dim3 gridDim(C_ROWS, 1, 1);
  dim3 blockDim(C_COLS, 1, 1);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));
  matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A_COLS, B_COLS, C_COLS);
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  bool correct = verifyResults(h_C, h_C_cpu, C_ROWS * C_COLS);

  if(correct) {
    printf("Matrix multiplication is correct.\n");
  } else {
    printf("Matrix multiplication is incorrect.\n");
  }

  printf("Kernel execution time: %f ms\n", milliseconds);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}

