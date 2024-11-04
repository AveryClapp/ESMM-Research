#include <cuda_runtime.h>
#include "utils.cuh"
#include <iostream>


__global__ void matrixMultiply(const float* A, const float* B, float* C) {
  int row = blockIdx.x * 32 + threadIdx.x;      
  float aVal = A[row * 32 + threadIdx.x];
  for(int j = 0; j < 8; j++) {
    C[row * 8 + j] += aVal * B[threadIdx.x * 8 + j];
  }
}

int main() {
  float *A, *B, *C;
  float *d_A, *d_B, *d_C;
  
  cudaFree(0);
  A = (float*)malloc(1024 * 32 * sizeof(float));
  B = (float*)malloc(32 * 8 * sizeof(float));
  C = (float*)malloc(1024 * 8 * sizeof(float));

  randomize_matrix(A, 1024, 32);
  randomize_matrix(B, 32, 8);

  cudaMalloc(&d_A, 1024 * 32 * sizeof(float));
  cudaMalloc(&d_B, 32 * 8 * sizeof(float));
  cudaMalloc(&d_C, 1024 * 8 * sizeof(float));

  cudaMemcpy(d_A, A, 1024 * 32 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, 32 * 8 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32, 1);
  dim3 numBlocks(32, 1);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
  cudaEventRecord(stop);

  cudaMemcpy(C, d_C, 1024 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float time = 0.0f;
  cudaEventElapsedTime(&time, start, stop);

  std::cout << "GPU Timing: " << time << " ms" << std::endl;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
