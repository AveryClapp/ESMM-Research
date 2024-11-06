#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "utils.cuh"
#include <iostream>

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8

__global__ void matrixMultiply(const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < 1024) {
            for (int j = 0; j < 8; j++) {
                    float sum = 0.0f;
                          for (int k = 0; k < 32; k++) {
                                    sum += A[row * 32 + k] * B[k * 8 + j];
                                          }
                                C[row * 8 + j] = sum;
                                    }
              }

}
int main() {
  float *A, *B, *C, *C_cpu;
  float *d_A, *d_B, *d_C;

  cudaFree(0);
  A = (float*)malloc(1024 * 32 * sizeof(float));
  B = (float*)malloc(32 * 8 * sizeof(float));
  C = (float*)malloc(1024 * 8 * sizeof(float));
  C_cpu = (float*)malloc(1024 * 8 * sizeof(float));

  randomize_matrix(A, 1024, 32);
  randomize_matrix(B, 32, 8);

  cudaMalloc(&d_A, 1024 * 32 * sizeof(float));
  cudaMalloc(&d_B, 32 * 8 * sizeof(float));
  cudaMalloc(&d_C, 1024 * 8 * sizeof(float));

  cudaMemcpy(d_A, A, 1024 * 32 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, 32 * 8 * sizeof(float), cudaMemcpyHostToDevice);
  
  int threadsPerBlock = 32;
  int blocksPerGrid = (1024 + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
  cudaEventRecord(stop);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch matrixMultiply kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaEventSynchronize(stop);
  cudaMemcpy(C, d_C, 1024 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
  float time = 0.0f;
  cudaEventElapsedTime(&time, start, stop);

  for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 8; j++) {
      C_cpu[i * 8 + j] = 0.0f; // Initialize C_cpu[i][j] to zero
      for (int k = 0; k < 32; k++) {
        C_cpu[i * 8 + j] += A[i * 32 + k] * B[k * 8 + j];
      }
    }
  }

  bool correct = true;
  for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 8; j++) {
      if (fabs(C[i * 8 + j] - C_cpu[i * 8 + j]) > 1e-4) { 
        std::cout << C[i * 8 + j] << C_cpu[i * 8 + j] << std::endl;
        correct = false; // Matrices are not equal
        break; // Exit the inner loop if a discrepancy is found
      }
    }
    if (!correct) break; // Exit the outer loop if a discrepancy is found
  }

  if (correct) {
    std::cout << "Matrix multiplication is correct!" << std::endl;
  } else {
    std::cout << "Matrix multiplication is incorrect!" << std::endl;
  }

  std::cout << "GPU Timing: " << time << " ms" << std::endl;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
