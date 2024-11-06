#include <iostream>
#include <map>
#include "utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8


__global__ void kernel11111111(float*A, float*B, float* C) {
  int b_row = threadIdx.x + blockIdx.x * blockDim.x;
  if (b_row < B_ROWS) {
    float b_elements[8] = {
      B[b_row * B_COLS + 0],
      B[b_row * B_COLS + 1],
      B[b_row * B_COLS + 2],
      B[b_row * B_COLS + 3],
      B[b_row * B_COLS + 4],
      B[b_row * B_COLS + 5],
      B[b_row * B_COLS + 6],
      B[b_row * B_COLS + 7]
    };
    for (int a_row = 0; a_row < A_ROWS; a_row++) {
      float a_element = A[a_row * A_COLS + b_row];
      atomicAdd(&C[a_row * C_COLS + 0], a_element * b_elements[0]);
      atomicAdd(&C[a_row * C_COLS + 1], a_element * b_elements[1]);
      atomicAdd(&C[a_row * C_COLS + 2], a_element * b_elements[2]);
      atomicAdd(&C[a_row * C_COLS + 3], a_element * b_elements[3]);
      atomicAdd(&C[a_row * C_COLS + 4], a_element * b_elements[4]);
      atomicAdd(&C[a_row * C_COLS + 5], a_element * b_elements[5]);
      atomicAdd(&C[a_row * C_COLS + 6], a_element * b_elements[6]);
      atomicAdd(&C[a_row * C_COLS + 7], a_element * b_elements[7]);
    }
  }
}

__global__ void kernel10101010(float*A, float*B, float* C) {
  int b_row = threadIdx.x + blockIdx.x * blockDim.x;
  if (b_row < B_ROWS) {
    float b_elements[8] = {
      B[b_row * B_COLS + 0],
      B[b_row * B_COLS + 2],
      B[b_row * B_COLS + 4],
      B[b_row * B_COLS + 6],
    };
    for (int a_row = 0; a_row < A_ROWS; a_row++) {
      float a_element = A[a_row * A_COLS + b_row];
      atomicAdd(&C[a_row * C_COLS + 0], a_element * b_elements[0]);
      atomicAdd(&C[a_row * C_COLS + 2], a_element * b_elements[2]);
      atomicAdd(&C[a_row * C_COLS + 4], a_element * b_elements[4]);
      atomicAdd(&C[a_row * C_COLS + 6], a_element * b_elements[6]);
    }
  }
}

__global__ void kernel00000010(float*A, float*B, float* C) {
  int b_row = threadIdx.x + blockIdx.x * blockDim.x;
  if (b_row < B_ROWS) {
    float b_elements[8] = {
      B[b_row * B_COLS + 6],
    };
    for (int a_row = 0; a_row < A_ROWS; a_row++) {
      float a_element = A[a_row * A_COLS + b_row];
      atomicAdd(&C[a_row * C_COLS + 6], a_element * b_elements[6]);
    }
  }
}


void cpuMatrixMultiply(float* A, float* B, float* C) {
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

bool verify_results(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-4) {
  for (int i = 0; i < size; i++) {
    if (fabs(gpuResult[i] - cpuResult[i]) > tolerance) {
      printf("Mismatch at position %d: GPU = %f, CPU = %f\n", 
          i, gpuResult[i], cpuResult[i]);
      return false;
    }
  }
  return true;
}

int main() {
  std::vector<std::string> patterns = {"11111111", "10101010", "00000010"};
  typedef void (*Kernel)(float*,float*,float*);
  std::map<std::string, Kernel> kernelMap = {
    {"11111111", kernel11111111},
    {"10101010", kernel10101010},
    {"00000010", kernel00000010} 
  };

  float *h_A = new float[A_ROWS * A_COLS];
  float *h_B = new float[B_ROWS * B_COLS];
  float *h_C = new float[C_ROWS * C_COLS];

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A_ROWS * A_COLS * sizeof(float));
  cudaMalloc(&d_B, B_ROWS * B_COLS * sizeof(float));
  cudaMalloc(&d_C, C_ROWS * C_COLS * sizeof(float));
  cudaFree(0);
  for (size_t p = 0; p < patterns.size(); ++p) {
    std::string pattern = patterns[p];
    std::vector<int> nzColumns = stringToVector(pattern);

    // Randomly generate A and B (B with sparsity)
    randomize_matrix(h_A, 1024, 32);
    randomize_matrix_with_pattern(h_B, 32, 8, nzColumns);
    memset(h_C, 0, C_ROWS * C_COLS * sizeof(float));

    cudaMemcpy(d_A, h_A, A_ROWS * A_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_ROWS * B_COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, C_ROWS * C_COLS * sizeof(float), cudaMemcpyHostToDevice);

    Kernel kernel = kernelMap[pattern];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 32;
    int blocksPerGrid = CEIL_DIV(B_ROWS, threadsPerBlock);

    cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, C_ROWS * C_COLS * sizeof(float), cudaMemcpyDeviceToHost);

    float* h_C_verify = new float[C_ROWS * C_COLS];
    cpuMatrixMultiply(h_A, h_B, h_C_verify);

    bool passed = verify_results(h_C, h_C_verify, C_ROWS * C_COLS);

    printf("Pattern: %s\n", pattern.c_str());
    printf("Time: %f ms\n", milliseconds);
    printf("Verification: %s\n\n", passed ? "PASSED" : "FAILED");

    delete[] h_C_verify;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

