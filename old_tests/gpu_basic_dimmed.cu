#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8
/*
In order for us not to use atomicadd, we need to make sure that indices in C are being written to one at a time and
there are no race conditions. One way to do this is to have one thread be purely responsible for one index in C, 
which would require the thread to go across a row of A and down a col of B. This would drastically change the 
architecture of the problem we are trying to solve as we would then be launching 8 threads per block instead of 32,
this is inefficient as warp sizes are 32 at the smallest, so we're just wasting the overhead of 24 threads here. 
TLDR: one thread cant compute one index in C. So the ONLY other option is that we need to modify accesses to ensure
that one thread operates on one index of C. Since each kernel launch only considers one row of C, we need to only
consider one kernel (since muktiple kernels won't overlap with eachother as they only write to one row of C).
If this is the case, we need to somehow make acceses such that, 32 threads never access the same index in a row of
8. Hmm.... A thread accesses the index of C based on which column it is on. Pigeonhole principle tells us this is
impossible. 32 threads, 8 columns, impossible. So, as long as threads > columns, you can't eliminate atomic add.
*/
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int bTileSize, int aTileSize) {
  // Each thread handles one row of B
  int row = (blockIdx.y * blockDim.x + blockIdx.x) + (blockIdx.z * aTileSize);
  int b_row = threadIdx.x;
  if (b_row < B_ROWS) {
    // Store all the elements that this thread will process (next 4 elements (inclusive))
    float thread_elements[8] = {
      B[b_row * B_COLS + 0],
      B[b_row * B_COLS + 1],
      B[b_row * B_COLS + 2],
      B[b_row * B_COLS + 3],
      B[b_row * B_COLS + 4],
      B[b_row * B_COLS + 5],
      B[b_row * B_COLS + 6],
      B[b_row * B_COLS + 7]
    };
    float a_element = A[row * A_COLS + b_row];
    atomicAdd(&C[row * C_COLS + 0], a_element * thread_elements[0]);
    atomicAdd(&C[row * C_COLS + 1], a_element * thread_elements[1]);
    atomicAdd(&C[row * C_COLS + 2], a_element * thread_elements[2]);
    atomicAdd(&C[row * C_COLS + 3], a_element * thread_elements[3]);
    atomicAdd(&C[row * C_COLS + 4], a_element * thread_elements[4]);
    atomicAdd(&C[row * C_COLS + 5], a_element * thread_elements[5]);
    atomicAdd(&C[row * C_COLS + 6], a_element * thread_elements[6]);
    atomicAdd(&C[row * C_COLS + 7], a_element * thread_elements[7]);
  }
}

void matrixMultiplyCPU(float* A, float* B, float* C) {
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

// Verify results
bool verifyResults(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-5) {
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
  float *h_A = (float*)malloc(A_ROWS * A_COLS * sizeof(float));
  float *h_B = (float*)malloc(B_ROWS * B_COLS * sizeof(float));
  float *h_C = (float*)malloc(C_ROWS * C_COLS * sizeof(float));
  float *h_C_cpu = (float*)malloc(C_ROWS * C_COLS * sizeof(float));

  randomize_matrix(h_A, A_ROWS, A_COLS);
  randomize_matrix(h_B, B_ROWS, B_COLS);

  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc(&d_A, A_ROWS * A_COLS * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_B, B_ROWS * B_COLS * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_C, C_ROWS * C_COLS * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_A, h_A, A_ROWS * A_COLS * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, B_ROWS * B_COLS * sizeof(float), cudaMemcpyHostToDevice));

  cudaFree(0);
  cudaMemset(d_C, 0, C_ROWS * C_COLS * sizeof(float));

  dim3 gridDim(32,16,2);
  dim3 blockDim(32,1);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C,4,512);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaCheckError(cudaMemcpy(h_C, d_C, C_ROWS * C_COLS * sizeof(float), cudaMemcpyDeviceToHost));

  float time = 0.0f;
  cudaEventElapsedTime(&time, start, stop);

  std::cout << "GPU Timing: " << time << " ms" << std::endl;
  matrixMultiplyCPU(h_A, h_B, h_C_cpu);

  bool correct = verifyResults(h_C, h_C_cpu, C_ROWS * C_COLS);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
