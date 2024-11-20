#include <iostream>
#include "utils.cuh"
#include <map>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


// Error checking macro
#define CUDA_CHECK(call)                                                        \
  do {                                                                        \
    cudaError_t error = call;                                               \
    if (error != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n",      \
          cudaGetErrorString(error), error, __FILE__, __LINE__);      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                       \
  } while (0)

// Matrix dimensions
#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8

__global__ void kernel11111111(float* A, float* B, float* C) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  if(col < C_COLS) {
    float sum = 0.0f;
    for(int k = 0; k < A_COLS; ++k) {
      sum += A[row * A_COLS + k] * B[k * B_COLS + col];
    }
    C[row * C_COLS + col] = sum;
  }
}

// CUDA Kernel for pattern "10101010" (even columns active)
__global__ void kernel10101010(float* A, float* B, float* C) {
  int row = blockIdx.x;
  int threadIdxOffset = threadIdx.x;
  // Active columns are 0,2,4,6
  int colIndices[4] = {0, 2, 4, 6};
  int activeCols = 4;
  if(threadIdxOffset < activeCols) {
    int col = colIndices[threadIdxOffset];
    float sum = 0.0f;
    for(int k = 0; k < A_COLS; ++k) {
      sum += A[row * A_COLS + k] * B[k * B_COLS + col];
    }
    C[row * C_COLS + col] = sum;
  }
}

// CUDA Kernel for pattern "00000010" (only column 6 active)
__global__ void kernel00000010(float* A, float* B, float* C) {
  int row = blockIdx.x;
  int col = 6; // Only column 6 is active
  if(threadIdx.x == 0) { // Only one thread needed
    float sum = 0.0f;
    for(int k = 0; k < A_COLS; ++k) {
      sum += A[row * A_COLS + k] * B[k * B_COLS + col];
    }
    C[row * C_COLS + col] = sum;
  }
}

// CPU function for matrix multiplication
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

// Function to verify GPU results against CPU results
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
  srand(time(NULL)); // Seed for random number generation

  // Define sparsity patterns
  std::vector<std::string> patterns = {"11111111", "10101010", "00000010"};

  // Map patterns to corresponding kernels
  typedef void (*Kernel)(float*, float*, float*);
  std::map<std::string, Kernel> kernelMap = {
    {"11111111", kernel11111111},
    {"10101010", kernel10101010},
    {"00000010", kernel00000010} 
  };

  // Allocate host memory
  float *h_A = new float[A_ROWS * A_COLS];
  float *h_B = new float[B_ROWS * B_COLS];
  float *h_C = new float[C_ROWS * C_COLS];
  float *h_C_verify = new float[C_ROWS * C_COLS];

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, A_ROWS * A_COLS * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_B, B_ROWS * B_COLS * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_C, C_ROWS * C_COLS * sizeof(float)));

  // Iterate over each sparsity pattern
  for (size_t p = 0; p < patterns.size(); ++p) {
    std::string pattern = patterns[p];
    std::vector<int> activeCols = stringToVector(pattern);

    // Initialize matrices A and B
    randomize_matrix(h_A, A_ROWS, A_COLS);
    randomize_matrix_with_pattern(h_B, B_ROWS, B_COLS, activeCols);

    // Initialize matrix C to zero
    memset(h_C, 0, C_ROWS * C_COLS * sizeof(float));

    // Copy A and B to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_ROWS * A_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, B_ROWS * B_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, C_ROWS * C_COLS * sizeof(float)));

    // Select the appropriate kernel
    Kernel kernel = kernelMap[pattern];

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Define grid and block dimensions based on pattern
    dim3 gridDim(C_ROWS, 1, 1); // 1024 blocks, one per row of A/C
    dim3 blockDim(32, 1, 1);    // 32 threads per block (max)

    // Adjust blockDim.x based on the number of active columns
    int activeThreads = activeCols.size();
    if(pattern == "11111111") {
      blockDim.x = 8; // 8 active columns
    }
    else if(pattern == "10101010") {
      blockDim.x = 4; // 4 active columns
    }
    else if(pattern == "00000010") {
      blockDim.x = 1; // 1 active column
    }

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch the kernel
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate the elapsed time between the two events
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy the result matrix C back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, C_ROWS * C_COLS * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute the reference result on CPU
    cpuMatrixMultiply(h_A, h_B, h_C_verify);

    // Verify the results
    bool passed = verify_results(h_C, h_C_verify, C_ROWS * C_COLS);

    // Print the results
    printf("Pattern: %s\n", pattern.c_str());
    printf("Active Columns: %lu\n", activeCols.size());
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("Verification: %s\n\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_verify;

  // Reset the device and exit
  CUDA_CHECK(cudaDeviceReset());

  return 0;
}
