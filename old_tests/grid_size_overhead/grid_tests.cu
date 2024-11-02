#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>


// Used for calculating startup costs
__global__ void naiveKernel() {}

__global__ void simpleMatMul(const float* A, const float* B, float* C, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] + B[k * N + col]; 
    }
    C[row * N + col] = sum;
  }
}

int main() {
  int N = 1024;

  size_t matrixBytes = N * N * sizeof(float);
  float *d_A, *d_B, *d_C;
  // Allocate and initialize device memory
  cudaMalloc(&d_A, matrixBytes);
  cudaMalloc(&d_B, matrixBytes);
  cudaMalloc(&d_C, matrixBytes);

  // Initialize with random data
  std::vector<float> h_A(N * N), h_B(N * N);
  for (size_t i = 0; i < N * N; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  cudaMemcpy(d_A, h_A.data(), matrixBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), matrixBytes, cudaMemcpyHostToDevice);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 blockDim(16,16);
  dim3 gridConfigs[] = {
    dim3(1, 1),    // 32x32 total
    dim3(2, 2),    // 64x64 total
    dim3(4, 4),    // 128x128 total
    dim3(8, 8),    // 256x256 total
    dim3(16, 16),  // 512x512 total
    dim3(32, 32),   // 1024x1024 total
    dim3(64,64),
    dim3(128,128)
  };

  // Force loading of cuda runtime
  cudaFree(0);

  for (const auto &dim : gridConfigs) {
    cudaEventRecord(start);
    naiveKernel<<<dim,blockDim>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_milliseconds = 0;
    cudaEventElapsedTime(&naive_milliseconds, start, stop);
    std::cout << "Grid size: (" << dim.x << "," << dim.y << ")"
      << " Launch Time: " << naive_milliseconds << " ms" << std::endl;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}



