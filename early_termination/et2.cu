#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

__global__ void matMul(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matMulEarly(float* C, float* A, float* B, int N, int* colLengths) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    int length = colLengths[col];

    for (int k = 0; k < length; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Generate matrices with varying column lengths
void generateSparseMatrices(std::vector<float>& h_A, std::vector<float>& h_B, 
  std::vector<int>& colLengths, int N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> val_dis(-1.0, 1.0);
  std::uniform_real_distribution<> len_dis(0.1, 1.0);

  colLengths.resize(N);
  for (int i = 0; i < N; i++) {
    colLengths[i] = static_cast<int>(N * len_dis(gen));
  }

  h_A.assign(N * N, 0.0f);
  h_B.assign(N * N, 0.0f);

  for (int col = 0; col < N; col++) {
    //Fill in nonzero values for COL up to its random URV length
    int len = colLengths[col];
    for (int row = 0; row < len; row++) {
      h_B[row * N + col] = val_dis(gen);
    }
    for (int row = 0; row < N; row++) {
      h_A[row * N + col] = val_dis(gen);
    }
  }
}

int main() {
  const int N = 1024;
  size_t bytes = N * N * sizeof(float);

  std::vector<float> h_A, h_B;
  std::vector<int> colLengths;
  generateSparseMatrices(h_A, h_B, colLengths, N);

  std::vector<int> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  // Sort colLengths by # of nonzero values
  std::sort(indices.begin(), indices.end(),
      [&colLengths](int i1, int i2) { 
      return colLengths[i1] < colLengths[i2]; 
      });

  std::vector<float> h_A_sorted(N * N);
  std::vector<float> h_B_sorted(N * N);
  std::vector<int> sortedLengths(N);

  // Sort the array by number of nonzero elements in each column
  for(int i = 0; i < N; i++) {
    sortedLengths[i] = colLengths[indices[i]];
    for(int j = 0; j < N; j++) {
      h_B_sorted[j * N + i] = h_B[j * N + indices[i]];
      h_A_sorted[j * N + i] = h_A[j * N + indices[i]];
    }
  }

  float *d_A, *d_B, *d_C;
  int *d_lengths;
  cudaFree(0); // Initialize context
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  cudaMalloc(&d_lengths, N * sizeof(int));
  
  cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
      (N + blockDim.y - 1) / blockDim.y);

  // Benchmark regular matMul
  cudaEventRecord(start);
  matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float fullTime;
  cudaEventElapsedTime(&fullTime, start, stop);

  cudaMemcpy(d_A, h_A_sorted.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B_sorted.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lengths, sortedLengths.data(), N * sizeof(int), cudaMemcpyHostToDevice);

  // Benchmark matMulEarly
  cudaEventRecord(start);
  matMulEarly<<<gridDim, blockDim>>>(d_C, d_A, d_B, N, d_lengths);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float earlyTime;
  cudaEventElapsedTime(&earlyTime, start, stop);

  double avgLength = std::accumulate(colLengths.begin(), colLengths.end(), 0.0) / N;
  double maxLength = *std::max_element(colLengths.begin(), colLengths.end());

  std::cout << "Results for N = " << N << ":\n";
  std::cout << "Average column length: " << avgLength << " ("
    << (avgLength/N)*100 << "%)\n";
  std::cout << "Max column length: " << maxLength << " ("
    << (maxLength/N)*100 << "%)\n";
  std::cout << "Full computation time: " << fullTime << " ms\n";
  std::cout << "Early termination time: " << earlyTime << " ms\n";
  std::cout << "Speedup: " << fullTime/earlyTime << "x\n";

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_lengths);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
