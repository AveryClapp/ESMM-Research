#pragma once

#ifndef UTILS_CUH
#define UTILS_CUH
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <string>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define PATTERN_LENGTH 8

using std::cout;
using std::endl;

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


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  return;
}

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [kernel_choice] [runs] [options]" << endl;
    cout << "  kernel_choice: " << endl;
    cout << "    Single kernel: 1-12 (run specific kernel)" << endl;
    cout << "    Multiple kernels: \"1,3,5\" (comma-separated, no spaces)" << endl;
    cout << "    Range: \"1-5\" (run kernels 1 through 5)" << endl;
    cout << "    All: \"all\" (run all kernels 1-12)" << endl;
    cout << "  runs: number of runs per kernel (default: 1)" << endl;
    cout << "  options:" << endl;
    cout << "    --verbose or -v: print detailed results" << endl;
    cout << "    --help or -h: show this help" << endl;
    cout << endl;
    cout << "Available kernels:" << endl;
    cout << "  1:  Naive Implementation" << endl;
    cout << "  2:  Global Memory Coalescing" << endl;
    cout << "  3:  Shared Memory Blocks" << endl;
    cout << "  4:  One Dimensional Blocktiling" << endl;
    cout << "  5:  Two Dimensional Blocktiling" << endl;
    cout << "  6:  Vectorized Memory Accessing" << endl;
    cout << "  7:  1D Vectorized Approach" << endl;
    cout << "  8:  Basic Warptiling" << endl;
    cout << "  9:  1D Warptiling" << endl;
    cout << "  10: Emergent Sparsity Matrix Multiplication (ESMM)" << endl;
    cout << "  11: ESMM Warpskipping" << endl;
    cout << "  12: cuBLAS" << endl;
}

void randomize_matrix_with_pattern(float *mat, int M, int N,
                  std::vector<int> pattern) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int pattern_idx = (i * N + j) % PATTERN_LENGTH;
      if (pattern[pattern_idx] == 1) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i * N + j] = tmp;
      } else {
        mat[i * N + j] = 0.0f;
      }
    }
  }
}

void randomize_matrix(float *mat, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
      tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
      mat[i * N + j] = tmp;
    }
  }
}

std::vector<int> stringToVector(const std::string &str) {
  std::vector<int> vec;
  vec.reserve(str.length());
  for (char c : str) {
    vec.push_back(c - '0');
  }
  return vec;
}

// CPU implementation of matrix multiplication for verification
void matrixMultiplyCPU(const float *A, const float *B, float *C, int M, int N,
                       int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Function to verify GPU results against CPU results
bool verifyResults(const float *gpuResults, const float *cpuResults, int size,
                   float tolerance = 1e-3) {
  int mismatchCount = 0;
  float maxDiff = 0.0f;
  int firstMismatchIdx = -1;

  for (int i = 0; i < size; i++) {
    float diff = std::abs(gpuResults[i] - cpuResults[i]);
    if (diff > tolerance) {
      mismatchCount++;
      if (firstMismatchIdx == -1)
        firstMismatchIdx = i;
      maxDiff = std::max(maxDiff, diff);
    }
  }

  if (mismatchCount > 0) {
    std::cout << "Verification failed: " << mismatchCount
              << " mismatches out of " << size << " elements." << std::endl;
    std::cout << "Max difference: " << maxDiff << std::endl;
    if (firstMismatchIdx >= 0) {
      int row = firstMismatchIdx / (size / sqrt(size));
      int col = firstMismatchIdx % (int)(size / sqrt(size));
      std::cout << "First mismatch at index " << firstMismatchIdx
                << " (row=" << row << ", col=" << col << "): "
                << "GPU=" << gpuResults[firstMismatchIdx]
                << ", CPU=" << cpuResults[firstMismatchIdx] << std::endl;
    }
    return false;
  }

  return true;
}

__device__ inline void multiply_dense(int wSubRowIdx, int wSubColIdx,
								int WNITER, float regM_val, float* regN,
										float* threadResults) {
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__device__ inline void multiply_half(int wSubRowIdx, int wSubColIdx,
								int WNITER, float regM_val, float* regN,
										float* threadResults) {
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}


__device__ inline void multiply_quarter(int wSubRowIdx, int wSubColIdx,
								int WNITER, float regM_val, float* regN,
										float* threadResults) {
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

__device__ inline void multiply_eighth(int wSubRowIdx, int wSubColIdx,
								int WNITER, float regM_val, float* regN,
										float* threadResults) {
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
}

#endif
