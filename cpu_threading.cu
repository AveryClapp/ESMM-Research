// Similar implementation to cpu_basic, except now we thread on the middle loop so we have 32 threads that do 8 operations (one row of B).
#include "utils.cuh"
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8

// For every row in A (also C), create 32 threads so that each thread goes through 8 (B_COLS). This will allow for one whole row of 
// C to be computed per thread spawn. This will eventually be lifted/modified to a CUDA implementation where you launch one warp
// to compute a whole row of C.
void computeRowContribution(float* A, float* B, float* partialC, int rowB) {
  for(int i = 0; i < A_ROWS; ++i) {
    for(int j = 0; j < B_COLS; ++j) {
      partialC[i * C_COLS + j] += A[i * A_COLS + rowB] * B[rowB * B_COLS + j];
    }
  }
}

void threadedMatrixMultiplication(float* A, float* B, float* C) {
  std::vector<std::thread> allThreads;
  allThreads.reserve(B_ROWS);
  std::vector<float*> partialCs(B_ROWS, nullptr);
  for(int k = 0; k < B_ROWS; ++k) {
    partialCs[k] = new float[C_ROWS * C_COLS];
    std::memset(partialCs[k], 0, sizeof(float) * C_ROWS * C_COLS);
  }
  for(int k = 0; k < B_ROWS; ++k) {
    allThreads.emplace_back(computeRowContribution, A, B, partialCs[k], k);
  }
  for(auto& thread : allThreads) {
    thread.join();
  }
  for(int k = 0; k < B_ROWS; ++k) {
    for(int i = 0; i < C_ROWS; ++i) {
      for(int j = 0; j < C_COLS; ++j) {
        C[i * C_COLS + j] += partialCs[k][i * C_COLS + j];
      }
    }
    delete[] partialCs[k];
  }
}
int main() {
  float *A = new float[A_ROWS * A_COLS];
  float *B = new float[B_ROWS * B_COLS];
  float *C = new float[A_ROWS * B_COLS];

  std::memset(C, 0, sizeof(float) * C_ROWS * C_COLS); 
  randomize_matrix(A, A_ROWS, A_COLS);
  randomize_matrix(B, B_ROWS, B_COLS);

  auto start = std::chrono::high_resolution_clock::now();
  threadedMatrixMultiplication(A,B,C);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  printf("CPU (w/ threading) Timing: %f ms\n", duration.count() / 1000.0f);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
