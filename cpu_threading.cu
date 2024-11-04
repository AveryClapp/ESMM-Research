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
void processColumn(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int colA) {
  for (int row = 0; row < 1024; row++) {
    const float aVal = A[row * 32 + colA];
    for (int j = 0; j < 8; j++) {
      C[row * 8 + j] += aVal * B[colA * 8 + j];
    }
  }
}

void threadedMatrixMultiplication(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
  std::memset(C, 0, 1024 * 8 * sizeof(float));
  std::vector<std::thread> threads(32);

  for (int t = 0; t < 32; t++) {
    threads[t] = std::thread(processColumn, A, B, C, t);
  }

  for (auto& thread : threads) {
    thread.join();
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
