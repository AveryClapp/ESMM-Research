// Basic CPU implementation of matrix multiplication,
// for 0 : col 
//  for mid 0 : 32 
//    for inner 0 : 8
// Calculate partial sums for everything, need to unroll the loop.
#include "utils.cuh"
#include <chrono>
#include <stdio.h>

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 8
#define C_ROWS 1024
#define C_COLS 8

void basicMatrixMultiplication(float* A, float* B, float* C) {
  for (int i = 0; i < A_ROWS; ++i) {
    for (int j = 0; j < B_COLS; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < A_COLS; ++k) {
        sum += A[i * A_COLS + k] * B[k * B_COLS + j];
      }
      C[i * B_COLS + j] = sum;
    }
  }
}


int main() {
  float *A = new float[A_ROWS * A_COLS];
  float *B = new float[B_ROWS * B_COLS];
  float *C = new float[A_ROWS * B_COLS];

  randomize_matrix(A, A_ROWS, A_COLS);
  randomize_matrix(B, B_ROWS, B_COLS);

  auto start = std::chrono::high_resolution_clock::now();
  basicMatrixMultiplication(A,B,C);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("CPU Timing: %f ms\n", duration.count() / 1000.0f);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
