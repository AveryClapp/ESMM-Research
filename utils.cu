#include <stdio.h>
#include "utils.cuh"
/*
 * Helper file for common functions (creating matrices, CEIL_DIV)
 */

// CEIL_DIV Operation
#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  return;
};

// Randomly generate a matrix based on non-zero pattern specified 
void randomize_matrix_with_pattern(float *mat, int M, int N, std::vector<int> pattern) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int pattern_idx = (i * N + j) % pattern_length;
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

// Randomly generate matrix, don't worry about bit pattern
void randomize_matrix(float *mat, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
      tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
      mat[i * N + j] = tmp;  
    }
  }
}


void print_matrix(const float *A, int M, int N) {
  int i;
  printf("[");
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      printf("%5.2f ", A[i]);
    else
      printf("%5.2f, ", A[i]);
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        printf(";\n");
    }
  }
  printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, int M, int N) {
  double diff = 0.0;
  int total_size = M * N;

  for (int i = 0; i < total_size; i++) {
    diff = fabs((double)mat1[i] - (double)mat2[i]);
    if (diff > 1e-2) {
      printf("error at position (%d,%d): %5.2f,%5.2f\n", 
          i/N, i%N, mat1[i], mat2[i]);
      return false;
    }
  }
  return true;
}

