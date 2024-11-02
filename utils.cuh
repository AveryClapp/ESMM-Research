#ifndef UTILS_CUH
#define UTILS_CUH
#include <vector>

#define CEIL_DIV(M,N) ((M) + (N) - 1) / (N)

void cudaCheck(cudaError_t error, const char *file, int line);

void randomize_matrix_with_pattern(float* mat, int M, int N, std::vector<int> pattern);

void randomize_matrix(float* mat, int M, int N);

void print_matrix(const float* A, int M, int N);

bool verify_matrix(float* mat1, float* mat2, int M, int N);

#endif
