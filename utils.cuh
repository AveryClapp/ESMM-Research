#ifndef UTILS_CUH
#define UTILS_CUH
#include <vector>
#include <cuda_runtime.h>
#include <string>
#define CEIL_DIV(M,N) ((M) + (N) - 1) / (N)
#define PATTERN_LENGTH 8

void cudaCheck(cudaError_t error, const char *file, int line);

void randomize_matrix_with_pattern(float* mat, int M, int N, std::vector<int> pattern);

void randomize_matrix(float* mat, int M, int N);

std::vector<int> stringToVector(const std::string& str);

void print_matrix(const float* A, int M, int N);

bool verify_matrix(float* mat1, float* mat2, int M, int N);

#endif
