#ifndef UTILS_CUH
#define UTILS_CUH
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#define CEIL_DIV(M,N) (((M) + (N)-1) / (N))
#define PATTERN_LENGTH 8

void cudaCheck(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
		printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
 	return;
}

void randomize_matrix_with_pattern(float* mat, int M, int N, std::vector<int> pattern) {
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

void randomize_matrix(float* mat, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
			tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
			mat[i * N + j] = tmp;
		}
	}
}

std::vector<int> stringToVector(const std::string& str) {
	std::vector<int> vec;
	vec.reserve(str.length());
	for (char c : str) {
		vec.push_back(c-'0');
	}
	return vec;
}

bool verify_matrix(float* mat1, float* mat2, int M, int N) {
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

void matrixMultiplyCPU(float* A, float* B, float* C, int rows, int cols) {
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			float sum = 0.0f;
			for (int i = 0; i < rows; i++) {
				sum += A[row * cols + i] * B[i * cols + col];
			}
			C[row * cols + col] = sum;
		}
	}
}

bool verifyResults(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-3) {
	for (int i = 0; i < size; i++) {
		if (fabs(gpuResult[i] - cpuResult[i]) > tolerance) {
			printf("Mismatch at position %d: GPU = %f, CPU = %f\n",
				i, gpuResult[i], cpuResult[i]);
			return false;
		}
	}
	return true;
}
#endif
