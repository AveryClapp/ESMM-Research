#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "./kernels/1D_Blocktiling.cu"

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

int main() {
	// Setup 
	constexpr int rows = 128;
	constexpr int cols = 128;
	constexpr int inners = 128;

	float *h_A = (float*)malloc(rows * cols * sizeof(float));
	float *h_B = (float*)malloc(rows * cols * sizeof(float));
	float *h_C = (float*)malloc(rows * cols * sizeof(float));
	float *h_C_cpu = (float*)malloc(rows * cols * sizeof(float));

	randomize_matrix(h_A, rows, cols);
	randomize_matrix(h_B, rows, cols);

	float *d_A, *d_B, *d_C;
	cudaCheckError(cudaMalloc(&d_A, rows * cols * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_B, rows * cols * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

	cudaCheckError(cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(d_B, h_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

	cudaMemset(d_C, 0, rows * cols * sizeof(float));

	//Run 1D blocktiling kernel
	constexpr int blockHeight = 32; // Height of tiled block	
	constexpr int blockWidth = 32; // Width of tiled block
	constexpr int blockInner = 8; // width of tiled block
	constexpr int resultsPerThread = 8;
	dim3 gridDim(CEIL_DIV(cols,blockWidth),CEIL_DIV(rows,blockHeight));
	dim3 blockDim((blockHeight * blockWidth)/blockInner);
	matMulBlocktiling<blockHeight, blockWidth, blockInner, resultsPerThread><<<gridDim, blockDim>>>(d_A, d_B, d_C, cols, inners);
	cudaCheckError(cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
	bool correct = verifyResults(h_C, h_C_cpu, rows * cols);
	printf("Matrix multiplication %s\n", correct ? "PASSED" : "FAILED");

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_cpu);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}


