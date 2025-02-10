#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "./kernels/1D_Blocktiling.cu"
#include "./kernels/prod.cu"
#include "./kernels/prod2.cu"
#include <chrono>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

#define START auto  start = std::chrono::high_resolution_clock::now();
#define END(kernel) \
	auto end = std::chrono::high_resolution_clock::now(); \
	auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
	std::cout << #kernel << " took " << time.count() / 1000.0 << " milliseconds" << std::endl;

int main() {
	// Setup 
	constexpr int rows = 8;
	constexpr int cols = 8;
	constexpr int inners = 8;

	// Allocate host matrices
	float *h_A = (float*)malloc(rows * cols * sizeof(float));
	float *h_B = (float*)malloc(rows * cols * sizeof(float));
	float *h_C = (float*)malloc(rows * cols * sizeof(float));
	float *h_C_cpu = (float*)malloc(rows * cols * sizeof(float));

	// Generate random data
	randomize_matrix(h_A, rows, cols);
	randomize_matrix(h_B, rows, cols);

	// Allocate device matrices
	float *d_A, *d_B, *d_C;
	cudaCheckError(cudaMalloc(&d_A, rows * cols * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_B, rows * cols * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

	// Copy random data to device matrices
	cudaCheckError(cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(d_B, h_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

	// Reset output matrix to 0
	cudaMemset(d_C, 0, rows * cols * sizeof(float));
	
	// divide matrices into blocks and set threads per block

	//Run 1D blocktiling kernel
	constexpr int blockHeight = 64;
	constexpr int blockWidth = 64;
	constexpr int blockInner = 8;
	constexpr int resultsPerThread = 8;
	dim3 gridDim(CEIL_DIV(cols,blockWidth), CEIL_DIV(rows,blockHeight));
	dim3 blockDim((blockWidth * blockHeight) / resultsPerThread);
	START;		
	//sgemm1DBlocktiling<blockHeight,blockWidth,blockInner,resultsPerThread><<<gridDim, blockDim>>>(rows,cols,inners,d_A,d_B,d_C);
	esmm_shmem_multi2<<<dim3(1,1), dim3(8), 8*8*2>>>(rows, cols, inners, 8, d_A, d_B, d_C);
	END("PROD")
	cudaCheckError(cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
	// Reset C matrix
	cudaMemset(d_C, 0, rows * cols * sizeof(float));
	matrixMultiplyCPU(h_A, h_B, h_C_cpu, rows, cols);
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


