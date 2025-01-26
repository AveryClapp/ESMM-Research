#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
#include "kernels.cu"

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
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

// Verify results
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

int main() {
	constexpr int rows = 32;
	constexpr int cols = 32;
	constexpr int inners = 32;

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

	cudaFree(0);
	cudaMemset(d_C, 0, rows * cols * sizeof(float));
	//Tiling of a (32x32) * (32x32) matrix multiplication
	dim3 gridDim(4,2);
	dim3 blockDim(8,16);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	sequential_rectangles<<<gridDim, blockDim.x * blockDim.y>>>(d_A, d_B, d_C, blockDim.x, blockDim.y, inners);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaCheckError(cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

	float time = 0.0f;
	cudaEventElapsedTime(&time, start, stop);

	std::cout << "GPU Timing: " << time << " ms" << std::endl;
	matrixMultiplyCPU(h_A, h_B, h_C_cpu, rows, cols);

	bool correct = verifyResults(h_C, h_C_cpu, rows * cols);
		//printf("Matrix
		//multiplication
		//%s\n",
		//correct ?
		//"PASSED" :
		//"FAILED");

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_cpu);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}


