#pragma once

/**
* Basic GPU implementation of matrix multiplication
*/

#include <cuda_runtime.h>


__global__ void basic(int rows, int cols, int inners, const float* A, const float* B, float* C) {
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < rows && y < cols) {
		float tmp = 0.0;
		for (int i = 0; i < inners; ++i) {
			tmp += A[x * inners + i] * B[i * cols + y];
		}
		C[x * cols + y] = tmp;
	}
}
