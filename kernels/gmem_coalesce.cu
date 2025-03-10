#pragma once

#include <cuda_runtime.h>


template <int BLOCKSIZE>
__global__ void gmem_coalesce(int rows, int cols, int inners, const float* A, const float* B, float* C) {
	const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
	const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

	if (x < rows && y < cols) {
		float tmp = 0.0;
		for (int i = 0; i < inners; ++i) {
			tmp += A[x * inners + i] * B[i * cols + y];
		}
		C[x * cols + y] = tmp * C[x * cols + y];
	}
}
