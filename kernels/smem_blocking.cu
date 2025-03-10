#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BLOCKSIZE>
__global__ void smem_blocking(int rows, int cols, int inners, const float *A, const float *B, float *C) {

	const uint cRow = blockIdx.x;
	const uint cCol = blockIdx.y;

	__shared__ float As[BLOCKSIZE * BLOCKSIZE];
	__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

	const uint threadCol = threadIdx.x % BLOCKSIZE;
	const uint threadRow = threadIdx.x / BLOCKSIZE;

	A += cRow * BLOCKSIZE * inners;
	B += cCol * BLOCKSIZE;
	C += cRow * BLOCKSIZE * cols + cCol * BLOCKSIZE;

	float tmp = 0.0;
	for (int bkIdx = 0; bkIdx < inners; bkIdx += BLOCKSIZE) {
		As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * inners + threadCol];
		Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * cols + threadCol];
		__syncthreads();
	
		A += BLOCKSIZE;
		B += BLOCKSIZE * cols;

		for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
			tmp += As[threadRow * BLOCKSIZE + dotIdx] *
			Bs[dotIdx * BLOCKSIZE + threadCol];
		}
		__syncthreads();
	}
	C[threadRow * cols + threadCol] = tmp + C[threadRow * cols + threadCol];
}
