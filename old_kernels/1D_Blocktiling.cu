#pragma once

/* Kernel #4, One Dimensional Blockitling (multiple results per thread) */

#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include "../include/utils.cuh"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void one_blocktiling(int M, int N, int K, const float *A, const float *B, float *C) {
	// Main column and row of iterative tiling
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const int threadCol = threadIdx.x % BN;
	const int threadRow = threadIdx.x / BN;

	
	__shared__ float As[BM * BK];
	__shared__ float Bs[BN * BK];
	

	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	//Block-level indexing for SMEM 
	assert(BM * BK == blockDim.x);
	assert(BN * BK == blockDim.x);
	const uint innerColA = threadIdx.x % BK;
	const uint innerRowA = threadIdx.x / BK;
	const uint innerColB = threadIdx.x % BN; 
	const uint innerRowB = threadIdx.x / BN;

	float threadResults[TM] = {0.0};

	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
		Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
		__syncthreads();

		A += BK;
		B += BK * N;

		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			float tmpB = Bs[dotIdx * BN + threadCol];
			for (uint resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TM; ++resIdx) {
		C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
}
