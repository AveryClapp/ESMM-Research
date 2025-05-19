#pragma once

/* Kernel #5, 2D Blocktiling (multple * multiple results per thread) */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
	two_blocktiling(int M, int N, int K, const float *A,
						const float *B, float *C) {
	// Determines where the block will start
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	// We calculate BM * BN elements per block, must find how many threads are
	// needed total (including both dimensions)
	const uint totalResultsBlocktile = BM * BN;
	const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

	assert(numThreadsBlocktile == blockDim.x);

	// Blocked groups of cols and sequential rows
	// Assign threadCol and threadRow in row-major order 
	const int threadCol = threadIdx.x % (BN / TN);
	const int threadRow = threadIdx.x / (BN / TN);

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	// Advance matrix pointers to the start of the block
	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	// Get the location of the thread within A block
	const uint innerRowA = threadIdx.x / BK;
	const uint innerColA = threadIdx.x % BK;
	// Define how large our "steps" are in SMEM
	const uint strideA = numThreadsBlocktile / BK;

	const uint innerRowB = threadIdx.x / BN;
	const uint innerColB = threadIdx.x % BN;
	const uint strideB = numThreadsBlocktile / BN;

	float threadResults[TM * TN] = {0.0}; // All thread results

	// Middle loop caching
	float regM[TM] = {0.0};
	float regN[TN] = {0.0}; 

	// Every advance the block through the matrix
	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		
		/* Load elements into SMEM in column order for less bank conflicts */
		for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
			As[(innerRowA + loadOffset) * BK + innerColA] =
				A[(innerRowA + loadOffset) * K + innerColA];
		}
		for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
			Bs[(innerRowB + loadOffset) * BN + innerColB] =
				B[(innerRowB + loadOffset) * N + innerColB];
		}
		__syncthreads();

		// Advance the matrix pointers to the start of the next block
		A += BK;
		B += BK * N;
			
		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			// Store values to be used in inner loop
			for (uint i = 0; i < TM; ++i) {
				regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
			}
			// Store values to be used in inner loop
			for (uint i = 0; i < TN; ++i) {
				regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
			}

			// Calculate TM * TN elements in current block
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[resIdxM * TN + resIdxN] +=
						regM[resIdxM] * regN[resIdxN];
				}
			}
		}
		__syncthreads();
	}
	
	// Accumulate results from thread results registerfile into C
	for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
		for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
			C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
				threadResults[resIdxM * TN + resIdxN];
		}
	}
}
