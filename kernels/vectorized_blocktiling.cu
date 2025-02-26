#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
	vectorized_blocktiling(int M, int N, int K, const float *A,
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

	const uint innerRowB = threadIdx.x / BN;
	const uint innerColB = threadIdx.x % BN;

	float threadResults[TM * TN] = {0.0}; // All thread results

	// Middle loop caching
	float regM[TM] = {0.0};
	float regN[TN] = {0.0}; 

	// Every advance the block through the matrix
	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
		
		// Load elements from row major order in A to column major in As
		As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
   		As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
	    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
		As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

		// Load the float4 value from global memory
		 tmp = reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];

		// Store the float4 value to shared memory
		*reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4]) = tmp;
		//reinterpret_cast<const float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast< const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
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
			float4 tmp;
			tmp.x = threadResults[resIdxM * TN + resIdxN];
			tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
			tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
			tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
			reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
		}
	}
}
