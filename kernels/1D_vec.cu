#pragma once

/* Kernel #6, Transposing A matrix into SMEM for better memory access */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
	one_d_vec(int M, int N, int K, float *A,
					 float *B, float *C) {
	// Determines where the block will start
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	// We calculate BM * BN elements per block, must find how many threads are
	// needed total (including both dimensions)
	const uint totalResultsBlocktile = BM * BN;
	const uint numThreadsBlocktile = totalResultsBlocktile / (TM);

	assert(numThreadsBlocktile == blockDim.x);

	// Blocked groups of cols and sequential rows
	// Assign threadCol and threadRow in row-major order 
	const int threadCol = threadIdx.x % (BN);
	const int threadRow = threadIdx.x / (BN);

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	// Advance matrix pointers to the start of the block
	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	const uint innerRowA = threadIdx.x / (BK / 4);
	const uint innerColA = threadIdx.x % (BK / 4);
  	const uint innerRowB = threadIdx.x / (BN / 4);
	const uint innerColB = threadIdx.x % (BN / 4);

	float threadResults[TM] = {0.0}; // All thread results

	// Middle loop caching
	float regM[TM] = {0.0};
	float regN[TN] = {0.0}; 

	// Every advance the block through the matrix
	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		float4 tmp = 
		  reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
		
		// Load elements from row major order in A to column major in As
		As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
   		As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
	    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
		As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

		// Load the float4 value from global memory
		reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
		  reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
		__syncthreads();

		// Advance the matrix pointers to the start of the next block
		A += BK;
		B += BK * N;
			
		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			// Store values to be used in inner loop
			for (uint i = 0; i < TM; ++i) {
				regM[i] = As[dotIdx * BM + threadRow * TM + i];
			}

			// Just one element, can simplify this when it starts to work
			for (uint i = 0; i < TN; ++i) {
				regN[i] = Bs[dotIdx * BN + threadCol + i];
			}

			// Calculate TM * TN elements in current block
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[resIdxM + resIdxN] +=
						regM[resIdxM] * regN[resIdxN];
				}
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TM; ++resIdx) {
			    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
	// Accumulate results from thread results registerfile into C
//	for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
	//	for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
	//		float4 tmp;
	//		tmp.x = threadResults[resIdxM + resIdxN];
	//		tmp.y = threadResults[resIdxM + resIdxN + 1];
	//		tmp.z = threadResults[resIdxM + resIdxN + 2];
	//		tmp.w = threadResults[resIdxM + resIdxN + 3];
	//		reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol + resIdxN])[0] = tmp;
	//	}
	//}
}
