#pragma once

/* Preprocessor for A matrix to encode horizontal sparsity */

#include "../utils.cuh"
#include <cuda_runtime.h>

	template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	preprocess_A(int M, int N, int K, float *A, int* A_LIST) {
		const uint cRow = blockIdx.y;
		const uint cCol = blockIdx.x;

		const uint warpIdx = threadIdx.x / WARPSIZE;
		//
		const uint warpRow = warpIdx / (BN / WN);

		constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
		constexpr uint WSUBM = WM / WMITER;
		constexpr uint WSUBN = WN / WNITER;
		constexpr uint inners = 1024;
		constexpr uint NUM_WARP_ROWS = (BM + WM - 1) / WM;

		// Target 50% or lower sparsity
		constexpr uint MAX_SPARSE_OFFSETS = BK / 2;
		constexpr uint ELEMENTS_PER_PATTERN = 5;
		constexpr uint denseListSize = (inners/BK) * NUM_WARP_ROWS * WMITER * ELEMENTS_PER_PATTERN;

		const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
		const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); 

		// Enough space to encode BK/2 + 1 elements for each 32x8 block
		__shared__ int denseList[denseListSize];

		// Initialize shared memory to zero
		for (uint i = threadIdx.x; i < denseListSize; i += blockDim.x) {
			denseList[i] = 0;
		}
		__syncthreads();

		A += cRow * BM * K;

		for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
			for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
				for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
					const uint globalRow = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM;
					const uint globalCol = dotIdx + bkIdx;
					float val = A[globalRow * K + globalCol];
					uint32_t active = __ballot_sync(0xFFFFFFFF, val != 0.0f);
					// Only let the first warp in each warpRow pair write to avoid races
					bool shouldWrite = (threadIdxInWarp == 0) && (active != 0) && ((warpIdx & 1) == 0);
					if (shouldWrite) {
						// Start of the segment in the denseList for this K Block
						const uint kBlockBase = (bkIdx / BK) * (NUM_WARP_ROWS * WMITER * ELEMENTS_PER_PATTERN);
						// Move to the current
						const uint countIdx = kBlockBase + (warpRow * WMITER + wSubRowIdx) * ELEMENTS_PER_PATTERN;
						int currentCount = atomicAdd(&denseList[countIdx], 1);
						if (currentCount < MAX_SPARSE_OFFSETS) {
							denseList[countIdx + currentCount + 1] = dotIdx;
						} else {
							denseList[countIdx] = -1;
						}
					}
				}
				__syncthreads();
			}
			__syncthreads();
		}

		// CeilDiv size of dense list (should already be divisible by 4)
		constexpr uint denseListSizeInt4 = (denseListSize + 3) / 4;

		const uint blockOffset = (cRow * gridDim.x + cCol) * denseListSizeInt4;

		int4* denseListVec = reinterpret_cast<int4*>(denseList);
		int4* A_LIST_Vec = reinterpret_cast<int4*>(A_LIST);

		// Each thread writes 4 and increment by 256 (Each warp writes 2560 elements to this runs 10 times)
		for (uint i = threadIdx.x; i < denseListSizeInt4; i += blockDim.x) {
			A_LIST_Vec[blockOffset + i] = denseListVec[i];
		}
	}
