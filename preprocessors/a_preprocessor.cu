#pragma once

/* Preprocessor for A matrix to encode horizontal sparsity */

#include "../utils.cuh"
#include <cuda_runtime.h>

	template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	preprocess_A(int M, int N, int K, float *A, int* A_LIST) {
		const uint cRow = blockIdx.y;

		const uint warpIdx = threadIdx.x / WARPSIZE;
		//
		const uint warpRow = warpIdx / (BN / WN);

		constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
		constexpr uint WSUBM = WM / WMITER;
		constexpr uint WSUBN = WN / WNITER;
		constexpr uint inners = 1024;
		constexpr uint NUM_WARP_ROWS = (BM + WM - 1) / WM;

		// Bitmask encoding: 1 byte per pattern (8 bits for BK=8)
		constexpr uint numMasks = (inners/BK) * NUM_WARP_ROWS * WMITER;
		constexpr uint numInts = (numMasks + 3) / 4;  // Pack 4 masks per int

		const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
		const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

		// Store bitmasks as bytes in shared memory
		__shared__ uint8_t sparsityMasks[numMasks];

		// Initialize shared memory to zero
		for (uint i = threadIdx.x; i < numMasks; i += blockDim.x) {
			sparsityMasks[i] = 0;
		}
		__syncthreads();

		A += cRow * BM * K;

		// Each warp processes its assigned rows
		// Only even-numbered warps write to avoid races (warpIdx & 1 == 0)
		if ((warpIdx & 1) == 0) {
			for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
				for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
					uint8_t mask = 0;

					// Scan all BK columns
					for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
						const uint globalRow = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM;
						const uint globalCol = dotIdx + bkIdx;
						float val = A[globalRow * K + globalCol];
						uint32_t active = __ballot_sync(0xFFFFFFFF, val != 0.0f);

						// If any thread in the warp has non-zero value, set the bit
						if (threadIdxInWarp == 0 && active != 0) {
							mask |= (1 << dotIdx);
						}
					}

					// Write the mask to shared memory
					if (threadIdxInWarp == 0) {
						const uint kBlock = bkIdx / BK;
						const uint maskIdx = kBlock * NUM_WARP_ROWS * WMITER + warpRow * WMITER + wSubRowIdx;
						sparsityMasks[maskIdx] = mask;
					}
				}
			}
		}
		__syncthreads();

		// Pack 4 masks per int and write to global memory
		const uint blockOffset = cRow * numInts;

		for (uint i = threadIdx.x; i < numInts; i += blockDim.x) {
			int packed = 0;
			// Pack 4 consecutive masks into one int
			for (int j = 0; j < 4 && (i * 4 + j) < numMasks; j++) {
				packed |= (sparsityMasks[i * 4 + j] << (j * 8));
			}
			A_LIST[blockOffset + i] = packed;
		}
	}
