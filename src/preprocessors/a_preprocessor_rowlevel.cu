#pragma once

/* Row-Level Preprocessor for A matrix - Config Agnostic */

#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * @param BK The K-block size (8 or 16)
 * @param NUM_THREADS Number of threads launched with the kernel
 */
template <const int BK, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	preprocess_A_rowlevel(int M, int N, int K, float *A, uint8_t* A_LIST) {

	constexpr int WARP_SIZE = 32;
	const uint numKBlocks = K / BK;

	const uint warpId = threadIdx.x / WARP_SIZE;
	const uint laneId = threadIdx.x % WARP_SIZE;

	constexpr int ROWS_PER_WARP = WARP_SIZE / BK;
	constexpr int THREADS_PER_ROW = BK;

	constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * (NUM_THREADS / WARP_SIZE);

	for (uint rowBlockBase = blockIdx.x * ROWS_PER_BLOCK;
	     rowBlockBase < M;
	     rowBlockBase += gridDim.x * ROWS_PER_BLOCK) {

		const uint localRowInWarp = laneId / THREADS_PER_ROW;
		const uint threadPosInRow = laneId % THREADS_PER_ROW;

		const uint row = rowBlockBase + warpId * ROWS_PER_WARP + localRowInWarp;

		if (row >= M) return;

		#pragma unroll 16
		for (uint kBlock = 0; kBlock < numKBlocks; kBlock++) {

			// Use __ldg for read-only cache optimization
			const uint kOffset = kBlock * BK + threadPosInRow;
			const float val = __ldg(&A[row * K + kOffset]);

			const uint32_t ballot = __ballot_sync(0xffffffff, val != 0.0f);

			uint8_t mask;
			if constexpr (BK == 8) {
				const uint shift = localRowInWarp * 8;
				mask = (ballot >> shift) & 0xFF;
			} else if constexpr (BK == 16) {
				const uint shift = localRowInWarp * 16;
				mask = (ballot >> shift) & 0xFFFF;
			}

			if (threadPosInRow == 0) {
				// Store as uint16_t to handle both BK=8 and BK=16
				A_LIST[row * numKBlocks + kBlock] = mask;
			}
		}
	}
}
