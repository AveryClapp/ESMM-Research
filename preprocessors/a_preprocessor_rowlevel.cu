#pragma once

/* Row-Level Preprocessor for A matrix - Config Agnostic */

#include "../utils.cuh"
#include <cuda_runtime.h>

/*
 * Stores one bitmask per row per K-block
 * Layout: row * numKBlocks + kBlock
 * Any kernel config can read this!
 *
 * @param BK The K-block size (typically 8 or 16)
 */
template <const int BK, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	preprocess_A_rowlevel(int M, int N, int K, float *A, uint8_t* A_LIST) {

	const uint row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;

	const uint numKBlocks = K / BK;

	// Each thread processes one full row
	for (uint kBlock = 0; kBlock < numKBlocks; kBlock++) {
		uint8_t mask = 0;

		// Check all BK columns in this K-block
		for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
			if (A[row * K + kBlock * BK + dotIdx] != 0.0f) {
				mask |= (1 << dotIdx);
			}
		}

		// Store mask: simple row-major layout
		A_LIST[row * numKBlocks + kBlock] = mask;
	}
}
