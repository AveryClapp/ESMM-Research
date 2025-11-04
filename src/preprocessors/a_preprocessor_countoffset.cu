#pragma once

/* Count+Offset Preprocessor for A Matrix - Compact Sparsity Encoding */

#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * Encodes sparsity as: count (1 byte) + up to 4 offsets (4 bytes)
 * For BK=8: Each block stores how many non-zeros and their positions
 *
 * Memory layout per row per K-block (5 bytes total):
 *   [count (1 byte)][offset0 (1 byte)][offset1 (1 byte)][offset2 (1 byte)][offset3 (1 byte)]
 *
 * For patterns with >4 non-zeros, we use all 8 positions (store count only)
 */

struct CountOffset {
    uint8_t count;      // Number of non-zero elements (0-8)
    uint8_t offsets[4]; // Positions of first 4 non-zeros (0-7 for BK=8)
};

/*
 * Preprocesses matrix A to extract count+offset metadata
 * Each row gets one CountOffset struct per K-block
 *
 * @param M Number of rows in A
 * @param N Number of columns in output C (unused, for consistency)
 * @param K Number of columns in A
 * @param A Input matrix A (M x K)
 * @param metadata Output buffer: M * (K/BK) CountOffset structs
 */
template <const int BK, const int NUM_THREADS>
__global__ void preprocess_A_countoffset(
    int M, int N, int K,
    const float* __restrict__ A,
    CountOffset* __restrict__ metadata
) {
    const int numKBlocks = K / BK;
    const int WARP_SIZE = 32;
    const int ROWS_PER_WARP = WARP_SIZE / BK;
    const int ROWS_PER_BLOCK = ROWS_PER_WARP * (NUM_THREADS / WARP_SIZE);

    const int blockStartRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each group of BK lanes processes one row
    const int rowsPerWarp = WARP_SIZE / BK;
    const int rowInWarp = laneId / BK;
    const int colInRow = laneId % BK;

    const int localRow = warpId * rowsPerWarp + rowInWarp;
    const int globalRow = blockStartRow + localRow;

    if (globalRow >= M) return;

    // Process all K-blocks for this row
    for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
        const int globalCol = kBlock * BK + colInRow;
        const float value = A[globalRow * K + globalCol];

        // Each thread checks if its element is non-zero
        const bool isNonZero = (value != 0.0f);

        // Count non-zeros using ballot
        const unsigned mask = __ballot_sync(0xFFFFFFFF, isNonZero);

        // Extract count and offsets (first lane in each row-group does this)
        if (colInRow == 0) {
            CountOffset co;

            // Count bits in the relevant BK-bit section
            const int shift = rowInWarp * BK;
            const unsigned rowMask = (mask >> shift) & ((1u << BK) - 1);
            co.count = __popc(rowMask);

            // Extract offsets of non-zero positions
            int offsetIdx = 0;
            for (int i = 0; i < BK && offsetIdx < 4; i++) {
                if (rowMask & (1u << i)) {
                    co.offsets[offsetIdx++] = i;
                }
            }

            // Zero out unused offset slots
            for (int i = offsetIdx; i < 4; i++) {
                co.offsets[i] = 0;
            }

            // Write to global memory
            metadata[globalRow * numKBlocks + kBlock] = co;
        }
    }
}
