#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Standalone tuning harness for row-level preprocessor
 * Tests different BK and NUM_THREADS configurations
 */

// These will be overridden by the tuner
#ifndef TUNE_BK
#define TUNE_BK 8
#endif

#ifndef TUNE_NUM_THREADS
#define TUNE_NUM_THREADS 256
#endif

extern "C" __global__ void __launch_bounds__(TUNE_NUM_THREADS)
    preprocess_A_rowlevel_tune(int M, int N, int K, float *A, uint8_t* A_LIST) {

    constexpr int BK = TUNE_BK;
    constexpr int NUM_THREADS = TUNE_NUM_THREADS;
    constexpr int WARP_SIZE = 32;
    const uint numKBlocks = K / BK;

    // Warp-level configuration
    const uint warpId = threadIdx.x / WARP_SIZE;
    const uint laneId = threadIdx.x % WARP_SIZE;

    // Calculate how many rows each warp processes simultaneously
    constexpr int ROWS_PER_WARP = WARP_SIZE / BK;
    constexpr int THREADS_PER_ROW = BK;

    // Total rows processed per block
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * (NUM_THREADS / WARP_SIZE);

    // Each block processes ROWS_PER_BLOCK rows
    for (uint rowBlockBase = blockIdx.x * ROWS_PER_BLOCK;
         rowBlockBase < M;
         rowBlockBase += gridDim.x * ROWS_PER_BLOCK) {

        // Which row within the warp's rows does this thread work on?
        const uint localRowInWarp = laneId / THREADS_PER_ROW;
        const uint threadPosInRow = laneId % THREADS_PER_ROW;

        // Global row index for this thread
        const uint row = rowBlockBase + warpId * ROWS_PER_WARP + localRowInWarp;

        if (row >= M) return;

        // Process K-blocks with loop unrolling for better ILP
        #pragma unroll 4
        for (uint kBlock = 0; kBlock < numKBlocks; kBlock++) {

            // Each thread reads one element from the K-block
            const uint kOffset = kBlock * BK + threadPosInRow;
            const float val = __ldg(&A[row * K + kOffset]);

            // Use __ballot_sync to create bitmask directly from predicate
            const uint32_t ballot = __ballot_sync(0xffffffff, val != 0.0f);

            // Extract the relevant bits for this row's mask
            uint8_t mask;
            if constexpr (BK == 8) {
                const uint shift = localRowInWarp * 8;
                mask = (ballot >> shift) & 0xFF;
            } else if constexpr (BK == 16) {
                const uint shift = localRowInWarp * 16;
                mask = (ballot >> shift) & 0xFFFF;
            }

            // First thread in each row group writes the result
            if (threadPosInRow == 0) {
                A_LIST[row * numKBlocks + kBlock] = mask;
            }
        }
    }
}
