#pragma once

/* Optimized Block-wise Preprocessor: Coalesced memory access via shared memory */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * Optimized GPU Preprocessing Kernel
 *
 * Key optimizations:
 * 1. Shared memory transpose for coalesced global memory access
 * 2. Process multiple K-blocks in one pass (batches of 4)
 * 3. Efficient warp-level reduction using shuffle
 * 4. Bank conflict avoidance with padding
 *
 * Memory access pattern:
 * - Coalesced reads: consecutive threads read consecutive addresses
 * - Transpose in shared memory
 * - Compute patterns from transposed data
 */
template <const int BK, const int WM, const int NUM_THREADS>
__global__ void preprocess_blockwise_patterns(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ blockPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int TILE_M = WM;
    constexpr int TILE_K = BK * 4;

    __shared__ float smem[TILE_K][TILE_M + 1];

    const int numKBlocks = K / BK;
    const int numWarpRows = M / WM;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int globalWarpRow = blockIdx.x * (NUM_THREADS / WARP_SIZE) + warpId;

    if (globalWarpRow >= numWarpRows) return;

    const int globalRowBase = globalWarpRow * WM;

    for (int kBlockBatch = 0; kBlockBatch < numKBlocks; kBlockBatch += 4) {
        constexpr int LOADS_PER_THREAD = (TILE_M * TILE_K) / NUM_THREADS;

        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; i++) {
            const int flatIdx = threadIdx.x + i * NUM_THREADS;
            const int row = flatIdx / TILE_K;
            const int col = flatIdx % TILE_K;

            const int globalRow = globalRowBase + row;
            const int globalCol = kBlockBatch * BK + col;

            if (globalRow < M && globalCol < K) {
                smem[col][row] = A[globalRow * K + globalCol];
            } else {
                smem[col][row] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kb = 0; kb < 4 && (kBlockBatch + kb) < numKBlocks; kb++) {
            uint8_t threadPattern = 0;

            if (laneId < TILE_M) {
                const int kOffset = kb * BK;

                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (smem[kOffset + k][laneId] != 0.0f) {
                        threadPattern |= (1 << k);
                    }
                }

            uint8_t warpPattern = threadPattern;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                warpPattern |= __shfl_xor_sync(0xFFFFFFFF, warpPattern, offset);
            }

            if (laneId == 0) {
                const int outIdx = globalWarpRow * numKBlocks + kBlockBatch + kb;
                blockPatterns[outIdx] = warpPattern;
            }
        }

        __syncthreads();
    }
}

/*
 * GPU-based preprocessing for block-wise patterns
 */
BlockPatternMetadata analyze_sparsity_pattern_gpu(float* d_A, int M, int K, int WM, int BK) {
    BlockPatternMetadata meta;
    meta.numWarpRows = M / WM;
    meta.numKBlocks = K / BK;

    const int totalBlocks = meta.numWarpRows * meta.numKBlocks;

    cudaMalloc(&meta.d_blockPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_blockPatterns, 0, totalBlocks * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;
    const int WARPS_PER_BLOCK = NUM_THREADS / 32;
    const int numBlocks = CEIL_DIV(meta.numWarpRows, WARPS_PER_BLOCK);

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(numBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    preprocess_blockwise_patterns<8, 32, NUM_THREADS>
        <<<gridDim, blockDim>>>(M, K, d_A, meta.d_blockPatterns);
  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
