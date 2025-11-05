#pragma once

/* GPU Preprocessor for B-Matrix Sparsity (Column-wise patterns) */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * B-Matrix GPU Preprocessing Kernel
 *
 * Analyzes B matrix (K × N) in blocks of BK × TN (typically 8 × 8)
 * Creates 8-bit patterns indicating which of TN columns are non-zero
 *
 * Key optimizations:
 * 1. Coalesced memory access
 * 2. Warp-level reduction using shuffle
 * 3. Process multiple N-blocks in one pass
 *
 * Memory layout:
 * - B is K × N, row-major
 * - Output: (K/BK) × (N/TN) patterns, each 1 byte
 */
template <const int BK, const int TN, const int NUM_THREADS>
__global__ void preprocess_b_patterns(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ blockPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int TILE_K = BK;      // 8 K-rows
    constexpr int TILE_N = TN * 4;  // Process 4 N-blocks at once (32 cols for TN=8)

    // Shared memory for tile: [TILE_K][TILE_N]
    __shared__ float smem[TILE_K][TILE_N];

    const int numKBlocks = K / BK;
    const int numNBlocks = N / TN;

    const int laneId = threadIdx.x % WARP_SIZE;

    // Each block processes one K-block across multiple N-blocks
    const int kBlock = blockIdx.x;
    if (kBlock >= numKBlocks) return;

    const int globalKBase = kBlock * BK;

    // Process N-blocks in batches of 4
    for (int nBlockBatch = 0; nBlockBatch < numNBlocks; nBlockBatch += 4) {
        // Load data into shared memory with coalesced access
        constexpr int ELEMENTS_PER_THREAD = (TILE_K * TILE_N) / NUM_THREADS;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = threadIdx.x + i * NUM_THREADS;
            const int row = flatIdx / TILE_N;
            const int col = flatIdx % TILE_N;

            const int globalRow = globalKBase + row;
            const int globalCol = nBlockBatch * TN + col;

            if (globalRow < K && globalCol < N) {
                smem[row][col] = B[globalRow * N + globalCol];
            } else {
                smem[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // Now compute patterns for up to 4 N-blocks
        #pragma unroll
        for (int nb = 0; nb < 4 && (nBlockBatch + nb) < numNBlocks; nb++) {
            uint8_t threadPattern = 0;

            // Each thread processes columns from one N-block
            if (laneId < BK) {  // Only first BK lanes participate
                const int kRow = laneId;
                const int nOffset = nb * TN;

                // OR together TN columns to create pattern
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    if (smem[kRow][nOffset + n] != 0.0f) {
                        threadPattern |= (1 << n);
                    }
                }
            }

            // Warp-level reduction: OR all row patterns together
            uint8_t blockPattern = threadPattern;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                blockPattern |= __shfl_xor_sync(0xFFFFFFFF, blockPattern, offset);
            }

            // First thread writes result
            if (laneId == 0) {
                const int outIdx = kBlock * numNBlocks + nBlockBatch + nb;
                blockPatterns[outIdx] = blockPattern;
            }
        }

        __syncthreads();
    }
}

/*
 * GPU-based preprocessing for B-matrix patterns
 */
BMatrixPatternMetadata analyze_b_sparsity_pattern_gpu(float* d_B, int K, int N, int BK, int TN) {
    BMatrixPatternMetadata meta;
    meta.numKBlocks = K / BK;
    meta.numNBlocks = N / TN;

    const int totalBlocks = meta.numKBlocks * meta.numNBlocks;

    // Allocate device memory for patterns
    cudaMalloc(&meta.d_blockPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_blockPatterns, 0, totalBlocks * sizeof(uint8_t));

    // Launch preprocessing kernel
    constexpr int NUM_THREADS = 256;
    const int numBlocks = meta.numKBlocks;

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(numBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (BK == 8 && TN == 8) {
        preprocess_b_patterns<8, 8, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_blockPatterns);
    } else {
        printf("Error: Unsupported BK=%d, TN=%d combination for B preprocessing\n", BK, TN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    printf("B-matrix GPU preprocessing: %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
