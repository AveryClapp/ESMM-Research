#pragma once

/* GPU Preprocessor for B-Matrix Sparsity (K-wise patterns per N-column) */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * B-Matrix GPU Preprocessing Kernel (K-WISE PATTERNS)
 *
 * CRITICAL: This preprocessor creates patterns across the K dimension!
 *
 * For each N-column group (width TN), creates an 8-bit pattern indicating
 * which K-rows (within a BK block) are non-zero.
 *
 * Pattern encoding:
 * - Bit i is set if ANY element in K-row i (across TN columns) is non-zero
 * - This allows the kernel to skip K iterations where B has all zeros
 *
 * Memory layout:
 * - B is K × N, row-major
 * - Output: (K/BK) × (N/TN) patterns
 *   Each pattern represents which K-rows are non-zero for that N-column group
 *
 * Example: For a BK=8, TN=8 block at (kBlock=0, nBlock=0):
 *   Pattern bit 0 = 1 if any B[0, 0:8] is non-zero
 *   Pattern bit 1 = 1 if any B[1, 0:8] is non-zero
 *   ...
 *   Pattern bit 7 = 1 if any B[7, 0:8] is non-zero
 */
template <const int BK, const int TN, const int NUM_THREADS>
__global__ void preprocess_b_patterns(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ blockPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numKBlocks = K / BK;
    const int numNBlocks = N / TN;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int kBlock = blockIdx.x;
    if (kBlock >= numKBlocks) return;

    const int globalKBase = kBlock * BK;

    // Each warp processes multiple N-blocks
    for (int nBlock = warpId; nBlock < numNBlocks; nBlock += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        const int globalNBase = nBlock * TN;

        constexpr int ELEMENTS_PER_WARP = BK * TN;
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;

        // Each thread checks a subset of the BK×TN block
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId + i * WARP_SIZE;
            if (flatIdx < ELEMENTS_PER_WARP) {
                const int kRow = flatIdx / TN;  // Which K-row within block (0-7)
                const int nCol = flatIdx % TN;  // Which N-col within block (0-7)

                const int globalRow = globalKBase + kRow;
                const int globalCol = globalNBase + nCol;

                if (globalRow < K && globalCol < N) {
                    float val = B[globalRow * N + globalCol];
                    if (val != 0.0f) {
                        // Set bit kRow (not nCol!) - this is the KEY change
                        threadPattern |= (1 << kRow);
                    }
                }
            }
        }

        // Combine patterns across warp using OR reduction
        uint8_t blockPattern = threadPattern;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockPattern |= __shfl_xor_sync(0xFFFFFFFF, blockPattern, offset);
        }

        // Write final pattern: one per (kBlock, nBlock) pair
        if (laneId == 0) {
            const int outIdx = kBlock * numNBlocks + nBlock;
            blockPatterns[outIdx] = blockPattern;
        }
    }
}

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
    printf("B-matrix GPU preprocessing (K-wise): %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
