#pragma once

/* GPU Preprocessor for B-Matrix Sparsity (Column-wise patterns) */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * B-Matrix GPU Preprocessing Kernel (OPTIMIZED)
 *
 * Analyzes B matrix (K × N) in blocks of BK × TN (typically 8 × 8)
 * Creates 8-bit patterns indicating which of TN columns are non-zero
 *
 * Key optimizations:
 * 1. Full warp utilization - all 32 threads active
 * 2. Each warp processes multiple N-blocks in parallel
 * 3. Coalesced memory access via shared memory
 * 4. Warp-level reductions for pattern combination
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
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numKBlocks = K / BK;
    const int numNBlocks = N / TN;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int kBlock = blockIdx.x;
    if (kBlock >= numKBlocks) return;

    const int globalKBase = kBlock * BK;

    for (int nBlockBase = warpId; nBlockBase < numNBlocks; nBlockBase += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        constexpr int ELEMENTS_PER_WARP = BK * TN;
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId + i * WARP_SIZE;
            if (flatIdx < ELEMENTS_PER_WARP) {
                const int kRow = flatIdx / TN;
                const int nCol = flatIdx % TN;

                const int globalRow = globalKBase + kRow;
                const int globalCol = nBlockBase * TN + nCol;

                if (globalRow < K && globalCol < N) {
                    float val = B[globalRow * N + globalCol];
                    if (val != 0.0f) {
                        threadPattern |= (1 << nCol);
                    }
                }
            }
        }

        uint8_t blockPattern = threadPattern;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockPattern |= __shfl_xor_sync(0xFFFFFFFF, blockPattern, offset);
        }

        if (laneId == 0) {
            const int outIdx = kBlock * numNBlocks + nBlockBase;
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
    printf("B-matrix GPU preprocessing: %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
