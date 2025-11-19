#pragma once

/*
 * ============================================================================
 * B-Matrix GPU Preprocessor: WARP-GRANULARITY PATTERNS
 * ============================================================================
 *
 * Strategy:
 *   Create one pattern per BK×WN block (aligned with warp execution)
 *   This enables warp-uniform skipping with ZERO divergence
 *
 * Pattern Encoding:
 *   - Each 8-bit pattern represents BK K-rows across WN columns
 *   - Bit k = 1 if ANY of the WN columns has non-zero in K-row k
 *   - Granularity: BK=8 rows × WN=32 columns = 256 elements → 1 byte
 *
 * Memory Layout:
 *   - B matrix: K × N (row-major)
 *   - Output: (K/BK) × (N/WN) patterns
 *   - For 4096×4096: (4096/8) × (4096/32) = 512 × 128 = 65,536 bytes = 64 KB
 *
 * Why This Works:
 *   - Each warp processes WN=32 consecutive columns
 *   - All 32 threads share ONE pattern → warp-uniform skipping
 *   - All threads skip the same K-iterations together → zero divergence
 *
 * Tradeoff:
 *   - Coarser granularity than TN-based patterns
 *   - If ANY column is non-zero, ALL columns are computed
 *   - Works best with structured/column-aligned sparsity
 */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_patterns_warp_granularity(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ blockPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numKBlocks = K / BK;
    const int numNBlocks = N / WN;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each block processes one K-block
    const int kBlock = blockIdx.x;
    if (kBlock >= numKBlocks) return;

    const int globalKBase = kBlock * BK;

    // Each warp processes multiple N-blocks (WN-wide columns)
    for (int nBlock = warpId; nBlock < numNBlocks; nBlock += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        const int globalNBase = nBlock * WN;

        constexpr int ELEMENTS_PER_WARP = BK * WN;  // 8 × 32 = 256 elements
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;  // 8 elements per thread

        // Each thread checks a subset of the BK×WN block
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId + i * WARP_SIZE;
            if (flatIdx < ELEMENTS_PER_WARP) {
                const int kRow = flatIdx / WN;  // Which K-row within block (0-7)
                const int nCol = flatIdx % WN;  // Which N-col within block (0-31)

                const int globalRow = globalKBase + kRow;
                const int globalCol = globalNBase + nCol;

                if (globalRow < K && globalCol < N) {
                    float val = B[globalRow * N + globalCol];
                    if (val != 0.0f) {
                        // Set bit for this K-row
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

BMatrixPatternMetadata analyze_b_sparsity_warp_granularity(float* d_B, int K, int N, int BK, int WN) {
    BMatrixPatternMetadata meta;
    meta.numKBlocks = K / BK;
    meta.numNBlocks = N / WN;  // Changed from TN to WN

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

    if (BK == 8 && WN == 32) {
        preprocess_b_patterns_warp_granularity<8, 32, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_blockPatterns);
    } else {
        printf("Error: Unsupported BK=%d, WN=%d combination for B warp preprocessing\n", BK, WN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B warp-granularity preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    printf("B-matrix GPU preprocessing (warp-granularity): %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
