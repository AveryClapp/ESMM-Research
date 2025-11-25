#pragma once

/*
 * ============================================================================
 * B-Matrix GPU Preprocessor: TN-GRANULARITY PATTERNS (8-column blocks)
 * ============================================================================
 *
 * Strategy:
 *   Create one pattern per BK×TN block (per thread-group granularity)
 *   Each pattern covers exactly the 8 columns that a thread group processes
 *
 * Pattern Encoding:
 *   - Each 8-bit pattern represents BK K-rows across TN columns
 *   - Bit k = 1 if ANY of the TN columns has non-zero in K-row k
 *   - Granularity: BK=8 rows × TN=8 columns = 64 elements → 1 byte
 *
 * Memory Layout:
 *   - B matrix: K × N (row-major)
 *   - Output: (K/BK) × (N/TN) patterns
 *   - For 4096×4096: (4096/8) × (4096/8) = 512 × 512 = 262,144 bytes = 256 KB
 *
 * Why This Works Better:
 *   - Each thread group processes exactly TN=8 consecutive columns
 *   - Pattern matches exactly what each group needs to know
 *   - Different groups can have different patterns → finer granularity
 *
 * Tradeoff vs WN-granularity:
 *   - PRO: Finer granularity (4× more patterns per row)
 *   - PRO: More precise (only computes what's needed per group)
 *   - CON: 4× more metadata (256 KB vs 64 KB for 4096×4096)
 *   - CON: Potential warp divergence (different groups may diverge)
 *   - CON: More pattern reads per warp (WNITER reads instead of 1)
 */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

template <const int BK, const int TN, const int NUM_THREADS>
__global__ void preprocess_b_patterns_tn_granularity(
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

    // Each block processes one K-block
    const int kBlock = blockIdx.x;
    if (kBlock >= numKBlocks) return;

    const int globalKBase = kBlock * BK;

    // Each warp processes multiple N-blocks (TN-wide columns)
    for (int nBlock = warpId; nBlock < numNBlocks; nBlock += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        const int globalNBase = nBlock * TN;

        constexpr int ELEMENTS_PER_WARP = BK * TN;  // 8 × 8 = 64 elements
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;  // 2 elements per thread

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
                        // Set bit for this column (across all K-rows)
                        threadPattern |= (1 << nCol);
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

BMatrixPatternMetadata analyze_b_sparsity_tn_granularity(float* d_B, int K, int N, int BK, int TN) {
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
        preprocess_b_patterns_tn_granularity<8, 8, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_blockPatterns);
    } else {
        printf("Error: Unsupported BK=%d, TN=%d combination for B TN preprocessing\n", BK, TN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B TN-granularity preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    printf("B-matrix GPU preprocessing (TN-granularity): %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
