#pragma once

/*
 * ============================================================================
 * B-Matrix Preprocessor: WN-Granularity Column Patterns
 * ============================================================================
 *
 * Strategy:
 *   Analyze B column-wise at WN-granularity (warp-level)
 *   Each pattern represents: WN columns × BK rows
 *   This matches how warps access B in the GEMM kernel
 *
 * Pattern Encoding:
 *   - B is [K×N] in row-major
 *   - Group into [K/BK] × [N/WN] blocks
 *   - Each pattern: BK rows × WN columns
 *   - Bit k = 1 if ANY of WN columns has non-zero at K-position k
 *   - Granularity: BK=8 rows × WN=32 columns → 1 byte
 *
 * Storage Layout:
 *   - Column-major pattern storage
 *   - Pattern[nBlock * numKBlocks + kBlock]
 *   - nBlock = column block index (0..N/WN-1)
 *   - kBlock = row block index (0..K/BK-1)
 *
 * Memory:
 *   - Patterns: (N/WN) × (K/BK) bytes
 *   - Example: 4096×4096 → (4096/32) × (4096/8) = 128 × 512 = 65,536 bytes
 */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Pattern Preprocessing Kernel: Analyze B column-wise
// ============================================================================

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_wn_column_patterns(
    int K, int N,
    const float* __restrict__ B,  // [K×N] row-major
    uint8_t* __restrict__ colPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numNBlocks = N / WN;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each block processes one column block (WN columns)
    const int nBlock = blockIdx.x;
    if (nBlock >= numNBlocks) return;

    const int globalNBase = nBlock * WN;

    // Each warp processes multiple K-blocks
    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        const int globalKBase = kBlock * BK;

        // Each warp analyzes a WN×BK block (32 columns × 8 rows = 256 elements)
        constexpr int ELEMENTS_PER_WARP = WN * BK;  // 256
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;  // 8

        // Each thread checks a subset of the WN×BK block
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId + i * WARP_SIZE;
            if (flatIdx < ELEMENTS_PER_WARP) {
                const int kRow = flatIdx / WN;  // Which K-row within block (0-7)
                const int nCol = flatIdx % WN;  // Which N-col within block (0-31)

                const int globalRow = globalKBase + kRow;
                const int globalCol = globalNBase + nCol;

                if (globalRow < K && globalCol < N) {
                    // B[k][n] access: row k, column n
                    float val = B[globalRow * N + globalCol];
                    if (val != 0.0f) {
                        // Set bit for this K-position
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

        // Write final pattern: one per (nBlock, kBlock) pair
        if (laneId == 0) {
            const int outIdx = nBlock * numKBlocks + kBlock;
            colPatterns[outIdx] = blockPattern;
        }
    }
}

// ============================================================================
// Metadata Structure
// ============================================================================

struct BColumnPatternMetadata {
    uint8_t* d_colPatterns;    // Patterns for each WN×BK block
    int numNBlocks;            // N / WN
    int numKBlocks;            // K / BK
};

void free_b_column_pattern_metadata(BColumnPatternMetadata& meta) {
    if (meta.d_colPatterns) {
        cudaFree(meta.d_colPatterns);
        meta.d_colPatterns = nullptr;
    }
}

// ============================================================================
// Main Preprocessing Function
// ============================================================================

BColumnPatternMetadata analyze_b_column_patterns_gpu(float* d_B, int K, int N, int WN, int BK) {
    BColumnPatternMetadata meta;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;

    // Allocate patterns
    const int totalBlocks = meta.numNBlocks * meta.numKBlocks;
    cudaMalloc(&meta.d_colPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_colPatterns, 0, totalBlocks * sizeof(uint8_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch preprocessing kernel
    constexpr int NUM_THREADS = 256;
    const int numBlocks = meta.numNBlocks;
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(numBlocks);

    if (BK == 8 && WN == 32) {
        preprocess_b_wn_column_patterns<8, 32, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_colPatterns);
    } else {
        printf("Error: Unsupported BK=%d, WN=%d combination for B column preprocessing\n", BK, WN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B column pattern preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    printf("B-column pattern preprocessing: %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
