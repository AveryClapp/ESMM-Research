#pragma once

/*
 * ============================================================================
 * B-Matrix Transpose + Pattern Preprocessing (Option A: Explicit Transpose)
 * ============================================================================
 *
 * Strategy:
 *   1. Transpose B[K×N] → B_T[N×K] in global memory
 *   2. Analyze B_T row-wise for sparsity patterns
 *   3. Store both B_T and patterns for runtime kernel
 *
 * Pattern Encoding:
 *   - B_T is [N×K] in row-major
 *   - Group into [N/WN] × [K/BK] blocks
 *   - Each pattern: WN rows × BK columns
 *   - Bit k = 1 if ANY of WN rows has non-zero at K-position k
 *   - Granularity: WN=32 rows × BK=8 columns → 1 byte
 *
 * Memory:
 *   - B_T: N×K floats (same size as B, just transposed)
 *   - Patterns: (N/WN) × (K/BK) bytes (~64 KB for 4096×4096)
 */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Transpose Kernel: B[K×N] → B_T[N×K]
// ============================================================================

template <const int TILE_DIM, const int BLOCK_ROWS>
__global__ void transpose_b_matrix(
    const float* __restrict__ B,
    float* __restrict__ B_T,
    int K, int N
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    // Read from B[K×N] with coalesced access
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < K && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = B[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Write to B_T[N×K] with coalesced access (transposed coordinates)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < N && x < K) {
            B_T[(y + j) * K + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Pattern Preprocessing: Analyze B_T row-wise (like A-sparse)
// ============================================================================

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_bt_row_patterns(
    int N, int K,
    const float* __restrict__ B_T,  // [N×K] row-major
    uint8_t* __restrict__ rowPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numNBlocks = N / WN;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each block processes one N-block (WN rows)
    const int nBlock = blockIdx.x;
    if (nBlock >= numNBlocks) return;

    const int globalNBase = nBlock * WN;

    // Each warp processes multiple K-blocks
    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        uint8_t threadPattern = 0;

        const int globalKBase = kBlock * BK;

        constexpr int ELEMENTS_PER_WARP = WN * BK;  // 32 × 8 = 256 elements
        constexpr int ELEMENTS_PER_THREAD = (ELEMENTS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;  // 8 elements per thread

        // Each thread checks a subset of the WN×BK block
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId + i * WARP_SIZE;
            if (flatIdx < ELEMENTS_PER_WARP) {
                const int nRow = flatIdx / BK;  // Which N-row within block (0-31)
                const int kCol = flatIdx % BK;  // Which K-col within block (0-7)

                const int globalRow = globalNBase + nRow;
                const int globalCol = globalKBase + kCol;

                if (globalRow < N && globalCol < K) {
                    // B_T[n, k] access: row n, column k
                    float val = B_T[globalRow * K + globalCol];
                    if (val != 0.0f) {
                        // Set bit for this K-position
                        threadPattern |= (1 << kCol);
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
            rowPatterns[outIdx] = blockPattern;
        }
    }
}

// ============================================================================
// Metadata Structure
// ============================================================================

struct BTransposeMetadata {
    float* d_B_T;              // Transposed B matrix [N×K]
    uint8_t* d_rowPatterns;    // Patterns for each WN×BK block
    int numNBlocks;            // N / WN
    int numKBlocks;            // K / BK
};

void free_b_transpose_metadata(BTransposeMetadata& meta) {
    if (meta.d_B_T) {
        cudaFree(meta.d_B_T);
        meta.d_B_T = nullptr;
    }
    if (meta.d_rowPatterns) {
        cudaFree(meta.d_rowPatterns);
        meta.d_rowPatterns = nullptr;
    }
}

// ============================================================================
// Main Preprocessing Function
// ============================================================================

BTransposeMetadata preprocess_b_transpose_simple(float* d_B, int K, int N, int BK, int WN) {
    BTransposeMetadata meta;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;

    // Allocate B_T
    cudaMalloc(&meta.d_B_T, N * K * sizeof(float));

    // Allocate patterns
    const int totalBlocks = meta.numNBlocks * meta.numKBlocks;
    cudaMalloc(&meta.d_rowPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_rowPatterns, 0, totalBlocks * sizeof(uint8_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Transpose B → B_T
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid((N + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM);

    transpose_b_matrix<TILE_DIM, BLOCK_ROWS><<<transposeGrid, transposeBlock>>>(d_B, meta.d_B_T, K, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B transpose kernel error: %s\n", cudaGetErrorString(error));
    }

    // Step 2: Analyze patterns on B_T
    constexpr int NUM_THREADS = 256;
    const int numBlocks = meta.numNBlocks;
    dim3 patternBlock(NUM_THREADS);
    dim3 patternGrid(numBlocks);

    if (BK == 8 && WN == 32) {
        preprocess_bt_row_patterns<8, 32, NUM_THREADS>
            <<<patternGrid, patternBlock>>>(N, K, meta.d_B_T, meta.d_rowPatterns);
    } else {
        printf("Error: Unsupported BK=%d, WN=%d combination for B transpose preprocessing\n", BK, WN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B transpose pattern preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    printf("B-matrix transpose + pattern preprocessing: %d blocks (%.1f KB metadata) in %.3f ms\n",
           totalBlocks, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
