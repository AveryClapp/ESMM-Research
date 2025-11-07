#pragma once

/* B-Matrix Transpose and Preprocessing for Row-wise Sparsity */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * Transpose Kernel: B (K×N) → B^T (N×K)
 *
 * Uses shared memory tiling for efficient transpose:
 * - Coalesced reads from B
 * - Coalesced writes to B^T
 * - Bank conflict avoidance with padding
 */
template <const int TILE_DIM, const int BLOCK_ROWS>
__global__ void transpose_matrix(
    int K, int N,
    const float* __restrict__ B,
    float* __restrict__ BT
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    // Read B (K×N) with coalesced access
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        if (row < K && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = B[row * N + x];
        }
    }

    __syncthreads();

    // Write B^T (N×K) with coalesced access (transposed coordinates)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        if (row < N && x < K) {
            BT[row * K + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/*
 * B^T Row-wise Pattern Preprocessing
 *
 * Analyzes B^T (N×K) in 8×WN blocks (BK × warp-width)
 * Creates 8-bit patterns indicating which of BK=8 elements are non-zero
 * All threads in warp see same pattern → enables warp-uniform execution
 *
 * Output: (N/WN) × (K/BK) patterns
 */
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_bt_rowwise_patterns(
    int N, int K,
    const float* __restrict__ BT,
    uint8_t* __restrict__ blockPatterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int TILE_N = WN;
    constexpr int TILE_K = BK * 4; // Process 4 K-blocks at once

    __shared__ float smem[TILE_K][TILE_N + 1];

    const int numKBlocks = K / BK;
    const int numNBlocks = N / WN;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int globalWarpRow = blockIdx.x * (NUM_THREADS / WARP_SIZE) + warpId;

    if (globalWarpRow >= numNBlocks) return;

    const int globalRowBase = globalWarpRow * WN;

    // Process K-blocks in batches of 4
    for (int kBlockBatch = 0; kBlockBatch < numKBlocks; kBlockBatch += 4) {
        constexpr int LOADS_PER_THREAD = (TILE_N * TILE_K) / NUM_THREADS;

        // Load tile into shared memory with coalesced access
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; i++) {
            const int flatIdx = threadIdx.x + i * NUM_THREADS;
            const int row = flatIdx / TILE_K;
            const int col = flatIdx % TILE_K;

            const int globalRow = globalRowBase + row;
            const int globalCol = kBlockBatch * BK + col;

            if (globalRow < N && globalCol < K) {
                smem[col][row] = BT[globalRow * K + globalCol];
            } else {
                smem[col][row] = 0.0f;
            }
        }

        __syncthreads();

        // Each warp processes WN rows (32 rows for WN=32)
        // All threads in warp OR together patterns across their WN rows
        #pragma unroll
        for (int kb = 0; kb < 4 && (kBlockBatch + kb) < numKBlocks; kb++) {
            uint8_t threadPattern = 0;
            const int kOffset = kb * BK;

            // Each thread checks 8 elements from its assigned row
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                if (laneId < WN && smem[kOffset + k][laneId] != 0.0f) {
                    threadPattern |= (1 << k);
                }
            }

            // Warp-level reduction: OR all patterns together
            // Result: pattern representing OR of all WN rows
            uint8_t warpPattern = threadPattern;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                warpPattern |= __shfl_xor_sync(0xFFFFFFFF, warpPattern, offset);
            }

            // First thread writes result
            if (laneId == 0) {
                const int outIdx = globalWarpRow * numKBlocks + kBlockBatch + kb;
                blockPatterns[outIdx] = warpPattern;
            }
        }

        __syncthreads();
    }
}

/*
 * Complete B-Transpose Preprocessing Pipeline
 *
 * 1. Transpose B (K×N) to B^T (N×K)
 * 2. Analyze B^T row-wise sparsity patterns
 * 3. Return metadata for runtime kernel
 */
struct BTPatternMetadata {
    float* d_BT;               // Transposed matrix (N×K)
    uint8_t* d_blockPatterns;  // Pattern for each BK×WN block
    int numNBlocks;            // Number of N-blocks (N / WN)
    int numKBlocks;            // Number of K-blocks (K / BK)
};

inline void free_bt_pattern_metadata(BTPatternMetadata& meta) {
    if (meta.d_BT) {
        cudaFree(meta.d_BT);
        meta.d_BT = nullptr;
    }
    if (meta.d_blockPatterns) {
        cudaFree(meta.d_blockPatterns);
        meta.d_blockPatterns = nullptr;
    }
}

BTPatternMetadata preprocess_b_transpose(float* d_B, int K, int N, int WN, int BK) {
    BTPatternMetadata meta;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;

    // Allocate B^T
    cudaMalloc(&meta.d_BT, N * K * sizeof(float));

    // Step 1: Transpose B → B^T
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid(CEIL_DIV(N, TILE_DIM), CEIL_DIV(K, TILE_DIM));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    transpose_matrix<TILE_DIM, BLOCK_ROWS>
        <<<transposeGrid, transposeBlock>>>(K, N, d_B, meta.d_BT);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float transpose_ms = 0;
    cudaEventElapsedTime(&transpose_ms, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Transpose kernel error: %s\n", cudaGetErrorString(error));
    }

    // Step 2: Analyze B^T row-wise patterns
    const int totalBlocks = meta.numNBlocks * meta.numKBlocks;
    cudaMalloc(&meta.d_blockPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_blockPatterns, 0, totalBlocks * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;
    const int WARPS_PER_BLOCK = NUM_THREADS / 32;
    const int numBlocks = CEIL_DIV(meta.numNBlocks, WARPS_PER_BLOCK);

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(numBlocks);

    cudaEventRecord(start);

    if (BK == 8 && WN == 32) {
        preprocess_bt_rowwise_patterns<8, 32, NUM_THREADS>
            <<<gridDim, blockDim>>>(N, K, meta.d_BT, meta.d_blockPatterns);
    } else {
        printf("Error: Unsupported BK=%d, WN=%d combination for B^T preprocessing\n", BK, WN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pattern_ms = 0;
    cudaEventElapsedTime(&pattern_ms, start, stop);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B^T pattern preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;
    float transposeGB = (N * K * sizeof(float) * 2) / 1e9f; // Read B + Write B^T

    printf("B-Transpose preprocessing:\n");
    printf("  Transpose: %.3f ms (%.2f GB/s)\n",
           transpose_ms, transposeGB / (transpose_ms / 1000.0f));
    printf("  Patterns:  %.3f ms (%d blocks, %.1f KB metadata)\n",
           pattern_ms, totalBlocks, metadataKB);
    printf("  Total:     %.3f ms\n", transpose_ms + pattern_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
