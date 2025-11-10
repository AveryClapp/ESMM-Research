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
 * B Column-wise Pattern Preprocessing (No Transpose!)
 *
 * Analyzes B (K×N) columns in WN-wide vertical strips (WN columns × BK rows)
 * Creates 8-bit patterns indicating which of BK=8 elements are non-zero
 * All threads in warp see same pattern → enables warp-uniform execution
 *
 * Conceptually: treats WN consecutive columns as a "logical row" for pattern analysis
 * Output: (N/WN) × (K/BK) patterns (same as before, but reading columns not rows)
 */
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_column_patterns(
    int K, int N,
    const float* __restrict__ B,
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
    const int globalWarpCol = blockIdx.x * (NUM_THREADS / WARP_SIZE) + warpId;

    if (globalWarpCol >= numNBlocks) return;

    const int globalColBase = globalWarpCol * WN;

    // Process K-blocks in batches of 4
    for (int kBlockBatch = 0; kBlockBatch < numKBlocks; kBlockBatch += 4) {
        constexpr int LOADS_PER_THREAD = (TILE_N * TILE_K) / NUM_THREADS;

        // Load tile from B (K×N) - reading columns, which is strided
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; i++) {
            const int flatIdx = threadIdx.x + i * NUM_THREADS;
            const int k = flatIdx / TILE_N;  // which k within tile
            const int n = flatIdx % TILE_N;  // which n within tile

            const int globalK = kBlockBatch * BK + k;
            const int globalN = globalColBase + n;

            if (globalK < K && globalN < N) {
                smem[k][n] = B[globalK * N + globalN];  // Reading columns (strided)
            } else {
                smem[k][n] = 0.0f;
            }
        }

        __syncthreads();

        // Each warp processes WN columns (32 columns for WN=32)
        // All threads in warp OR together patterns across their WN columns
        #pragma unroll
        for (int kb = 0; kb < 4 && (kBlockBatch + kb) < numKBlocks; kb++) {
            uint8_t threadPattern = 0;
            const int kOffset = kb * BK;

            // Each thread checks BK=8 elements from its assigned column
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                if (laneId < WN && smem[kOffset + k][laneId] != 0.0f) {
                    threadPattern |= (1 << k);
                }
            }

            // Warp-level reduction: OR all patterns together
            // Result: pattern representing OR of all WN columns
            uint8_t warpPattern = threadPattern;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                warpPattern |= __shfl_xor_sync(0xFFFFFFFF, warpPattern, offset);
            }

            // First thread writes result
            if (laneId == 0) {
                const int outIdx = globalWarpCol * numKBlocks + kBlockBatch + kb;
                blockPatterns[outIdx] = warpPattern;
            }
        }

        __syncthreads();
    }
}

/*
 * B Pattern Preprocessing (No Transpose!)
 *
 * 1. Analyze B (K×N) column-wise sparsity patterns
 * 2. Return metadata for runtime kernel
 */
struct BTPatternMetadata {
    float* d_BT;               // NOW UNUSED - kept for API compatibility, set to nullptr
    uint8_t* d_blockPatterns;  // Pattern for each BK×WN block
    int numNBlocks;            // Number of N-blocks (N / WN)
    int numKBlocks;            // Number of K-blocks (K / BK)
};

inline void free_bt_pattern_metadata(BTPatternMetadata& meta) {
    // d_BT is no longer allocated, so skip it
    if (meta.d_blockPatterns) {
        cudaFree(meta.d_blockPatterns);
        meta.d_blockPatterns = nullptr;
    }
}

BTPatternMetadata preprocess_b_transpose(float* d_B, int K, int N, int WN, int BK) {
    BTPatternMetadata meta;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;
    meta.d_BT = nullptr;  // No longer needed!

    // Analyze B column-wise patterns directly (no transpose!)
    const int totalBlocks = meta.numNBlocks * meta.numKBlocks;
    cudaMalloc(&meta.d_blockPatterns, totalBlocks * sizeof(uint8_t));
    cudaMemset(meta.d_blockPatterns, 0, totalBlocks * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;
    const int WARPS_PER_BLOCK = NUM_THREADS / 32;
    const int numBlocks = CEIL_DIV(meta.numNBlocks, WARPS_PER_BLOCK);

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(numBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (BK == 8 && WN == 32) {
        preprocess_b_column_patterns<8, 32, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_blockPatterns);
    } else if (BK == 8 && WN == 64) {
        preprocess_b_column_patterns<8, 64, NUM_THREADS>
            <<<gridDim, blockDim>>>(K, N, d_B, meta.d_blockPatterns);
    } else {
        printf("Error: Unsupported BK=%d, WN=%d combination for B preprocessing\n", BK, WN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pattern_ms = 0;
    cudaEventElapsedTime(&pattern_ms, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("B pattern preprocessing kernel error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalBlocks / 1024.0f;

    printf("B preprocessing (no transpose!):\n");
    printf("  Patterns:  %.3f ms (%d blocks, %.1f KB metadata)\n",
           pattern_ms, totalBlocks, metadataKB);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
