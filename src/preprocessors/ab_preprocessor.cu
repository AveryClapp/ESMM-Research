#pragma once

/*
 * ============================================================================
 * Joint A+B Preprocessor: Warp-Uniform Pattern Generation
 * ============================================================================
 *
 * A-Pattern (WM×BK blocks):
 *   - A is [M×K] row-major
 *   - Each pattern covers WM=64 rows × BK=8 columns
 *   - Bit k = 1 if ANY of WM rows has non-zero at column k
 *   - Indexed by: mBlock * numKBlocks + kBlock
 *   - Warp-uniform because all threads in warp share same warpRow
 *
 * B-Pattern (WN×BK blocks):
 *   - B is [K×N] row-major
 *   - Each pattern covers BK=8 rows × WN=32 columns
 *   - Bit k = 1 if ANY of WN columns has non-zero at row k
 *   - Indexed by: nBlock * numKBlocks + kBlock
 *   - Warp-uniform because all threads in warp share same warpCol
 *
 * Joint Pattern:
 *   joint = a_pattern & b_pattern
 *   - Bit k = 1 only if BOTH A and B have non-zeros at K-position k
 *   - Skip iteration when bit = 0: guaranteed zero contribution
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

struct ABPatternMetadata {
    uint8_t* d_a_patterns;  // [numMBlocks × numKBlocks]
    uint8_t* d_b_patterns;  // [numNBlocks × numKBlocks]
    int numMBlocks;         // M / WM
    int numNBlocks;         // N / WN
    int numKBlocks;         // K / BK
    float a_sparsity;       // Effective A skip rate
    float b_sparsity;       // Effective B skip rate
    float joint_sparsity;   // Expected joint skip rate
};

inline void free_ab_pattern_metadata(ABPatternMetadata& meta) {
    if (meta.d_a_patterns) cudaFree(meta.d_a_patterns);
    if (meta.d_b_patterns) cudaFree(meta.d_b_patterns);
    meta.d_a_patterns = nullptr;
    meta.d_b_patterns = nullptr;
}

// ============================================================================
// A-Pattern Preprocessing Kernel (Original - WM×BK granularity)
// ============================================================================

template <const int BK, const int WM, const int NUM_THREADS>
__global__ void preprocess_a_patterns_kernel(
    int M, int K,
    const float* __restrict__ A,    // [M×K] row-major
    uint8_t* __restrict__ patterns  // [numMBlocks × numKBlocks]
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numMBlocks = M / WM;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each thread block processes one M-block (WM=64 rows)
    const int mBlock = blockIdx.x;
    if (mBlock >= numMBlocks) return;

    const int globalMBase = mBlock * WM;

    // Each warp processes multiple K-blocks
    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        // WM×BK = 64×8 = 512 elements per block
        // 32 threads per warp → 16 elements per thread
        constexpr int ELEMENTS_PER_THREAD = (WM * BK) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId * ELEMENTS_PER_THREAD + i;
            const int mRow = flatIdx / BK;    // Which M-row within block (0-63)
            const int kCol = flatIdx % BK;    // Which K-col within block (0-7)

            const int globalM = globalMBase + mRow;
            const int globalK = globalKBase + kCol;

            if (globalM < M && globalK < K) {
                float val = A[globalM * K + globalK];
                if (val != 0.0f) {
                    threadPattern |= (1 << kCol);
                }
            }
        }

        // Warp-level OR reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
        }

        if (laneId == 0) {
            patterns[mBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// A-Pattern Preprocessing Kernel (8×8 granularity for true 8×32 tiles)
// ============================================================================

template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_patterns_8x8_kernel(
    int M, int K,
    const float* __restrict__ A,    // [M×K] row-major
    uint8_t* __restrict__ patterns  // [numTileRows × numKBlocks]
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numTileRows = M / TILE_M;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    // Each thread block processes one tile-row (TILE_M=8 rows)
    const int tileRow = blockIdx.x;
    if (tileRow >= numTileRows) return;

    const int globalMBase = tileRow * TILE_M;

    // Each warp processes multiple K-blocks
    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        // TILE_M×BK = 8×8 = 64 elements per tile
        // 32 threads per warp → 2 elements per thread
        constexpr int ELEMENTS_PER_THREAD = (TILE_M * BK) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId * ELEMENTS_PER_THREAD + i;
            const int mRow = flatIdx / BK;    // Which M-row within tile (0-7)
            const int kCol = flatIdx % BK;    // Which K-col within block (0-7)

            const int globalM = globalMBase + mRow;
            const int globalK = globalKBase + kCol;

            if (globalM < M && globalK < K) {
                float val = A[globalM * K + globalK];
                if (val != 0.0f) {
                    threadPattern |= (1 << kCol);
                }
            }
        }

        // Warp-level OR reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
        }

        if (laneId == 0) {
            patterns[tileRow * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// B-Pattern Preprocessing Kernel
// ============================================================================

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_ab_b_patterns_kernel(
    int K, int N,
    const float* __restrict__ B,    // [K×N] row-major
    uint8_t* __restrict__ patterns  // [numNBlocks × numKBlocks]
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numNBlocks = N / WN;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int nBlock = blockIdx.x;
    if (nBlock >= numNBlocks) return;

    const int globalNBase = nBlock * WN;

    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        constexpr int ELEMENTS_PER_THREAD = (WN * BK) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId * ELEMENTS_PER_THREAD + i;
            const int kRow = flatIdx / WN;
            const int nCol = flatIdx % WN;

            const int globalK = globalKBase + kRow;
            const int globalN = globalNBase + nCol;

            if (globalK < K && globalN < N) {
                float val = B[globalK * N + globalN];
                if (val != 0.0f) {
                    threadPattern |= (1 << kRow);
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
        }

        if (laneId == 0) {
            patterns[nBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// Pattern Analysis
// ============================================================================

__global__ void analyze_ab_patterns_kernel(
    const uint8_t* patterns, int totalPatterns,
    int* totalBitsSet, int* fullyZero
) {
    __shared__ int s_bits[256];
    __shared__ int s_zero[256];

    const int tid = threadIdx.x;
    s_bits[tid] = 0;
    s_zero[tid] = 0;

    for (int i = tid; i < totalPatterns; i += blockDim.x) {
        uint8_t p = patterns[i];
        s_bits[tid] += __popc(p);
        s_zero[tid] += (p == 0) ? 1 : 0;
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_bits[tid] += s_bits[tid + stride];
            s_zero[tid] += s_zero[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(totalBitsSet, s_bits[0]);
        atomicAdd(fullyZero, s_zero[0]);
    }
}

// ============================================================================
// Main Preprocessing Function
// ============================================================================

template <const int BK = 8, const int WM = 64, const int WN = 32>
ABPatternMetadata preprocess_ab_patterns(
    const float* d_A, const float* d_B,
    int M, int N, int K
) {
    ABPatternMetadata meta;
    meta.numMBlocks = M / WM;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;

    int a_total = meta.numMBlocks * meta.numKBlocks;
    int b_total = meta.numNBlocks * meta.numKBlocks;

    cudaMalloc(&meta.d_a_patterns, a_total * sizeof(uint8_t));
    cudaMalloc(&meta.d_b_patterns, b_total * sizeof(uint8_t));
    cudaMemset(meta.d_a_patterns, 0, a_total * sizeof(uint8_t));
    cudaMemset(meta.d_b_patterns, 0, b_total * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Preprocess A
    preprocess_a_patterns_kernel<BK, WM, NUM_THREADS>
        <<<meta.numMBlocks, NUM_THREADS>>>(M, K, d_A, meta.d_a_patterns);

    // Preprocess B
    preprocess_ab_b_patterns_kernel<BK, WN, NUM_THREADS>
        <<<meta.numNBlocks, NUM_THREADS>>>(K, N, d_B, meta.d_b_patterns);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Analyze A patterns
    int *d_a_bits, *d_a_zero;
    cudaMalloc(&d_a_bits, sizeof(int));
    cudaMalloc(&d_a_zero, sizeof(int));
    cudaMemset(d_a_bits, 0, sizeof(int));
    cudaMemset(d_a_zero, 0, sizeof(int));
    analyze_ab_patterns_kernel<<<1, 256>>>(meta.d_a_patterns, a_total, d_a_bits, d_a_zero);

    int h_a_bits, h_a_zero;
    cudaMemcpy(&h_a_bits, d_a_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_a_zero, d_a_zero, sizeof(int), cudaMemcpyDeviceToHost);

    // Analyze B patterns
    int *d_b_bits, *d_b_zero;
    cudaMalloc(&d_b_bits, sizeof(int));
    cudaMalloc(&d_b_zero, sizeof(int));
    cudaMemset(d_b_bits, 0, sizeof(int));
    cudaMemset(d_b_zero, 0, sizeof(int));
    analyze_ab_patterns_kernel<<<1, 256>>>(meta.d_b_patterns, b_total, d_b_bits, d_b_zero);

    int h_b_bits, h_b_zero;
    cudaMemcpy(&h_b_bits, d_b_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b_zero, d_b_zero, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a_bits); cudaFree(d_a_zero);
    cudaFree(d_b_bits); cudaFree(d_b_zero);

    // Compute sparsity stats
    meta.a_sparsity = 100.0f * (1.0f - (float)h_a_bits / (a_total * BK));
    meta.b_sparsity = 100.0f * (1.0f - (float)h_b_bits / (b_total * BK));

    // Joint sparsity estimate (probabilistic)
    float a_density = (float)h_a_bits / (a_total * BK);
    float b_density = (float)h_b_bits / (b_total * BK);
    meta.joint_sparsity = 100.0f * (1.0f - a_density * b_density);

    printf("A+B Pattern Preprocessing:\n");
    printf("  A patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           a_total, a_total / 1024.0f, meta.a_sparsity);
    printf("  B patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           b_total, b_total / 1024.0f, meta.b_sparsity);
    printf("  Joint skip rate (estimated): %.1f%%\n", meta.joint_sparsity);
    printf("  Preprocessing time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}

// ============================================================================
// 8×32 Granularity Preprocessing (TILE_M=8, WN=32)
// ============================================================================

template <const int BK = 8, const int TILE_M = 8, const int WN = 32>
ABPatternMetadata preprocess_ab_patterns_8x32(
    const float* d_A, const float* d_B,
    int M, int N, int K
) {
    ABPatternMetadata meta;
    // A: patterns at 8-row granularity
    const int numTileRows = M / TILE_M;
    // B: patterns at WN-col granularity (same as before)
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;

    // Store numTileRows in numMBlocks for compatibility
    meta.numMBlocks = numTileRows;

    int a_total = numTileRows * meta.numKBlocks;
    int b_total = meta.numNBlocks * meta.numKBlocks;

    cudaMalloc(&meta.d_a_patterns, a_total * sizeof(uint8_t));
    cudaMalloc(&meta.d_b_patterns, b_total * sizeof(uint8_t));
    cudaMemset(meta.d_a_patterns, 0, a_total * sizeof(uint8_t));
    cudaMemset(meta.d_b_patterns, 0, b_total * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Preprocess A at 8×8 granularity
    preprocess_a_patterns_8x8_kernel<BK, TILE_M, NUM_THREADS>
        <<<numTileRows, NUM_THREADS>>>(M, K, d_A, meta.d_a_patterns);

    // Preprocess B at 8×WN granularity (same as before)
    preprocess_ab_b_patterns_kernel<BK, WN, NUM_THREADS>
        <<<meta.numNBlocks, NUM_THREADS>>>(K, N, d_B, meta.d_b_patterns);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Analyze A patterns
    int *d_a_bits, *d_a_zero;
    cudaMalloc(&d_a_bits, sizeof(int));
    cudaMalloc(&d_a_zero, sizeof(int));
    cudaMemset(d_a_bits, 0, sizeof(int));
    cudaMemset(d_a_zero, 0, sizeof(int));
    analyze_ab_patterns_kernel<<<1, 256>>>(meta.d_a_patterns, a_total, d_a_bits, d_a_zero);

    int h_a_bits, h_a_zero;
    cudaMemcpy(&h_a_bits, d_a_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_a_zero, d_a_zero, sizeof(int), cudaMemcpyDeviceToHost);

    // Analyze B patterns
    int *d_b_bits, *d_b_zero;
    cudaMalloc(&d_b_bits, sizeof(int));
    cudaMalloc(&d_b_zero, sizeof(int));
    cudaMemset(d_b_bits, 0, sizeof(int));
    cudaMemset(d_b_zero, 0, sizeof(int));
    analyze_ab_patterns_kernel<<<1, 256>>>(meta.d_b_patterns, b_total, d_b_bits, d_b_zero);

    int h_b_bits, h_b_zero;
    cudaMemcpy(&h_b_bits, d_b_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b_zero, d_b_zero, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a_bits); cudaFree(d_a_zero);
    cudaFree(d_b_bits); cudaFree(d_b_zero);

    // Compute sparsity stats
    meta.a_sparsity = 100.0f * (1.0f - (float)h_a_bits / (a_total * BK));
    meta.b_sparsity = 100.0f * (1.0f - (float)h_b_bits / (b_total * BK));

    // Joint sparsity estimate (probabilistic)
    float a_density = (float)h_a_bits / (a_total * BK);
    float b_density = (float)h_b_bits / (b_total * BK);
    meta.joint_sparsity = 100.0f * (1.0f - a_density * b_density);

    printf("A+B Pattern Preprocessing (8×32 granularity):\n");
    printf("  A patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           a_total, a_total / 1024.0f, meta.a_sparsity);
    printf("  B patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           b_total, b_total / 1024.0f, meta.b_sparsity);
    printf("  Joint skip rate (estimated): %.1f%%\n", meta.joint_sparsity);
    printf("  Preprocessing time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}

// ============================================================================
// K29: A+B Pattern Preprocessing (32×8 A-tiles, 8×32 B-tiles)
// ============================================================================

template <const int BK = 8, const int TILE_M = 32, const int WN = 32>
ABPatternMetadata preprocess_ab_patterns_32x32(
    const float* d_A, const float* d_B,
    int M, int N, int K
) {
    constexpr int NUM_THREADS = 256;
    const int numTileRows = M / TILE_M;  // M/32 patterns for A
    const int numWarpCols = N / WN;      // N/32 patterns for B
    const int numKBlocks = K / BK;

    ABPatternMetadata meta;
    meta.numMBlocks = numTileRows;
    meta.numNBlocks = numWarpCols;
    meta.numKBlocks = numKBlocks;

    cudaMalloc(&meta.d_a_patterns, numTileRows * numKBlocks * sizeof(uint8_t));
    cudaMalloc(&meta.d_b_patterns, numWarpCols * numKBlocks * sizeof(uint8_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Preprocess A at 32×8 granularity (reuse existing kernel with WM=32)
    preprocess_a_patterns_kernel<BK, TILE_M, NUM_THREADS>
        <<<numTileRows, NUM_THREADS>>>(M, K, d_A, meta.d_a_patterns);

    // Preprocess B at 8×WN granularity (same as K24/K28)
    preprocess_ab_b_patterns_kernel<BK, WN, NUM_THREADS>
        <<<numWarpCols, NUM_THREADS>>>(K, N, d_B, meta.d_b_patterns);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate sparsity rates
    std::vector<uint8_t> h_a_patterns(numTileRows * numKBlocks);
    std::vector<uint8_t> h_b_patterns(numWarpCols * numKBlocks);
    cudaMemcpy(h_a_patterns.data(), meta.d_a_patterns, numTileRows * numKBlocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_patterns.data(), meta.d_b_patterns, numWarpCols * numKBlocks, cudaMemcpyDeviceToHost);

    int a_total_bits = numTileRows * numKBlocks * 8;
    int b_total_bits = numWarpCols * numKBlocks * 8;
    int a_zero_bits = 0, b_zero_bits = 0;

    for (auto pat : h_a_patterns) {
        a_zero_bits += (8 - __builtin_popcount(pat));
    }
    for (auto pat : h_b_patterns) {
        b_zero_bits += (8 - __builtin_popcount(pat));
    }

    meta.a_sparsity = (100.0f * a_zero_bits) / a_total_bits;
    meta.b_sparsity = (100.0f * b_zero_bits) / b_total_bits;
    meta.joint_sparsity = meta.a_sparsity + meta.b_sparsity - (meta.a_sparsity * meta.b_sparsity / 100.0f);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    int a_total = numTileRows * numKBlocks;
    int b_total = numWarpCols * numKBlocks;

    printf("A+B Pattern Preprocessing (32×32 granularity):\n");
    printf("  A patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           a_total, a_total / 1024.0f, meta.a_sparsity);
    printf("  B patterns: %d (%.1f KB), skip rate: %.1f%%\n",
           b_total, b_total / 1024.0f, meta.b_sparsity);
    printf("  Joint skip rate (estimated): %.1f%%\n", meta.joint_sparsity);
    printf("  Preprocessing time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
