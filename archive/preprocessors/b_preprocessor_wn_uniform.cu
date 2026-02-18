#pragma once

/*
 * ============================================================================
 * B-Matrix Preprocessor: WN-Granularity Column Patterns
 * ============================================================================
 *
 * Goal: Enable warp-uniform K-iteration skipping for B-sparsity
 *
 * Pattern Encoding:
 *   - B is [K×N] in row-major
 *   - Divide into [K/BK] × [N/WN] logical blocks
 *   - Each block: BK=8 rows × WN=32 columns
 *   - Pattern byte: bit k = 1 if ANY of WN columns has non-zero at row k
 *
 * Why WN=32 granularity?
 *   - Each warp processes exactly WN=32 columns
 *   - All 32 threads in a warp access the SAME K-row of B
 *   - Pattern lookup is warp-uniform → zero divergence
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ============================================================================
// Metadata Structure
// ============================================================================

struct BPatternMetadata {
    uint8_t* d_patterns;
    int numNBlocks;
    int numKBlocks;
    int totalPatterns;
    float sparsityPercent;
};

inline void free_b_pattern_metadata(BPatternMetadata& meta) {
    if (meta.d_patterns) {
        cudaFree(meta.d_patterns);
        meta.d_patterns = nullptr;
    }
}

// ============================================================================
// Preprocessing Kernel
// ============================================================================

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_patterns_kernel_wn(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ patterns
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

        // 16 iff WN == 64, 8 if WN == 32
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
            const int outIdx = nBlock * numKBlocks + kBlock;
            patterns[outIdx] = threadPattern;
        }
    }
}

// ============================================================================
// Sparsity Analysis Kernel
// ============================================================================

__global__ void analyze_patterns_kernel(
    const uint8_t* patterns,
    int totalPatterns,
    int* totalBitsSet,
    int* fullyZeroPatterns,
    int* fullyDensePatterns
) {
    __shared__ int s_bits[256];
    __shared__ int s_zero[256];
    __shared__ int s_dense[256];

    const int tid = threadIdx.x;
    s_bits[tid] = 0;
    s_zero[tid] = 0;
    s_dense[tid] = 0;

    for (int i = tid; i < totalPatterns; i += blockDim.x) {
        uint8_t p = patterns[i];
        s_bits[tid] += __popc(p);
        s_zero[tid] += (p == 0x00) ? 1 : 0;
        s_dense[tid] += (p == 0xFF) ? 1 : 0;
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_bits[tid] += s_bits[tid + stride];
            s_zero[tid] += s_zero[tid + stride];
            s_dense[tid] += s_dense[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(totalBitsSet, s_bits[0]);
        atomicAdd(fullyZeroPatterns, s_zero[0]);
        atomicAdd(fullyDensePatterns, s_dense[0]);
    }
}

// ============================================================================
// Main Preprocessing Function
// ============================================================================

template <const int BK = 8, const int WN = 32>
BPatternMetadata preprocess_b_patterns(const float* d_B, int K, int N) {
    BPatternMetadata meta;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;
    meta.totalPatterns = meta.numNBlocks * meta.numKBlocks;

    cudaMalloc(&meta.d_patterns, meta.totalPatterns * sizeof(uint8_t));
    cudaMemset(meta.d_patterns, 0, meta.totalPatterns * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;
    dim3 block(NUM_THREADS);
    dim3 grid(meta.numNBlocks);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    preprocess_b_patterns_kernel_wn<BK, WN, NUM_THREADS><<<grid, block>>>(
        K, N, d_B, meta.d_patterns
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    int* d_bits, *d_zero, *d_dense;
    cudaMalloc(&d_bits, sizeof(int));
    cudaMalloc(&d_zero, sizeof(int));
    cudaMalloc(&d_dense, sizeof(int));
    cudaMemset(d_bits, 0, sizeof(int));
    cudaMemset(d_zero, 0, sizeof(int));
    cudaMemset(d_dense, 0, sizeof(int));

    analyze_patterns_kernel<<<1, 256>>>(
        meta.d_patterns, meta.totalPatterns, d_bits, d_zero, d_dense
    );

    int h_bits, h_zero, h_dense;
    cudaMemcpy(&h_bits, d_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_zero, d_zero, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dense, d_dense, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_bits);
    cudaFree(d_zero);
    cudaFree(d_dense);

    int maxBits = meta.totalPatterns * BK;
    meta.sparsityPercent = 100.0f * (1.0f - (float)h_bits / maxBits);

    printf("B-Pattern Preprocessing:\n");
    printf("  Matrix: %d x %d, Blocks: %d x %d\n", K, N, meta.numKBlocks, meta.numNBlocks);
    printf("  Total patterns: %d (%.1f KB)\n", meta.totalPatterns, meta.totalPatterns / 1024.0f);
    printf("  Fully zero tiles: %d (%.1f%% tile skip rate)\n",
           h_zero, 100.0f * h_zero / meta.totalPatterns);
    printf("  Effective K-skip rate: %.1f%%\n", meta.sparsityPercent);
    printf("  Preprocessing time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return meta;
}
