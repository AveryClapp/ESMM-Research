#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

struct ABPatternMetadata {
    uint8_t* d_a_patterns;  // [numMBlocks × numKBlocks]
    uint8_t* d_b_patterns;  // [numNBlocks × numKBlocks]
    bool owns_b_patterns;   // Whether this struct owns d_b_patterns (should cudaFree)
    int numMBlocks;         // M / TILE_M_A
    int numNBlocks;         // N / WN
    int numKBlocks;         // K / BK
    float a_sparsity;       // Effective A skip rate
    float b_sparsity;       // Effective B skip rate
    float joint_sparsity;   // Expected joint skip rate
};

inline void free_ab_pattern_metadata(ABPatternMetadata& meta) {
    if (meta.d_a_patterns) cudaFree(meta.d_a_patterns);
    if (meta.owns_b_patterns && meta.d_b_patterns) cudaFree(meta.d_b_patterns);
    meta.d_a_patterns = nullptr;
    meta.d_b_patterns = nullptr;
}

// ============================================================================
// A-Pattern Preprocessing Kernel (unified for any M-tile granularity)
// Used by K20 (TILE_M=64) and K21 (TILE_M=8) via preprocess_ab<>.
// ============================================================================

template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_patterns_kernel(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numTiles = M / TILE_M;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    const int globalMBase = tileIdx * TILE_M;

    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        constexpr int ELEMENTS_PER_THREAD = (TILE_M * BK) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId * ELEMENTS_PER_THREAD + i;
            const int mRow = flatIdx / BK;
            const int kCol = flatIdx % BK;

            const int globalM = globalMBase + mRow;
            const int globalK = globalKBase + kCol;

            if (globalM < M && globalK < K) {
                float val = A[globalM * K + globalK];
                if (val != 0.0f) {
                    threadPattern |= (1 << kCol);
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
        }

        if (laneId == 0) {
            patterns[tileIdx * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// B-Pattern Preprocessing Kernel
// Used by K20, K21, and K25 via preprocess_ab<> and preprocess_b_fused<>.
// ============================================================================

template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_patterns_kernel(
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
// Pattern Analysis (used to compute sparsity stats)
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
// K25 Fused Preprocessors (optimized ballot-sync / float4 variants)
// Moved here from esmm_ab_simple_fused.cu so all preprocessing is in one place.
// ============================================================================

// Coalesced A preprocessor: all 32 threads read same TILE_M rows across 32
// consecutive K-columns, then ballot_sync reduces to a per-BK-block pattern.
template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_fused(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
    constexpr int K_CHUNK = WARP_SIZE;
    constexpr int BK_BLOCKS_PER_CHUNK = K_CHUNK / BK;

    const int numTiles = M / TILE_M;
    const int numKBlocks = K / BK;
    const int numKChunks = K / K_CHUNK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    const int globalMBase = tileIdx * TILE_M;

    for (int kChunk = warpId; kChunk < numKChunks; kChunk += WARPS_PER_BLOCK) {
        const int globalKChunkBase = kChunk * K_CHUNK;

        uint32_t myBit = 0;

        #pragma unroll 4
        for (int mRow = 0; mRow < TILE_M; mRow++) {
            const int globalM = globalMBase + mRow;
            float val = A[globalM * K + globalKChunkBase + laneId];
            if (val != 0.0f) {
                myBit = 1;
            }
        }

        uint32_t ballot = __ballot_sync(0xFFFFFFFF, myBit);

        #pragma unroll
        for (int bkOffset = 0; bkOffset < BK_BLOCKS_PER_CHUNK; bkOffset++) {
            const int kBlock = kChunk * BK_BLOCKS_PER_CHUNK + bkOffset;
            uint8_t pattern = (ballot >> (bkOffset * BK)) & 0xFF;
            if (laneId == bkOffset) {
                patterns[tileIdx * numKBlocks + kBlock] = pattern;
            }
        }
    }
}

// Optimized B preprocessor using float4 loads.
// B is K×N row-major, so consecutive N-values are consecutive in memory.
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_fused(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
    constexpr int FLOAT4_PER_ROW = WN / 4;
    constexpr int TOTAL_FLOAT4 = BK * FLOAT4_PER_ROW;
    constexpr int FLOAT4_PER_THREAD = TOTAL_FLOAT4 / WARP_SIZE;

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

        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            const int float4Idx = laneId + i * WARP_SIZE;
            const int kRow = float4Idx / FLOAT4_PER_ROW;
            const int nBase = (float4Idx % FLOAT4_PER_ROW) * 4;

            const int globalK = globalKBase + kRow;
            const int globalN = globalNBase + nBase;

            float4 vals = *reinterpret_cast<const float4*>(&B[globalK * N + globalN]);
            if (vals.x != 0.0f || vals.y != 0.0f || vals.z != 0.0f || vals.w != 0.0f) {
                threadPattern |= (1 << kRow);
            }
        }

        threadPattern = __reduce_or_sync(0xFFFFFFFF, threadPattern);

        if (laneId == 0) {
            patterns[nBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// Unified Preprocessing Wrapper
// Replaces preprocess_ab_patterns<BK,WM,WN> (K20) and
// preprocess_ab_patterns_8x32<BK,TILE_M,WN> (K21).
// K20 calls: preprocess_ab<8, 64, 32>(d_A, d_B, M, N, K)
// K21 calls: preprocess_ab<8,  8, 32>(d_A, d_B, M, N, K)
// ============================================================================

template <const int BK = 8, const int TILE_M_A = 64, const int WN = 32>
ABPatternMetadata preprocess_ab(
    const float* d_A, const float* d_B,
    int M, int N, int K
) {
    ABPatternMetadata meta;
    meta.numMBlocks = M / TILE_M_A;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;
    meta.owns_b_patterns = true;

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

    preprocess_a_patterns_kernel<BK, TILE_M_A, NUM_THREADS>
        <<<meta.numMBlocks, NUM_THREADS>>>(M, K, d_A, meta.d_a_patterns);

    preprocess_b_patterns_kernel<BK, WN, NUM_THREADS>
        <<<meta.numNBlocks, NUM_THREADS>>>(K, N, d_B, meta.d_b_patterns);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Analyze patterns for sparsity stats
    int *d_a_bits, *d_a_zero;
    cudaMalloc(&d_a_bits, sizeof(int));
    cudaMalloc(&d_a_zero, sizeof(int));
    cudaMemset(d_a_bits, 0, sizeof(int));
    cudaMemset(d_a_zero, 0, sizeof(int));
    analyze_ab_patterns_kernel<<<1, 256>>>(meta.d_a_patterns, a_total, d_a_bits, d_a_zero);

    int h_a_bits, h_a_zero;
    cudaMemcpy(&h_a_bits, d_a_bits, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_a_zero, d_a_zero, sizeof(int), cudaMemcpyDeviceToHost);

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

    meta.a_sparsity = 100.0f * (1.0f - (float)h_a_bits / (a_total * BK));
    meta.b_sparsity = 100.0f * (1.0f - (float)h_b_bits / (b_total * BK));

    float a_density = (float)h_a_bits / (a_total * BK);
    float b_density = (float)h_b_bits / (b_total * BK);
    meta.joint_sparsity = 100.0f * (1.0f - a_density * b_density);

    printf("A+B Pattern Preprocessing (TILE_M=%d, WN=%d):\n", TILE_M_A, WN);
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
