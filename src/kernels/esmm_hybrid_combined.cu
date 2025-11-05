#pragma once

/*
 * ============================================================================
 * Kernel 18: ESMM Combined A+B Sparsity
 * ============================================================================
 *
 * Strategy:
 *   Combines A-sparsity (K-dimension) and B-sparsity (N-dimension) for maximum
 *   performance. Each warp processes only the non-zero K elements and only
 *   computes non-zero N columns.
 *
 * Architecture:
 *   - A-Sparsity: 8×32 blocks (BK × WM), LUT-based offset lookup
 *   - B-Sparsity: 8×8 blocks (BK × TN), pattern-specialized multiply dispatch
 *   - Runtime: Load both patterns, outer loop on A count, inner dispatch on B pattern
 *   - Computation: Full compile-time unrolling for both dimensions
 *
 * Performance Characteristics:
 *   - Memory: A patterns (~64 KB) + B patterns (~256 KB) for 4096×4096
 *   - Divergence: Minimal (warp-uniform A pattern, thread-level B pattern)
 *   - Theoretical speedup: (1 - A_sparsity) × (1 - B_sparsity)
 *
 * Best for:
 *   Models with sparsity in both activations (A) and weights (B), such as:
 *   - Sparse transformers, MoE models, pruned networks
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../../include/unrolled_inners.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_preprocessor_hybrid.cu"
#include "b_pattern_dispatch.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Helper templates for compile-time unrolled computation (A+B sparsity)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE>
__device__ void compute_sparse_block_combined(
    const uint8_t* A_offsets,
    const uint8_t* B_patterns,
    const uint cCol, const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    const uint kBlock, const int numNBlocks,
    float* As, float* Bs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Pre-compute nBlocks for each wSubColIdx (moved outside sparse loop)
    uint nBlocks[WNITER];
    uint8_t B_patterns_cache[WNITER];
    bool all_dense = true;  // Check if all patterns are 0xFF (dense)
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        const uint globalColBase = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
        nBlocks[wSubColIdx] = globalColBase >> 3;  // Bitshift instead of /TN (TN=8)
        const uint B_patternIdx = kBlock * numNBlocks + nBlocks[wSubColIdx];
        B_patterns_cache[wSubColIdx] = B_patterns[B_patternIdx];
        if (B_patterns_cache[wSubColIdx] != 0xFF) all_dense = false;
    }

    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = A_offsets[sparse_idx];

        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
            regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
            regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
            regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
            regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
            regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
            regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
            regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol *
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
        }

        // Compute outer product with B-sparsity dispatch
        // Fast path: if all B blocks are dense, skip dispatch entirely
        if (all_dense) {
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
                        regM[wSubRowIdx], regN, threadResults);
                }
            }
        } else {
            // Slow path: dispatch based on B patterns
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const uint8_t B_pattern = B_patterns_cache[wSubColIdx];
                    dispatch_b_pattern(B_pattern, wSubRowIdx, wSubColIdx, WNITER,
                        regM[wSubRowIdx], regN, threadResults);
                }
            }
        }
    }
}

// ============================================================================
// Main kernel: Combined A+B sparsity
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_combined_blockwise(int M, int N, int K, float *A, float *B, float *C,
                            const uint8_t* __restrict__ A_blockPatterns,
                            const uint8_t* __restrict__ B_blockPatterns,
                            const int numKBlocks, const int numNBlocks) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Calculate global warp row for pattern lookup
    const uint globalWarpRow = cRow * (BM / WM) + warpRow;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // *** Load A pattern for this warp tile's 8×32 block ***
        const uint A_blockId = globalWarpRow * numKBlocks + kBlock;
        const uint8_t A_pattern = A_blockPatterns[A_blockId];

        // *** LUT-based offset lookup for A (eliminates runtime loop) ***
        const uint8_t count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[A_pattern].offsets;

        // Early exit if all zeros
        if (count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load A tile
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // Load B tile
        for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        // *** MAGIC: Switch on A count for compile-time unrolling, dispatch B patterns ***
        switch (count) {
            case 1:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 2:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 3:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 4:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 5:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 6:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 7:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 8:
                compute_sparse_block_combined<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results back to global memory
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp;
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
