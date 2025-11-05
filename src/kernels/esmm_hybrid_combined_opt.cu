#pragma once

/*
 * ============================================================================
 * Kernel 18 (Optimized): ESMM Combined A+B Sparsity with Conditional Loading
 * ============================================================================
 *
 * Key Optimization for <=50% B Sparsity:
 *   - Conditionally load regN elements based on B pattern bits
 *   - Saves memory bandwidth (not just compute) for sparse B matrices
 *   - Uses bit-checking to avoid loading unused elements
 *
 * Performance Characteristics:
 *   - At 50% B sparsity: ~50% reduction in Bs loads + 50% reduction in FMAs
 *   - At 25% B sparsity: ~75% reduction in Bs loads + 75% reduction in FMAs
 *   - Overhead: 8 bit checks per wSubColIdx (negligible vs memory saves)
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
// Optimized helper: Conditional loading based on B pattern
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE>
__device__ void compute_sparse_block_combined_opt(
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

    // Pre-compute nBlocks and cache B patterns
    uint nBlocks[WNITER];
    uint8_t B_patterns_cache[WNITER];
    #pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        const uint globalColBase = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
        nBlocks[wSubColIdx] = globalColBase >> 3;
        const uint B_patternIdx = kBlock * numNBlocks + nBlocks[wSubColIdx];
        B_patterns_cache[wSubColIdx] = B_patterns[B_patternIdx];
    }

    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = A_offsets[sparse_idx];

        // Load A values (same as before)
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        // *** OPTIMIZATION: Conditionally load B values based on pattern ***
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint8_t B_pattern = B_patterns_cache[wSubColIdx];
            const uint baseAddr = (dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
            const uint regNBase = wSubColIdx * TN;

            // Only load elements indicated by pattern bits
            // Compiler will unroll this and optimize away the constant checks
            if (B_pattern & 0x01) regN[regNBase + 0] = Bs[baseAddr + 0];
            if (B_pattern & 0x02) regN[regNBase + 1] = Bs[baseAddr + 1];
            if (B_pattern & 0x04) regN[regNBase + 2] = Bs[baseAddr + 2];
            if (B_pattern & 0x08) regN[regNBase + 3] = Bs[baseAddr + 3];
            if (B_pattern & 0x10) regN[regNBase + 4] = Bs[baseAddr + 4];
            if (B_pattern & 0x20) regN[regNBase + 5] = Bs[baseAddr + 5];
            if (B_pattern & 0x40) regN[regNBase + 6] = Bs[baseAddr + 6];
            if (B_pattern & 0x80) regN[regNBase + 7] = Bs[baseAddr + 7];
        }

        // Compute outer product with B-sparsity dispatch
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

// ============================================================================
// Main kernel: Combined A+B sparsity (Optimized)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_combined_blockwise_opt(int M, int N, int K, float *A, float *B, float *C,
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

    const uint globalWarpRow = cRow * (BM / WM) + warpRow;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Load A pattern and get LUT data
        const uint A_blockId = globalWarpRow * numKBlocks + kBlock;
        const uint8_t A_pattern = A_blockPatterns[A_blockId];
        const uint8_t count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[A_pattern].offsets;

        // Early exit if all zeros
        if (count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load tiles into shared memory
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        // Dispatch based on A count with optimized B loading
        switch (count) {
            case 1:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 2:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 3:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 4:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 5:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 6:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 7:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    offsets, B_blockPatterns, cCol, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, kBlock, numNBlocks,
                    As, Bs, threadResults);
                break;
            case 8:
                compute_sparse_block_combined_opt<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
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
