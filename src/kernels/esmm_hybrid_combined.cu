#pragma once

/*
 * ============================================================================
 * Kernel 18: ESMM Combined A+B Sparsity (Sparse Direct Loading)
 * ============================================================================
 *
 * Strategy:
 *   Combines A-sparsity (K-dimension) and B-sparsity (N-dimension) with
 *   sparse direct loading - only loads B elements that are actually non-zero.
 *
 * Architecture:
 *   - A-Sparsity: 8×32 blocks (BK × WM), LUT-based offset lookup
 *   - B-Sparsity: 8×8 blocks (BK × TN), conditional load and multiply
 *   - Loading: Check pattern bits, only load non-zero elements (saves bandwidth)
 *   - Computation: Inline conditional FMAs (no dispatch overhead)
 *
 * Performance Characteristics:
 *   - Memory: Bandwidth proportional to actual B-sparsity (not full 100%)
 *   - Divergence: Predicated loads/stores may reduce warp efficiency
 *   - Theoretical speedup: (1 - A_sparsity) × compute + bandwidth savings from B
 *
 * Optimizations:
 *   - No function call overhead (dispatch_b_pattern removed)
 *   - Only loads elements that will be used
 *   - Compiler may use predicated instructions for conditional loads
 *
 * Best for:
 *   Sparse models with <50% B-density where bandwidth savings > overhead
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../../include/unrolled_inners.cuh"
#include "../../include/b_pattern_dispatch.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_preprocessor_hybrid.cu"
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

    uint nBlocks[WNITER];
    uint8_t B_patterns_cache[WNITER];
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        const uint globalColBase = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
        nBlocks[wSubColIdx] = globalColBase >> 3;
        const uint B_patternIdx = kBlock * numNBlocks + nBlocks[wSubColIdx];
        B_patterns_cache[wSubColIdx] = B_patterns[B_patternIdx];
    }

    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = A_offsets[sparse_idx];

        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint8_t B_pattern = B_patterns_cache[wSubColIdx];
            const uint baseAddr = (dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;

            // Vectorized loads with ternary predication (compiler converts to predicated instructions)
            float4 tmp1 = reinterpret_cast<const float4*>(&Bs[baseAddr + 0])[0];
            float4 tmp2 = reinterpret_cast<const float4*>(&Bs[baseAddr + 4])[0];

            regN[wSubColIdx * TN + 0] = (B_pattern & 0x01) ? tmp1.x : 0.0f;
            regN[wSubColIdx * TN + 1] = (B_pattern & 0x02) ? tmp1.y : 0.0f;
            regN[wSubColIdx * TN + 2] = (B_pattern & 0x04) ? tmp1.z : 0.0f;
            regN[wSubColIdx * TN + 3] = (B_pattern & 0x08) ? tmp1.w : 0.0f;
            regN[wSubColIdx * TN + 4] = (B_pattern & 0x10) ? tmp2.x : 0.0f;
            regN[wSubColIdx * TN + 5] = (B_pattern & 0x20) ? tmp2.y : 0.0f;
            regN[wSubColIdx * TN + 6] = (B_pattern & 0x40) ? tmp2.z : 0.0f;
            regN[wSubColIdx * TN + 7] = (B_pattern & 0x80) ? tmp2.w : 0.0f;
        }

        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const float regM_val = regM[wSubRowIdx];
            const int threadResRowBase = wSubRowIdx * (WNITER * TN);

            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                const uint8_t B_pattern = B_patterns_cache[wSubColIdx];
                const int regNBase = wSubColIdx * TN;
                const int threadResBase = threadResRowBase + wSubColIdx * TN;

                // Use unconditional FMAs - zeros from regN make these no-ops
                threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
                threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
                threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
                threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
                threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
                threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
                threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
                threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
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

        const uint A_blockId = globalWarpRow * numKBlocks + kBlock;
        const uint8_t A_pattern = A_blockPatterns[A_blockId];

        const uint8_t count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[A_pattern].offsets;

        // Early exit if all zeros
        /*
        if (count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }
        */

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
