#pragma once

/*
 * ============================================================================
 * Kernel 20: Joint A+B Sparsity with PRECOMPUTED Intersections
 * ============================================================================
 *
 * Strategy:
 *   1. Preprocessing computes joint patterns: A ∩ B (bitwise AND)
 *   2. Main kernel does SINGLE pattern lookup (not two)
 *   3. No runtime intersection - just look up precomputed joint offsets
 *
 * Benefits over runtime intersection:
 *   - 64× fewer lookups (1 vs 8×8 comparisons)
 *   - Zero intersection overhead
 *   - Simpler control flow
 *   - Better register allocation
 *
 * Performance:
 *   - Should match K17 speed at A-only (6.5 ms)
 *   - Should achieve 2× speedup at joint 50% sparsity (~3.2 ms)
 *
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/joint_preprocessor.cu"
#include <cuda_runtime.h>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

// Reuse the generic compute function from btranspose
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE>
__device__ __forceinline__ void compute_sparse_block_btranspose(
    const uint8_t* offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* BTs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = offsets[sparse_idx];

        const uint mBase = warpRow * WM + threadRowInWarp * TM;
        
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint tm = 0; tm < TM; ++tm) {
                const uint mOffset = mBase + wSubRowIdx * WSUBM + tm;
                const uint baseAddr = dotIdx * BM + mOffset;
                regM[wSubRowIdx * TM + tm] = As[baseAddr];
            }
        }

        const float* BTs_row = &BTs[dotIdx * BN];
        const uint nBase = warpCol * WN + threadColInWarp * TN;
        
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            #pragma unroll
            for (uint tn = 0; tn < TN; ++tn) {
                const uint nOffset = nBase + wSubColIdx * WSUBN + tn;
                regN[wSubColIdx * TN + tn] = BTs_row[nOffset];
            }
        }

        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (uint tm = 0; tm < TM; ++tm) {
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        const uint resIdx = (wSubRowIdx * TM + tm) * (WNITER * TN) + 
                                          wSubColIdx * TN + tn;
                        
                        threadResults[resIdx] += regM[wSubRowIdx * TM + tm] * 
                                                regN[wSubColIdx * TN + tn];
                    }
                }
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_joint_precomputed(int M, int N, int K,
                           float *A, float *B, float *C,
                           const uint8_t* __restrict__ jointPatterns,
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

    __shared__ float As[(BK + 1) * BM];
    __shared__ float BTs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    const uint globalWarpRow = cRow * (BM / WM) + warpRow;
    const uint globalColBlock = cCol * (BN / WN) + warpCol;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // ====================================================================
        // SINGLE LOOKUP: Precomputed joint pattern (A ∩ B)
        // K-MAJOR LAYOUT: All warps in same K-iteration access consecutive memory!
        // ====================================================================
        // OLD (M-major): (globalWarpRow * numNBlocks + globalColBlock) * numKBlocks + kBlock
        // NEW (K-major): kBlock * numMBlocks * numNBlocks + globalWarpRow * numNBlocks + globalColBlock
        const uint numMBlocks = M / WM;  // Total M-warps in matrix
        const uint patternIdx = kBlock * numMBlocks * numNBlocks + globalWarpRow * numNBlocks + globalColBlock;
        const uint8_t joint_pattern = jointPatterns[patternIdx];
        const uint8_t count = PATTERN_LUT_BK8[joint_pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[joint_pattern].offsets;

        // Early exit: Skip if joint pattern is fully sparse
        if (count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load shared memory only for productive blocks
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // ====================================================================
        // COMPUTE: Use precomputed joint offsets (already intersected!)
        // ====================================================================
        switch (count) {
            case 1: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 2: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 3: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 4: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 5: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 6: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 7: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
            case 8: compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results back to C
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    C_interim[(threadRowInWarp * TM + resIdxM) * N +
                              threadColInWarp * TN + resIdxN] = threadResults[i];
                }
            }
        }
    }
}


