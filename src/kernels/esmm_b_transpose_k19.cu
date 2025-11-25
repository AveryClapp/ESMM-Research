#pragma once

/*
 * ============================================================================
 * Kernel 19: ESMM B-Transpose Sparse (K19)
 * ============================================================================
 *
 * Strategy:
 *   Transpose B so its row-sparsity can be exploited by skipping K-iterations,
 *   matching A-sparse (K16) architecture and performance.
 *
 * Key Insight:
 *   After B[K×N] → B_T[N×K]:
 *   - Pattern per WN rows: "which K-positions are non-zero"
 *   - Skip entire K-iterations (dotIdx) when pattern bit is 0
 *   - Identical architecture to K16, just different matrix
 *
 * Performance Target: ~175 µs (same as K16)
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/b_transpose_preprocessor_simple.cu"
#include <cuda_runtime.h>

// Compute sparse block (template instantiated for SIZE=1..8)
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE>
__device__ void compute_sparse_block_k19(
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

        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        // Load B_T from shared memory
        // B_T is [N×K] in row-major, so BTs is loaded as [BN×BK]
        // We want BTs[n, k] which is BTs[n * BK + k]
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint nBase = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
            regN[wSubColIdx * TN + 0] = BTs[(nBase + 0) * BK + dotIdx];
            regN[wSubColIdx * TN + 1] = BTs[(nBase + 1) * BK + dotIdx];
            regN[wSubColIdx * TN + 2] = BTs[(nBase + 2) * BK + dotIdx];
            regN[wSubColIdx * TN + 3] = BTs[(nBase + 3) * BK + dotIdx];
            regN[wSubColIdx * TN + 4] = BTs[(nBase + 4) * BK + dotIdx];
            regN[wSubColIdx * TN + 5] = BTs[(nBase + 5) * BK + dotIdx];
            regN[wSubColIdx * TN + 6] = BTs[(nBase + 6) * BK + dotIdx];
            regN[wSubColIdx * TN + 7] = BTs[(nBase + 7) * BK + dotIdx];
        }

        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
                    regM[wSubRowIdx], regN, threadResults);
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_b_transpose_k19(int M, int N, int K, float *A, float *B_T, float *C,
                         const uint8_t* __restrict__ rowPatterns,
                         const int numKBlocks) {

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

    __shared__ float As[BK * (BM + 1)];
    __shared__ float BTs[BN * BK];

    A += cRow * BM * K;
    B_T += cCol * BN * K;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    const uint globalWarpCol = cCol * (BN / WN) + warpCol;
    const uint innerRowBT = threadIdx.x / (BK / 4);
    const uint innerColBT = threadIdx.x % (BK / 4);

    constexpr uint rowStrideBT = NUM_THREADS / (BK / 4);

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        const uint blockId = globalWarpCol * numKBlocks + kBlock;
        const uint8_t pattern = rowPatterns[blockId];
        const uint8_t count = PATTERN_LUT_BK8[pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;

        if (count == 0) {
            A += BK;
            B_T += BK;
            continue;
        }

        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }


        for (int32_t offset = 0; offset + rowStrideBT <= BN; offset += rowStrideBT) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                    &B_T[(innerRowBT + offset) * K + bkIdx + innerColBT * 4])[0];
            BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 0] = tmp.x;
            BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 1] = tmp.y;
            BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 2] = tmp.z;
            BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 3] = tmp.w;
        }
        __syncthreads();

        switch (count) {
            case 1:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 2:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 3:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 4:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 5:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 6:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 7:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 8:
                compute_sparse_block_k19<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
        }

        A += BK;
        B_T += BK;
        __syncthreads();
    }

    // Write results back to C
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
