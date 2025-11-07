#pragma once

/*
 * ============================================================================
 * Kernel 19: ESMM B-Transpose (Warp-Uniform B-Sparsity)
 * ============================================================================
 *
 * Strategy:
 *   Transpose B to B^T and flip warp tile orientation to enable warp-uniform
 *   B-sparsity checks. All threads in a warp compute outputs for the SAME
 *   columns, so they all need the same B^T row → same sparsity pattern!
 *
 * Architecture Changes vs Kernel 17:
 *   - Warp tile: 64 rows × 32 columns (was 32×64)
 *   - Thread tile: TM=8, TN=1 (was TM=1, TN=8)
 *   - B is transposed to B^T (N×K)
 *   - B^T patterns: 8×32 blocks (BK × WN), row-wise encoding
 *
 * Why This Works:
 *   - All 32 threads compute same 32 output columns
 *   - All threads need same B^T rows → same patterns
 *   - Pattern checks are warp-uniform → zero divergence!
 *
 * Performance Characteristics:
 *   - Memory: 1 byte per 8×32 block (~64 KB for 4096×4096)
 *   - Divergence: Zero (warp-uniform B-pattern checks)
 *   - C writes: Improved coalescing (consecutive threads → consecutive columns)
 *   - Trade-off: B^T loading less coalesced, but transpose cost amortized
 *
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/b_transpose_preprocessor.cu"
#include <cuda_runtime.h>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

// ============================================================================
// Helper templates for compile-time unrolled computation with B^T sparsity
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE,
          const int AS_PAD, const int BTS_PAD>
__device__ void compute_sparse_block_btranspose(
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
            #pragma unroll
            for (uint tm = 0; tm < TM; ++tm) {
                const uint m = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + tm;
                regM[wSubRowIdx * TM + tm] = As[dotIdx + m * (BK + AS_PAD)];
            }
        }

        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint localCol = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
            regN[wSubColIdx * TN] = BTs[dotIdx + localCol * (BK + BTS_PAD)];
        }

        // Compute outer product: each thread computes TM × TN = 8×1 outputs
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (uint tm = 0; tm < TM; ++tm) {
                    const int resIdx = (wSubRowIdx * TM + tm) * (WNITER * TN) +
                                      wSubColIdx * TN;
                    threadResults[resIdx] += regM[wSubRowIdx * TM + tm] *
                                             regN[wSubColIdx * TN];
                }
            }
        }
    }
}

// ============================================================================
// Main kernel: B^T sparsity with warp-uniform pattern checks
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_btranspose(int M, int N, int K,
                    float *A, float *B, float *C,
                    const uint8_t* __restrict__ blockPatterns,
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

    // Padding to avoid bank conflicts (32 banks, 4-byte words)
    // For BM=128, BK=8: each row is 8 floats, add 1 float padding
    constexpr uint AS_PAD = 1;
    constexpr uint BTS_PAD = 1;

    __shared__ float As[(BK + AS_PAD) * BM];  // Column-major: [BK × BM] with padding
    __shared__ float BTs[(BK + BTS_PAD) * BN];  // Column-major: [BK × BN] with padding

    A += cRow * BM * K;
    B += cCol * BN;  // B is K×N, offset by columns
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    // For B (K×N): distribute threads for coalesced loads
    // Want consecutive threads to load consecutive memory locations
    // B[k*N + n] - consecutive in N dimension
    const uint innerRowB = threadIdx.x / (BN / 4);  // Which K-row (thread 0-31 → k=0, 32-63 → k=1, etc.)
    const uint innerColB = threadIdx.x % (BN / 4);  // Which N-column group (0-31)
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;  // How many K-rows per iteration

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Calculate global column block for B^T pattern lookup
    const uint globalColBlock = cCol * (BN / WN) + warpCol;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;


        const uint blockId = globalColBlock * numKBlocks + kBlock;
        const uint8_t pattern = blockPatterns[blockId];

        const uint8_t count = PATTERN_LUT_BK8[pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;

        if (count == 0) {
            A += BK;
            B += BK * N;  // B is K×N, advance by BK rows
            continue;
        }

        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            // Column-major with padding: As[k * (BK + AS_PAD) + m]
            As[(innerColA * 4 + 0) + (innerRowA + offset) * (BK + AS_PAD)] = tmp.x;
            As[(innerColA * 4 + 1) + (innerRowA + offset) * (BK + AS_PAD)] = tmp.y;
            As[(innerColA * 4 + 2) + (innerRowA + offset) * (BK + AS_PAD)] = tmp.z;
            As[(innerColA * 4 + 3) + (innerRowA + offset) * (BK + AS_PAD)] = tmp.w;
        }

        // Load B (K×N) with coalesced access - one iteration for BK=8, rowStrideB=8
        // Unrolled for performance
        const float4 tmp = reinterpret_cast<const float4 *>(
            &B[innerRowB * N + innerColB * 4])[0];

        // Write transposed to shared: BTs[k + n * stride]
        // This changes from B's row-major [K][N] to shared column-major [BK][BN]
        BTs[innerRowB + (innerColB * 4 + 0) * (BK + BTS_PAD)] = tmp.x;
        BTs[innerRowB + (innerColB * 4 + 1) * (BK + BTS_PAD)] = tmp.y;
        BTs[innerRowB + (innerColB * 4 + 2) * (BK + BTS_PAD)] = tmp.z;
        BTs[innerRowB + (innerColB * 4 + 3) * (BK + BTS_PAD)] = tmp.w;
        __syncthreads();

        // *** MAGIC: Switch on count for compile-time unrolling ***
        // All threads execute same case → warp-uniform!
        switch (count) {
            case 1:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 1, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 2:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 2, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 3:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 3, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 4:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 4, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 5:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 5, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 6:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 6, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 7:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 7, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 8:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 8, AS_PAD, BTS_PAD>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
        }

        A += BK;
        B += BK * N;  // B is K×N, advance by BK rows
        __syncthreads();
    }

    // Write results back to C
    // With TN=1, TM=8: each thread writes 8 values per WMITER iteration
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;

                    // Better coalescing: consecutive threads write consecutive columns!
                    C_interim[(threadRowInWarp * TM + resIdxM) * N +
                              threadColInWarp * TN + resIdxN] = threadResults[i];
                }
            }
        }
    }
}
