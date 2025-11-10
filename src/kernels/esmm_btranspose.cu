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
          const int WNITER, const int TM, const int TN, const int SIZE>
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

    // Manually unroll everything like K17 for maximum ILP
    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = offsets[sparse_idx];

        // Load A values - manually unrolled for TM=8
        // Pre-compute base address
        const uint mBase = warpRow * WM + threadRowInWarp * TM;

        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const uint mOffset = mBase + wSubRowIdx * WSUBM;
            // Column-major: As[k * BM + m] - consecutive loads!
            const uint baseAddr = dotIdx * BM + mOffset;
            regM[wSubRowIdx * TM + 0] = As[baseAddr + 0];
            regM[wSubRowIdx * TM + 1] = As[baseAddr + 1];
            regM[wSubRowIdx * TM + 2] = As[baseAddr + 2];
            regM[wSubRowIdx * TM + 3] = As[baseAddr + 3];
            regM[wSubRowIdx * TM + 4] = As[baseAddr + 4];
            regM[wSubRowIdx * TM + 5] = As[baseAddr + 5];
            regM[wSubRowIdx * TM + 6] = As[baseAddr + 6];
            regM[wSubRowIdx * TM + 7] = As[baseAddr + 7];
        }

        // Load B value - single load for TN=1
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint localCol = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
            // Column-major: BTs[k * BN + n]
            regN[wSubColIdx * TN] = BTs[dotIdx * BN + localCol];
        }

        // Compute outer product: 8 A values * 1 B value - manually unrolled
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                const float b_val = regN[wSubColIdx * TN];
                const int resBase = wSubRowIdx * TM * (WNITER * TN) + wSubColIdx * TN;

                // 8 independent multiply-accumulates for ILP
                threadResults[resBase + 0 * (WNITER * TN)] += regM[wSubRowIdx * TM + 0] * b_val;
                threadResults[resBase + 1 * (WNITER * TN)] += regM[wSubRowIdx * TM + 1] * b_val;
                threadResults[resBase + 2 * (WNITER * TN)] += regM[wSubRowIdx * TM + 2] * b_val;
                threadResults[resBase + 3 * (WNITER * TN)] += regM[wSubRowIdx * TM + 3] * b_val;
                threadResults[resBase + 4 * (WNITER * TN)] += regM[wSubRowIdx * TM + 4] * b_val;
                threadResults[resBase + 5 * (WNITER * TN)] += regM[wSubRowIdx * TM + 5] * b_val;
                threadResults[resBase + 6 * (WNITER * TN)] += regM[wSubRowIdx * TM + 6] * b_val;
                threadResults[resBase + 7 * (WNITER * TN)] += regM[wSubRowIdx * TM + 7] * b_val;
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

    // Use column-major layout with padding to avoid bank conflicts
    __shared__ float As[(BK + 1) * BM];  // Column-major: [BK × BM] with padding
    __shared__ float BTs[(BK + 1) * BN];  // Column-major: [BK × BN] with padding

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
            // Column-major with padding: As[k * BM + m]
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // Load B (K×N) and transpose to column-major BTs
        for (int32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];

            // Column-major with padding: BTs[k * BN + n]
            // We're transposing: B[k][n] → BTs[k][n]
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // *** MAGIC: Switch on count for compile-time unrolling ***
        // All threads execute same case → warp-uniform!
        switch (count) {
            case 1:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 2:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 3:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 4:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 5:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 6:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 7:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, BTs, threadResults);
                break;
            case 8:
                compute_sparse_block_btranspose<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
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
