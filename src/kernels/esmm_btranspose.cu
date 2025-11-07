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

    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = offsets[sparse_idx];

        // Load from shared memory A - each thread loads different rows
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint tm = 0; tm < TM; ++tm) {
                regM[wSubRowIdx * TM + tm] = As[(dotIdx * BM) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM + tm];
            }
        }

        // Load from shared memory B^T - all threads load same row (BROADCAST!)
        // B^T is stored as [N][K], so for each output column, we need one B^T row
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // Local column within the BN tile (not global!)
            const uint localCol = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;

            // B^T[localCol, k] - all threads load same value (broadcast from shared mem)
            regN[wSubColIdx * TN] = BTs[localCol * BK + dotIdx];
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
                    float *A, float *BT, float *C,
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

    __shared__ float As[BM * BK];
    __shared__ float BTs[BN * BK];  // B^T tile: BN rows × BK cols

    A += cRow * BM * K;
    BT += cCol * BN * K;  // B^T is N×K, so offset by rows
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    // For B^T (N×K), load similar to A
    const uint innerRowBT = threadIdx.x / (BK / 4);
    const uint innerColBT = threadIdx.x % (BK / 4);
    constexpr uint rowStrideBT = (NUM_THREADS * 4) / BK;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Calculate global column block for B^T pattern lookup
    const uint globalColBlock = cCol * (BN / WN) + warpCol;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Load B^T pattern for this column block
        // All threads in warp computing columns need same pattern → warp-uniform!
        const uint blockId = globalColBlock * numKBlocks + kBlock;
        const uint8_t pattern = blockPatterns[blockId];

        const uint8_t count = PATTERN_LUT_BK8[pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;

        // Warp-uniform skip: if entire B^T row is zero, all threads skip together!
        if (count == 0) {
            A += BK;
            BT += BK;
            continue;
        }

        // Load A tile (BM × BK) into shared memory
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // Load B^T tile (BN × BK) into shared memory
        // B^T is N×K, so we load BN rows × BK columns
        for (int8_t offset = 0; offset + rowStrideBT <= BN; offset += rowStrideBT) {
            if (innerRowBT + offset < BN) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                    &BT[(innerRowBT + offset) * K + innerColBT * 4])[0];
                // Store in shared memory: BTs[row * BK + col]
                BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 0] = tmp.x;
                BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 1] = tmp.y;
                BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 2] = tmp.z;
                BTs[(innerRowBT + offset) * BK + innerColBT * 4 + 3] = tmp.w;
            }
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
        BT += BK;
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
