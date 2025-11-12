#pragma once

/*
 * ============================================================================
 * Kernel 19+: ESMM Joint A+B Sparsity (B-Transpose + Intersection)
 * ============================================================================
 *
 * Strategy:
 *   1. Transpose B to B^T for warp-uniform B-sparsity checks
 *   2. Use BOTH A and B^T offset lists in K-dimension
 *   3. Compute INTERSECTION: only process K-indices where BOTH are non-zero
 *   4. 2D dispatch on (A_COUNT, B_COUNT) for fully unrolled loops
 *
 * Architecture:
 *   - Warp tile: 64 rows × 32 columns
 *   - Thread tile: TM=1, TN=8
 *   - A patterns: per WM×BK block (K-dimension)
 *   - B^T patterns: per WN×BK block (K-dimension, column-wise analysis)
 *   - 64 templated variants (8 A-counts × 8 B-counts)
 *
 * Joint Sparsity Benefits:
 *   - Early exit: Skip K-blocks where EITHER A or B is fully sparse
 *   - Intersection: Within each K-block, only compute FMAs where BOTH are non-zero
 *   - At 50% A-sparsity × 50% B-sparsity → 25% density → 4× theoretical speedup
 *
 * Performance Characteristics:
 *   - Memory: A-patterns + B-patterns (~128 KB for 4096×4096)
 *   - Divergence: Zero (warp-uniform checks, compile-time unrolled loops)
 *   - Compute: Multiplicative sparsity benefit (not additive)
 *   - Memory access: All coalesced for both A and B^T
 *
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_transpose_preprocessor.cu"
#include <cuda_runtime.h>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif


template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT, const int B_COUNT>
__device__ __forceinline__ void compute_joint_sparse_block(
    const uint8_t* a_offsets,
    const uint8_t* b_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* BTs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // ====================================================================
    // JOINT SPARSITY: Iterate over A offsets, check if B also has them
    // ====================================================================
    #pragma unroll
    for (int a_idx = 0; a_idx < A_COUNT; ++a_idx) {
        const uint8_t dotIdx = a_offsets[a_idx];

        // Check if B also has this K-index (fully unrolled at compile time)
        bool b_has_offset = false;
        #pragma unroll
        for (int b_idx = 0; b_idx < B_COUNT; ++b_idx) {
            if (b_offsets[b_idx] == dotIdx) {
                b_has_offset = true;
                break;
            }
        }

        if (!b_has_offset) continue;  // Skip: A has it but B doesn't

        // ====================================================================
        // Load A values from shared memory (column-major: As[k * BM + m])
        // ====================================================================
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

        // ====================================================================
        // Load B^T values from shared memory (row-major: BTs[k * BN + n])
        // All threads in warp access the same k-row → warp-uniform!
        // ====================================================================
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

        // ====================================================================
        // Outer product: C += A * B^T
        // Fully unrolled for maximum ILP
        // ====================================================================
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (uint tm = 0; tm < TM; ++tm) {
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        // Result indexing: [wSubRow * TM * totalCols + wSubCol * TN + specific position]
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

// ============================================================================
// 2D Dispatch on (A_COUNT, B_COUNT)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT>
__device__ __forceinline__ void dispatch_B_count(
    const uint8_t b_count,
    const uint8_t* a_offsets,
    const uint8_t* b_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* BTs,
    float* threadResults) {

    switch(b_count) {
        case 1: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,1>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 2: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,2>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 3: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,3>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 4: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,4>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 5: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,5>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 6: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,6>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 7: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,7>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 8: compute_joint_sparse_block<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,8>(
            a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__device__ __forceinline__ void dispatch_A_B_count(
    const uint8_t a_count,
    const uint8_t b_count,
    const uint8_t* a_offsets,
    const uint8_t* b_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* BTs,
    float* threadResults) {

    switch(a_count) {
        case 1: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,1>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 2: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,2>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 3: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,3>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 4: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,4>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 5: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,5>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 6: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,6>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 7: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,7>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
        case 8: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,8>(
            b_count, a_offsets, b_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, BTs, threadResults); break;
    }
}


template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_btranspose(int M, int N, int K,
                    float *A, float *B, float *C,
                    const uint8_t* __restrict__ a_blockPatterns,
                    const uint8_t* __restrict__ b_blockPatterns,
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

    // Pattern indexing for joint sparsity
    const uint globalWarpRow = cRow * (BM / WM) + warpRow;
    const uint globalColBlock = cCol * (BN / WN) + warpCol;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // ====================================================================
        // JOINT SPARSITY: Check BOTH A and B patterns BEFORE loading
        // ====================================================================
        const uint a_patternIdx = globalWarpRow * numKBlocks + kBlock;
        const uint b_patternIdx = globalColBlock * numKBlocks + kBlock;

        const uint8_t a_pattern = a_blockPatterns[a_patternIdx];
        const uint8_t b_pattern = b_blockPatterns[b_patternIdx];

        const uint8_t a_count = PATTERN_LUT_BK8[a_pattern].count;
        const uint8_t b_count = PATTERN_LUT_BK8[b_pattern].count;

        // Skip ENTIRE iteration if either A or B is fully sparse (skip loading!)
        if (a_count == 0 || b_count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Get BOTH offset lists for joint sparsity computation
        const uint8_t* a_offsets = PATTERN_LUT_BK8[a_pattern].offsets;
        const uint8_t* b_offsets = PATTERN_LUT_BK8[b_pattern].offsets;


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

            BTs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // ====================================================================
        // JOINT SPARSITY DISPATCH: 2D dispatch on (A_COUNT, B_COUNT)
        // Only processes K-indices where BOTH A and B are non-zero
        // ====================================================================
        dispatch_A_B_count<BM, BN, BK, WM, WN, WNITER, TM, TN>(
            a_count, b_count, a_offsets, b_offsets,
            warpRow, warpCol, threadRowInWarp, threadColInWarp,
            As, BTs, threadResults);

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
