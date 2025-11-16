#pragma once

/*
 * ============================================================================
 * Kernel 21: Joint A+B Sparsity with 1D Dispatch (SIMPLIFIED)
 * ============================================================================
 *
 * KEY INSIGHT: B-transpose makes B_patterns warp-uniform, so we can use
 * the same bitmask + LUT dispatch that works for A-sparsity!
 *
 * STRATEGY:
 *   1. Load A_pattern and B_pattern for current block intersection
 *   2. Look up offsets from PATTERN_LUT_BK8 for both
 *   3. Template-dispatch on A_count ONLY (not B_count)
 *   4. Runtime loop over B offsets (warp-uniform, no divergence!)
 *   5. Compute only where k_A == k_B (intersection)
 *
 * BENEFITS over 2D dispatch (K20):
 *   - Only 8 kernel variants instead of 64 (8× faster compile)
 *   - Smaller binary size (less instruction cache pressure)
 *   - Simpler code (easier to maintain/optimize)
 *   - ZERO performance loss (B loop is warp-uniform)
 *
 * PERFORMANCE:
 *   - Should match K17 at A-only sparsity
 *   - Should match K20 at joint sparsity
 *   - Much faster compilation
 */

#include "../include/utils.cuh"
#include "../include/metadata.cuh"
#include "../include/pattern_lut.cuh"
#include "../src/preprocessors/a_preprocessor_hybrid.cu"
#include "../src/preprocessors/b_transpose_preprocessor.cu"
#include <cuda_runtime.h>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

/*
 * compute_joint_sparse_1D: Simplified joint sparsity compute function
 *
 * Template Parameters:
 *   - A_SIZE: Compile-time A sparsity count (enables full unrolling)
 *   - Other params: Standard warp-tiling configuration
 *
 * Runtime Parameters:
 *   - A_offsets: Array of A's non-zero K indices (size = A_SIZE)
 *   - B_offsets: Array of B's non-zero K indices (size = B_count)
 *   - B_count: Runtime B sparsity count (warp-uniform!)
 *
 * OPTIMIZATION: The outer loop (A indices) is compile-time unrolled.
 *               The inner loop (B indices) is runtime but warp-uniform.
 *               Intersection check (k_A == k_B) has zero divergence!
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_SIZE>
__device__ __forceinline__ void compute_joint_sparse_1D(
    const uint8_t* __restrict__ A_offsets,    // A indices (compile-time size = A_SIZE)
    const uint8_t* __restrict__ B_offsets,    // B indices (runtime size = B_count)
    const uint8_t B_count,                     // Runtime B sparsity
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Outer loop: A indices (compile-time unrolled via template parameter)
    #pragma unroll
    for (uint a_idx = 0; a_idx < A_SIZE; a_idx++) {
        const uint8_t k_A = A_offsets[a_idx];

        // Load A values at k_A (fully unrolled)
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint tm = 0; tm < TM; ++tm) {
                const uint mOffset = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + tm;
                regM[wSubRowIdx * TM + tm] = As[k_A * BM + mOffset];
            }
        }

        // Inner loop: B indices (runtime loop, but WARP-UNIFORM!)
        // All threads in warp iterate same number of times (B_count is same for all)
        // Compiler will likely unroll this with #pragma unroll 8 hint
        #pragma unroll 8
        for (uint b_idx = 0; b_idx < B_count; b_idx++) {
            const uint8_t k_B = B_offsets[b_idx];

            // Intersection check: Only compute if k_A == k_B
            // This is WARP-UNIFORM because all threads compare same values
            // No branch divergence penalty!
            if (k_A == k_B) {
                // Load B values at k_B (fully unrolled)
                const float* Bs_row = &Bs[k_B * BN];
                const uint nBase = warpCol * WN + threadColInWarp * TN;

                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        const uint nOffset = nBase + wSubColIdx * WSUBN + tn;
                        regN[wSubColIdx * TN + tn] = Bs_row[nOffset];
                    }
                }

                // Compute outer product (fully unrolled)
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

                break;  // Found intersection, move to next A index
            }
        }
    }
}

/*
 * Main Kernel: Joint A+B Sparsity with 1D Template Dispatch
 *
 * Differences from K20:
 *   - Single switch on A_count (not nested A×B switch)
 *   - B_count passed as runtime parameter
 *   - 8 kernel instantiations instead of 64
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_joint_1d(
        int M, int N, int K,
        float* A, float* B, float* C,
        const uint8_t* __restrict__ A_blockPatterns,
        const uint8_t* __restrict__ B_blockPatterns,
        const int numKBlocks_A,  // K/BK for A (row-wise)
        const int numNBlocks_B,  // N/WN for B (column-wise)
        const int numKBlocks_B)  // K/BK for B (column-wise)
{
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
    const uint globalWarpCol = cCol * (BN / WN) + warpCol;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Get A pattern for this warp's row
        const uint A_blockId = globalWarpRow * numKBlocks_A + kBlock;
        const uint8_t A_pattern = A_blockPatterns[A_blockId];

        // Get B pattern for this warp's column
        const uint B_blockId = globalWarpCol * numKBlocks_B + kBlock;
        const uint8_t B_pattern = B_blockPatterns[B_blockId];

        // Early exit if either A or B is fully sparse
        if (A_pattern == 0 || B_pattern == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Look up offsets from LUT
        const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t B_count = PATTERN_LUT_BK8[B_pattern].count;
        const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;
        const uint8_t* B_offsets = PATTERN_LUT_BK8[B_pattern].offsets;

        // Load A tile (always needed if A_pattern != 0)
        #pragma unroll
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // Load B tile (always needed if B_pattern != 0)
        #pragma unroll
        for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        // 1D Template Dispatch: Only on A_count (8 cases instead of 64!)
        // B_count is passed as runtime parameter (warp-uniform)
        switch (A_count) {
            case 1:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 2:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 3:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 4:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 5:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 6:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 7:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
            case 8:
                compute_joint_sparse_1D<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                    A_offsets, B_offsets, B_count, warpRow, warpCol,
                    threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                break;
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results to C
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
                #pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp;
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                  threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
