#pragma once

/*
 * ============================================================================
 * Kernel 18 OPTIMIZED: ESMM Combined A+B Sparsity (Block-Level Skipping)
 * ============================================================================
 *
 * THREE-PHASE OPTIMIZATION STRATEGY:
 *
 * Phase 1: Block-Level B Load Skipping
 *   - Precompute which B blocks ANY thread needs (joint A∧B patterns)
 *   - Only load B tiles that contain useful data
 *   - Skip entire blocks when all patterns are zero
 *   - Expected: 33% speedup (9.8ms → 6.5ms)
 *
 * Phase 2: Pattern Type Dispatch
 *   - Classify patterns: fully_sparse (0), sparse (1-4), dense (5-7), fully_dense (8)
 *   - Optimized code paths for each type
 *   - Eliminate ternary overhead for common cases
 *   - Expected: +15% speedup (6.5ms → 5.5ms)
 *
 * Phase 3: Warp-Uniform Decisions
 *   - Make skip decisions at warp granularity
 *   - Reduce warp divergence
 *   - Expected: +4% speedup (5.5ms → 5.3ms)
 *
 * TOTAL EXPECTED: 9.8ms → 5.3ms (45% faster, 26% faster than K17's 7.2ms)
 *
 * Key Parameters:
 *   BM=128, BN=128, BK=8, WM=32, WN=64, WNITER=4, TM=1, TN=8
 *   NUM_THREADS=256, WARPS_PER_BLOCK=8
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

// ============================================================================
// Main kernel: Optimized Combined A+B sparsity
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
        const uint laneIdx = threadIdx.x % WARPSIZE;
        const uint warpCol = warpIdx % (BN / WN);
        const uint warpRow = warpIdx / (BN / WN);

        constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
        constexpr uint WSUBM = WM / WMITER;
        constexpr uint WSUBN = WN / WNITER;
        constexpr uint WARPS_PER_BLOCK = NUM_THREADS / WARPSIZE;

        const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
        const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
        const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

        // ============ SHARED MEMORY DECLARATIONS ============
        __shared__ float As[BM * BK];
        __shared__ float Bs[BN * BK];

        // Phase 1: Block-level decisions
        __shared__ bool need_B_block[512];  // Max K/BK (4096/8 = 512)

        // Phase 2: Pattern storage
        __shared__ uint8_t joint_patterns[512][WNITER];
        __shared__ uint8_t pattern_popcounts[512][WNITER];

        // Phase 3: Warp-level decisions
        __shared__ bool warp_needs_compute[WARPS_PER_BLOCK][512];

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

        // ============ ONE-TIME SETUP (All 3 Phases) ============

        // Phase 2: Precompute joint patterns for ALL columns
        if (threadIdx.x < (BN / TN)) {  // Parallel: 16 threads
            const int colIdx = threadIdx.x;
            const uint globalCol = cCol * BN + colIdx * TN;
            const uint nBlock = globalCol / TN;

            for (int kb = 0; kb < numKBlocks; kb++) {
                uint8_t a_pattern = A_blockPatterns[globalWarpRow * numKBlocks + kb];
                uint8_t b_pattern = B_blockPatterns[kb * numNBlocks + nBlock];
                uint8_t joint = a_pattern & b_pattern;

                joint_patterns[kb][colIdx] = joint;
                pattern_popcounts[kb][colIdx] = __popc(joint);
            }
        }

        // Phase 1: Block-level decisions
        if (threadIdx.x == 0) {
            for (int kb = 0; kb < numKBlocks; kb++) {
                bool any_nonzero = false;
                for (int colIdx = 0; colIdx < (BN / TN); colIdx++) {
                    if (pattern_popcounts[kb][colIdx] > 0) {
                        any_nonzero = true;
                        break;
                    }
                }
                need_B_block[kb] = any_nonzero;
            }
        }

        // Phase 3: Warp-level decisions  
        if (laneIdx == 0) {
            for (int kb = 0; kb < numKBlocks; kb++) {
                bool any = false;
                const int warpColStart = warpCol * (WN / TN);
                const int warpColEnd = warpColStart + (WN / TN);

                for (int colIdx = warpColStart; colIdx < warpColEnd; colIdx++) {
                    if (pattern_popcounts[kb][colIdx] > 0) {
                        any = true;
                        break;
                    }
                }
                warp_needs_compute[warpIdx][kb] = any;
            }
        }

        __syncthreads();

        // ============ MAIN COMPUTATION LOOP ============
        float regM[WMITER * TM];
        float regN[WNITER * TN];

        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            const uint kb = bkIdx / BK;

            // Phase 3: Warp-uniform early exit
            if (!warp_needs_compute[warpIdx][kb]) {
                A += BK;
                B += BK * N;
                continue;
            }

            // Phase 1: Conditional B loading
            // Always load A (needed by multiple B columns)
            for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                        &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }

            // Conditionally load B based on Phase 1 decision
            if (need_B_block[kb]) {
                for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                    reinterpret_cast<float4 *>(
                            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                        reinterpret_cast<const float4 *>(
                                &B[(innerRowB + offset) * N + innerColB * 4])[0];
                }
            }
            __syncthreads();

            // Phase 1: Block-level skip
            if (!need_B_block[kb]) {
                A += BK;
                B += BK * N;
                __syncthreads();
                continue;
            }

            // Get A pattern and dispatch
            const uint A_blockId = globalWarpRow * numKBlocks + kb;
            const uint8_t A_pattern = A_blockPatterns[A_blockId];
            const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
            const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;

            // Early exit if A is all zeros
            if (A_count == 0) {
                A += BK;
                B += BK * N;
                __syncthreads();
                continue;
            }

            // Inner computation loop - iterate over non-zero A elements
            for (int sparse_idx = 0; sparse_idx < A_count; ++sparse_idx) {
                const uint8_t dotIdx = A_offsets[sparse_idx];

                // Load A into registers
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp * TM];
                }

                // Phase 2: Pattern-based B loading with dispatch
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const int colIdx = warpCol * (WN / TN) + wSubColIdx * (WSUBN / TN);
                    uint8_t pattern = joint_patterns[kb][colIdx];
                    uint8_t count = pattern_popcounts[kb][colIdx];

                    const uint baseAddr = (dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;

                    if (count == 0) {
                        // Fully sparse - skip entirely
#pragma unroll
                        for (int i = 0; i < TN; i++) {
                            regN[wSubColIdx * TN + i] = 0.0f;
                        }
                    }
                    else if (count == 8) {
                        // Fully dense - no masking needed
                        float4 tmp1 = reinterpret_cast<const float4*>(&Bs[baseAddr + 0])[0];
                        float4 tmp2 = reinterpret_cast<const float4*>(&Bs[baseAddr + 4])[0];
                        regN[wSubColIdx * TN + 0] = tmp1.x;
                        regN[wSubColIdx * TN + 1] = tmp1.y;
                        regN[wSubColIdx * TN + 2] = tmp1.z;
                        regN[wSubColIdx * TN + 3] = tmp1.w;
                        regN[wSubColIdx * TN + 4] = tmp2.x;
                        regN[wSubColIdx * TN + 5] = tmp2.y;
                        regN[wSubColIdx * TN + 6] = tmp2.z;
                        regN[wSubColIdx * TN + 7] = tmp2.w;
                    }
                    else if (count >= 5) {
                        // Mostly dense - load all, mask few
                        float4 tmp1 = reinterpret_cast<const float4*>(&Bs[baseAddr + 0])[0];
                        float4 tmp2 = reinterpret_cast<const float4*>(&Bs[baseAddr + 4])[0];
                        regN[wSubColIdx * TN + 0] = (pattern & 0x01) ? tmp1.x : 0.0f;
                        regN[wSubColIdx * TN + 1] = (pattern & 0x02) ? tmp1.y : 0.0f;
                        regN[wSubColIdx * TN + 2] = (pattern & 0x04) ? tmp1.z : 0.0f;
                        regN[wSubColIdx * TN + 3] = (pattern & 0x08) ? tmp1.w : 0.0f;
                        regN[wSubColIdx * TN + 4] = (pattern & 0x10) ? tmp2.x : 0.0f;
                        regN[wSubColIdx * TN + 5] = (pattern & 0x20) ? tmp2.y : 0.0f;
                        regN[wSubColIdx * TN + 6] = (pattern & 0x40) ? tmp2.z : 0.0f;
                        regN[wSubColIdx * TN + 7] = (pattern & 0x80) ? tmp2.w : 0.0f;
                    }
                    else {
                        // Sparse - conditional scalar loads
#pragma unroll
                        for (int i = 0; i < TN; i++) {
                            regN[wSubColIdx * TN + i] = (pattern & (1 << i)) ?
                                Bs[baseAddr + i] : 0.0f;
                        }
                    }
                }

                // FMAs - compute outer product
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    const float regM_val = regM[wSubRowIdx];
                    const int threadResRowBase = wSubRowIdx * (WNITER * TN);

#pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        const int regNBase = wSubColIdx * TN;
                        const int threadResBase = threadResRowBase + wSubColIdx * TN;

                        // Unrolled FMAs
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

            A += BK;
            B += BK * N;
            __syncthreads();
        }

        // ============ WRITE RESULTS ============

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
