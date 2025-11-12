#pragma once

/*
 * ============================================================================
 * Kernel 19 OPTIMIZED: ESMM Combined A+B Sparsity with Block-Level B-Skipping
 * ============================================================================
 *
 * STRATEGY: Skip loading entire B tiles when no computation is needed
 *
 * Key optimizations:
 *   1. Load B patterns to shared memory (coalesced, 16 bytes)
 *   2. Each thread checks if it needs B data (via joint A & B patterns)
 *   3. Warp-level voting (__any_sync) to aggregate needs
 *   4. Block-level aggregation to decide: load B or skip?
 *   5. ONLY load B tile if ANY thread needs it (saves bandwidth!)
 *   6. ONLY compute if we loaded B
 *
 * Expected performance at 50% sparsity:
 *   - 75% of B tiles can be skipped (joint density = 25%)
 *   - Save: 0.75 × 4KB × 512 = 1.5GB bandwidth per iteration
 *   - Target: 1.3-1.5× faster than K17 (A-only)
 *
 * At 90% sparsity:
 *   - 99% of B tiles skippable
 *   - Target: 5-10× faster than K17
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int SIZE>
__device__ void compute_sparse_block_combined_cached(
    const uint8_t* A_offsets,
    const uint8_t* shared_B_patterns,  // Read from shared memory instead of global
    const uint cCol, const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Cache B patterns from shared memory (already loaded)
    uint localNBlocks[WNITER];
    uint8_t B_patterns_cache[WNITER];
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        const uint globalColBase = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
        const uint globalNBlock = globalColBase >> 3;
        const uint localNBlock = globalNBlock - (cCol * (BN / TN));  // Local to this thread block
        localNBlocks[wSubColIdx] = localNBlock;
        B_patterns_cache[wSubColIdx] = shared_B_patterns[localNBlock];
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

            // Vectorized loads with ternary predication
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
                const int regNBase = wSubColIdx * TN;
                const int threadResBase = threadResRowBase + wSubColIdx * TN;

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

        // ============ SHARED MEMORY ============
        __shared__ float As[BM * BK];
        __shared__ float Bs[BN * BK];

        // Cache B patterns for this thread block's columns
        __shared__ uint8_t shared_B_patterns[BN / TN];

        // Block-level flag: does this block need B data?
        __shared__ bool block_needs_B;

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

            // STEP 1: Load B patterns for this K-block into shared memory (COALESCED!)
            constexpr uint PATTERNS_PER_BLOCK = BN / TN;
            const uint B_pattern_base = kBlock * numNBlocks + cCol * (BN / TN);
            for (uint offset = threadIdx.x; offset < PATTERNS_PER_BLOCK; offset += NUM_THREADS) {
                const uint globalNBlock = cCol * (BN / TN) + offset;
                if (globalNBlock < numNBlocks) {
                    shared_B_patterns[offset] = B_blockPatterns[B_pattern_base + offset];
                } else {
                    shared_B_patterns[offset] = 0;
                }
            }
            __syncthreads();

            // STEP 2: Get A pattern
            const uint A_blockId = globalWarpRow * numKBlocks + kBlock;
            const uint8_t A_pattern = A_blockPatterns[A_blockId];
            const uint8_t count = PATTERN_LUT_BK8[A_pattern].count;
            const uint8_t* offsets = PATTERN_LUT_BK8[A_pattern].offsets;

            // STEP 3: Check if this block needs B (BEFORE loading B!)
            // Each thread checks its columns
            bool thread_needs_B = false;
            if (count > 0) {  // Skip if A is all zeros
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const uint globalColBase = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
                    const uint globalNBlock = globalColBase >> 3;
                    const uint localNBlock = globalNBlock - (cCol * (BN / TN));
                    const uint8_t B_pattern = shared_B_patterns[localNBlock];

                    // Check joint pattern
                    if ((A_pattern & B_pattern) != 0) {
                        thread_needs_B = true;
                        break;
                    }
                }
            }

            // Warp-level vote: does ANY thread in this warp need B?
            bool warp_needs_B = __any_sync(0xFFFFFFFF, thread_needs_B);

            // Block-level aggregation using atomics (first thread of each warp)
            if (threadIdx.x == 0) {
                block_needs_B = false;  // Initialize
            }
            __syncthreads();

            if ((threadIdx.x % WARPSIZE) == 0 && warp_needs_B) {
                block_needs_B = true;  // Any warp needs B -> block needs B
            }
            __syncthreads();

            // STEP 4: Load A (always) and B (conditionally)
            // Load A always
            for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }

            // Load B ONLY if needed (KEY OPTIMIZATION!)
            if (block_needs_B) {
                for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                    reinterpret_cast<float4 *>(
                        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                        reinterpret_cast<const float4 *>(
                            &B[(innerRowB + offset) * N + innerColB * 4])[0];
                }
            }
            __syncthreads();

            // STEP 5: Compute ONLY if we loaded B
            if (block_needs_B && count > 0) {
                switch (count) {
                case 1:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 2:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 3:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 4:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 5:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 6:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 7:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                case 8:
                    compute_sparse_block_combined_cached<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                        offsets, shared_B_patterns, cCol, warpRow, warpCol,
                        threadRowInWarp, threadColInWarp, As, Bs, threadResults);
                    break;
                }
            }  // End if (block_needs_B && count > 0)

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
