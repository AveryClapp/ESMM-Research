#pragma once
/*
 * ============================================================================
 * Kernel: ESMM B-Sparse TN-Granularity (8-column blocks)
 * ============================================================================
 *
 * Strategy:
 *   Use TN-granularity patterns (BK × TN blocks) to enable per-thread-group
 *   sparsity exploitation. Each pattern covers the 8 columns that a thread
 *   group actually works on.
 *
 * Architecture:
 *   - Preprocessing: OR together sparsity across TN=8 columns → single 8-bit pattern
 *   - Runtime: Each thread group loads its own pattern (4 patterns per warp)
 *   - Computation: Template dispatch enables full loop unrolling
 *
 * Performance Characteristics:
 *   - Memory: 4× more patterns than WN-granularity (~256 KB for 4096×4096)
 *   - Divergence: Potentially higher (different thread groups may diverge)
 *   - Precision: Better (only compute what each thread group needs)
 *
 * Key Difference from WN-granularity:
 *   - WN=32: One pattern per warp → all threads use same pattern → warp-uniform
 *   - TN=8: One pattern per thread group → different groups may diverge → thread-divergent
 *   - Tradeoff: Finer granularity vs. potential warp divergence
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../../include/b_sparse_helpers.cuh"
#include "../preprocessors/b_preprocessor_tn.cu"
#include <cuda_runtime.h>

// ============================================================================
// Main kernel: B-sparse with TN-granularity patterns (8 columns per pattern)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_b_sparse_tn(int M, int N, int K, float *A, float *B, float *C,
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

        float regM[WMITER * TM];
        float regN[WNITER * TN];

        const uint baseGlobalTNBlock = (cCol * BN + warpCol * WN) / TN;
        const int numTNBlocks = N / TN;  // Number of 8-column blocks

        for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
            const uint kBlock = bkIdx / BK;

            // Load A tile
            for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                const float4 tmp = reinterpret_cast<const float4 *>(
                        &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }

            // Load B tile
            for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                reinterpret_cast<float4 *>(
                        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4 *>(
                            &B[(innerRowB + offset) * N + innerColB * 4])[0];
            }
            __syncthreads();

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // Load from shared memory A
                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp * TM];
                }

                // Load from shared memory B
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
                    regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
                    regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
                    regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
                    regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
                    regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
                    regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
                    regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
                }

                // Compute outer product
                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    #pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        const uint localTNBlockInWarp = wSubColIdx * (WSUBN / TN) + threadColInWarp;
                        const uint globalTNBlock = baseGlobalTNBlock + localTNBlockInWarp;
                        const uint blockId = kBlock * numTNBlocks + globalTNBlock;
                        const uint8_t pattern = blockPatterns[blockId];
                        const uint8_t count = PATTERN_LUT_BK8[pattern].count;
                        const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;
                        const float regM_val = regM[wSubRowIdx];
                        
                        switch (count) {
                            case 2:
                                multiply_offsets_2(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsets);
                            case 4:
                                multiply_offsets_4(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsets);
                                break;
                            default:
                                multiply_offsets_8(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
                                break;
                        }
                    }
                }
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
