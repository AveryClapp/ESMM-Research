#pragma once
/*
 * ============================================================================
 * Kernel: ESMM B-Sparse Warp-Granularity
 * ============================================================================
 *
 * Strategy:
 *   Use warp-granularity patterns (BK × WN blocks) to enable warp-uniform
 *   skipping of K-iterations where B matrix is sparse.
 *
 * Architecture:
 *   - Preprocessing: OR together sparsity across WN=32 columns → single 8-bit pattern
 *   - Runtime: Load pattern per warp-column, reconstruct offsets, switch on count
 *   - Computation: Template dispatch enables full loop unrolling
 *
 * Performance Characteristics:
 *   - Memory: 1 byte per 8×32 block (~64 KB for 4096×4096)
 *   - Divergence: Zero (warp-uniform pattern)
 *   - Overhead: Minimal (single byte load + bit manipulation)
 *
 * Key Difference from A-Sparse:
 *   - A-sparse: Pattern indexed by (warpRow, kBlock) - sparsity in A's columns
 *   - B-sparse: Pattern indexed by (kBlock, warpCol) - sparsity in B's rows
 *   - Both skip the SAME K-iterations (dotIdx), just different patterns
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/b_preprocessor_warp.cu"
#include <cuda_runtime.h>

__forceinline__ __device__ void multiply_offsets_1(int wSubRowIdx, int wSubColIdx, 
        int WNITER, float regM_val, 
        float* regN, float* threadResults, 
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 1 Op using offsetList[0]
    int off0 = offsetList[0];
    threadResults[threadResBase + off0] += regM_val * regN[regNBase + off0];
}

__forceinline__ __device__ void multiply_offsets_2(int wSubRowIdx, int wSubColIdx, 
        int WNITER, float regM_val, 
        float* regN, float* threadResults, 
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 2 Ops
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
}

__forceinline__ __device__ void multiply_offsets_4(int wSubRowIdx, int wSubColIdx, 
        int WNITER, float regM_val, 
        float* regN, float* threadResults, 
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 4 Ops
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
    threadResults[threadResBase + offsetList[2]] += regM_val * regN[regNBase + offsetList[2]];
    threadResults[threadResBase + offsetList[3]] += regM_val * regN[regNBase + offsetList[3]];
}

__forceinline__ __device__ void multiply_offsets_8(int wSubRowIdx, int wSubColIdx, 
        int WNITER, float regM_val, 
        float* regN, float* threadResults, 
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 8 Ops
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
    threadResults[threadResBase + offsetList[2]] += regM_val * regN[regNBase + offsetList[2]];
    threadResults[threadResBase + offsetList[3]] += regM_val * regN[regNBase + offsetList[3]];
    threadResults[threadResBase + offsetList[4]] += regM_val * regN[regNBase + offsetList[4]];
    threadResults[threadResBase + offsetList[5]] += regM_val * regN[regNBase + offsetList[5]];
    threadResults[threadResBase + offsetList[6]] += regM_val * regN[regNBase + offsetList[6]];
    threadResults[threadResBase + offsetList[7]] += regM_val * regN[regNBase + offsetList[7]];
}

__forceinline__ __device__ void dispatch_multiply(int mode, int wSubRowIdx, int wSubColIdx, 
        int WNITER, float regM_val, 
        float* regN, float* threadResults, 
        const uint8_t* offsetList) {
    switch (mode) {
        case 1:
            multiply_offsets_1(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 2:
            multiply_offsets_2(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 4:
            multiply_offsets_4(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 8:
            multiply_offsets_8(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
            // Optional: default case for safety or error handling
    }
}

// ============================================================================
// Main kernel: B-sparse with warp-granularity patterns
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_b_sparse_warp(int M, int N, int K, float *A, float *B, float *C,
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


        // KEY DIFFERENCE: Pattern indexed by warp column (not warp row)
        const uint globalWarpCol = cCol * (BN / WN) + warpCol;
        const int numNBlocks = N / WN;

        for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
            const uint kBlock = bkIdx / BK;

            // B-sparse: Pattern indexed by (kBlock, warpCol)
            const uint blockId = kBlock * numNBlocks + globalWarpCol;
            const uint8_t pattern = blockPatterns[blockId];

            const uint8_t count = PATTERN_LUT_BK8[pattern].count;
            const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;

            if (count == 0) {
                A += BK;
                B += BK * N;
                continue;
            }

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
                    regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
                    regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
                    regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
                    regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
                    regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
                    regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
                    regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
                    regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol *
                        WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
                }

                // Compute outer productn
                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    #pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        dispatch_multiply(count, wSubRowIdx, wSubColIdx, WNITER, 
                                regM[WMITER], regN, threadResults, offsets);
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
