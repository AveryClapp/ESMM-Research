#pragma once

/*
 * ============================================================================
 * Joint A+B Sparse GEMM Kernel
 * ============================================================================
 *
 * KEY INSIGHT:
 *   Both A-pattern and B-pattern are warp-uniform.
 *   The intersection (AND) is also warp-uniform.
 *   Cost: ONE extra AND instruction per tile. That's it.
 *
 * PATTERN ACCESS:
 *   a_pattern: indexed by globalWarpRow (same for all 32 threads in warp)
 *   b_pattern: indexed by globalWarpCol (same for all 32 threads in warp)
 *   joint = a_pattern & b_pattern (one instruction, zero divergence)
 *
 * SKIP LOGIC:
 *   - A-only:  skip if A[k] is zero for all WM rows
 *   - B-only:  skip if B[k] is zero for all WN columns
 *   - Joint:   skip if EITHER is zero (intersection)
 *
 * EXPECTED SPEEDUP:
 *   At 50% A-sparsity + 50% B-sparsity (independent):
 *   - A-only skip: 50%
 *   - B-only skip: 50%
 *   - Joint skip:  75% (1 - 0.5 × 0.5)
 *   - Speedup: ~2-3× over dense
 */

#include <cuda_runtime.h>
#include "../preprocessors/ab_preprocessor.cu"

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_sparse(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ a_patterns,
    const uint8_t* __restrict__ b_patterns,
    int numKBlocks
) {
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
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // ========================================================================
    // WARP-UNIFORM PATTERN INDICES
    // ========================================================================
    // Both are the SAME for all 32 threads in a warp
    const uint globalWarpRow = cRow * (BM / WM) + warpRow;
    const uint globalWarpCol = cCol * (BN / WN) + warpCol;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const int kBlock = bkIdx / BK;

        // ====================================================================
        // JOINT PATTERN: ONE AND INSTRUCTION
        // ====================================================================
        const uint8_t a_pattern = a_patterns[globalWarpRow * numKBlocks + kBlock];
        const uint8_t b_pattern = b_patterns[globalWarpCol * numKBlocks + kBlock];
        const uint8_t joint = a_pattern & b_pattern;

        // Load tiles (must load even if this warp's joint=0, collaborative loading)
        #pragma unroll
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }

        #pragma unroll
        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // ====================================================================
        // INNER LOOP WITH JOINT PATTERN SKIPPING
        // ====================================================================
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // Skip if EITHER A or B is zero at this K-position
            // ALL 32 threads evaluate the SAME condition
            if (!(joint & (1 << dotIdx))) {
                continue;
            }

            // Load A values
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];
            }

            // Load B values
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                const uint bBase = (dotIdx * BN) + warpCol * WN +
                    wSubColIdx * WSUBN + threadColInWarp * TN;

                regN[wSubColIdx * TN + 0] = Bs[bBase + 0];
                regN[wSubColIdx * TN + 1] = Bs[bBase + 1];
                regN[wSubColIdx * TN + 2] = Bs[bBase + 2];
                regN[wSubColIdx * TN + 3] = Bs[bBase + 3];
                regN[wSubColIdx * TN + 4] = Bs[bBase + 4];
                regN[wSubColIdx * TN + 5] = Bs[bBase + 5];
                regN[wSubColIdx * TN + 6] = Bs[bBase + 6];
                regN[wSubColIdx * TN + 7] = Bs[bBase + 7];
            }

            // Outer product
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float aVal = regM[wSubRowIdx * TM + 0];
                    const int resBase = (wSubRowIdx * WNITER + wSubColIdx) * TN;

                    threadResults[resBase + 0] += aVal * regN[wSubColIdx * TN + 0];
                    threadResults[resBase + 1] += aVal * regN[wSubColIdx * TN + 1];
                    threadResults[resBase + 2] += aVal * regN[wSubColIdx * TN + 2];
                    threadResults[resBase + 3] += aVal * regN[wSubColIdx * TN + 3];
                    threadResults[resBase + 4] += aVal * regN[wSubColIdx * TN + 4];
                    threadResults[resBase + 5] += aVal * regN[wSubColIdx * TN + 5];
                    threadResults[resBase + 6] += aVal * regN[wSubColIdx * TN + 6];
                    threadResults[resBase + 7] += aVal * regN[wSubColIdx * TN + 7];
                }
            }
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;

            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                const int resBase = (wSubRowIdx * WNITER + wSubColIdx) * TN;

                float4 tmp;
                tmp.x = threadResults[resBase + 0];
                tmp.y = threadResults[resBase + 1];
                tmp.z = threadResults[resBase + 2];
                tmp.w = threadResults[resBase + 3];
                reinterpret_cast<float4*>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                    threadColInWarp * TN + 0])[0] = tmp;

                tmp.x = threadResults[resBase + 4];
                tmp.y = threadResults[resBase + 5];
                tmp.z = threadResults[resBase + 6];
                tmp.w = threadResults[resBase + 7];
                reinterpret_cast<float4*>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                    threadColInWarp * TN + 4])[0] = tmp;
            }
        }
    }
}
