#pragma once

// K23: A-sparse with block-wise uniform patterns. Just skipping like K19 (8x32 blocks)

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

// ============================================================================
// Main kernel: Block-wise uniform sparsity
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_a_sparse_blockwise_skip(int M, int N, int K, float *A, float *B, float *C,
                          const uint8_t* __restrict__ blockPatterns,
                          const int numKBlocks) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // Pad As to avoid bank conflicts (BM=64 is power-of-2)
    __shared__ float As[(BM + 1) * BK];
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

    const uint globalWarpRow = cRow * (BM / WM) + warpRow;

    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        const uint8_t a_pattern = blockPatterns[globalWarpRow * numKBlocks + kBlock];

        // Skip entire BK tile if completely zero
        if (!__popc(a_pattern)) {
            A += BK;
            B += BK * N;
            continue;
        }

        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            // Store with padding (BM+1) to avoid bank conflicts
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
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
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        if (!(a_pattern & (1 << dotIdx))) {
            continue;
        }
        // Load from shared memory A (using BM+1 for padded layout)
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        // Load from shared memory B
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

        // Compute outer product
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

    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;

            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
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
