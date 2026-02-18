#pragma once

// K25: Simple Fused - Fused preprocessing + K20 computation in one kernel launch.
// Preprocessing kernels (preprocess_a_fused, preprocess_b_fused) live in ab_preprocessor.cu.

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// Copy of K20 computation kernel (esmm_ab_sparse_optimized), co-located here
// so K25's single-launch fused approach is self-contained at the compute level.
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_compute_inline(
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
    const uint laneId = threadIdx.x % WARPSIZE;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    constexpr uint NUM_WARPS_M = BM / WM;
    constexpr uint NUM_WARPS_N = BN / WN;
    constexpr uint NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BK * (BM + 1)];
    __shared__ float Bs[BK * BN];
    // Per-block joint pattern: OR of all warp joint patterns in this tile.
    // MAX_K_BLOCKS covers K up to 16384 with BK=8 (16384/8 = 2048).
    constexpr int MAX_K_BLOCKS = 2048;
    __shared__ uint8_t block_joint[MAX_K_BLOCKS];

    // Compute block-level joint pattern (OR across all warps in this tile)
    {
        for (int k = threadIdx.x; k < numKBlocks; k += NUM_THREADS) {
            uint8_t block_pat = 0;
            #pragma unroll
            for (int wr = 0; wr < (int)NUM_WARPS_M; ++wr) {
                const int gRow = cRow * NUM_WARPS_M + wr;
                #pragma unroll
                for (int wc = 0; wc < (int)NUM_WARPS_N; ++wc) {
                    const int gCol = cCol * NUM_WARPS_N + wc;
                    const uint8_t a_pat = a_patterns[gRow * numKBlocks + k];
                    const uint8_t b_pat = b_patterns[gCol * numKBlocks + k];
                    block_pat |= (a_pat & b_pat);
                }
            }
            block_joint[k] = block_pat;
        }
    }

    const int gWarpRow = cRow * NUM_WARPS_M + warpRow;
    const int gWarpCol = cCol * NUM_WARPS_N + warpCol;
    const uint8_t* my_a_pats = a_patterns + gWarpRow * numKBlocks;
    const uint8_t* my_b_pats = b_patterns + gWarpCol * numKBlocks;

    __syncthreads();

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * WNITER * TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Block-level skip: if no warp in this tile has any non-zero joint, skip entirely
        if (block_joint[kBlock] == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }

        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // Warp-level skip: check per-warp joint pattern
        uint8_t cur_joint;
        if (laneId == 0) {
            cur_joint = my_a_pats[kBlock] & my_b_pats[kBlock];
        }
        cur_joint = __shfl_sync(0xFFFFFFFF, cur_joint, 0);

        if (cur_joint != 0) {
            float regM[WMITER];
            float regN[WNITER * TN];

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                if (!(cur_joint & (1 << dotIdx))) {
                    continue;
                }

                #pragma unroll WMITER
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp + 0];
                }

                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float4 tmp0 = reinterpret_cast<const float4*>(&Bs[(dotIdx * BN) + warpCol * WN +
                                            wSubColIdx * WSUBN + threadColInWarp * TN])[0];
                    const float4 tmp1 = reinterpret_cast<const float4*>(&Bs[(dotIdx * BN) + warpCol * WN +
                                            wSubColIdx * WSUBN + threadColInWarp * TN + 4])[0];
                    regN[wSubColIdx * TN + 0] = tmp0.x;
                    regN[wSubColIdx * TN + 1] = tmp0.y;
                    regN[wSubColIdx * TN + 2] = tmp0.z;
                    regN[wSubColIdx * TN + 3] = tmp0.w;
                    regN[wSubColIdx * TN + 4] = tmp1.x;
                    regN[wSubColIdx * TN + 5] = tmp1.y;
                    regN[wSubColIdx * TN + 6] = tmp1.z;
                    regN[wSubColIdx * TN + 7] = tmp1.w;
                }

                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    const float valM = regM[wSubRowIdx];
                    #pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        const uint resBase = wSubRowIdx * (WNITER * TN) + (wSubColIdx * TN);
                        const uint nBase = wSubColIdx * TN;
                        float4 acc_0 = *(float4*)(&threadResults[resBase]);
                        float4 reg_0 = *(float4*)(&regN[nBase]);
                        float4 acc_1 = *(float4*)(&threadResults[resBase + 4]);
                        float4 reg_1 = *(float4*)(&regN[nBase + 4]);

                        acc_0.x += valM * reg_0.x; acc_0.y += valM * reg_0.y;
                        acc_0.z += valM * reg_0.z; acc_0.w += valM * reg_0.w;
                        acc_1.x += valM * reg_1.x; acc_1.y += valM * reg_1.y;
                        acc_1.z += valM * reg_1.z; acc_1.w += valM * reg_1.w;

                        *(float4*)(&threadResults[resBase]) = acc_0;
                        *(float4*)(&threadResults[resBase + 4]) = acc_1;
                    }
                }
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            #pragma unroll
            for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                float4 tmp;
                const int i = (wSubRowIdx) * (WNITER * TN) +
                    wSubColIdx * TN + resIdxN;
                tmp.x = threadResults[i + 0];
                tmp.y = threadResults[i + 1];
                tmp.z = threadResults[i + 2];
                tmp.w = threadResults[i + 3];
                reinterpret_cast<float4*>(
                    &C_sub[(threadRowInWarp) * N +
                    threadColInWarp * TN + resIdxN])[0] = tmp;
            }
        }
    }
}
