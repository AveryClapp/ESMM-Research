#pragma once

// K29: K26 + two occupancy optimizations:
//   1. MAX_K_BLOCKS templated (not hardcoded 1024) — halves pattern smem for typical sizes
//   2. Float2 A-tile loads — all 256 threads contribute (vs 128 idle with float4)

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS,
          const int MAX_K_BLOCKS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_optimized_v3(
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

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
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
    __shared__ uint8_t joint_smem[NUM_WARPS * MAX_K_BLOCKS];
    __shared__ uint8_t block_joint[MAX_K_BLOCKS];

    // =========================================================================
    // Step 1: Precompute per-warp joint patterns into joint_smem
    // =========================================================================
    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;
        const int totalPatterns = NUM_WARPS * numKBlocks;

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / numKBlocks;
            const int kBlock = i % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            const int gWarpRow = globalMBlock * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            const uint8_t a_pat = a_patterns[gWarpRow * numKBlocks + kBlock];
            const uint8_t b_pat = b_patterns[gWarpCol * numKBlocks + kBlock];

            joint_smem[i] = a_pat & b_pat;
        }
        __syncthreads();
    }

    // =========================================================================
    // Step 2: Compute block-level joint (OR across all warps)
    // =========================================================================
    for (int k = threadIdx.x; k < numKBlocks; k += NUM_THREADS) {
        uint8_t bj = 0;
        #pragma unroll
        for (int w = 0; w < (int)NUM_WARPS; ++w) {
            bj |= joint_smem[w * numKBlocks + k];
        }
        block_joint[k] = bj;
    }
    __syncthreads();

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * numKBlocks;

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // Float2 A-tile indexing: all 256 threads map to valid rows (rowStrideA = BM)
    const uint innerRowA = threadIdx.x / (BK / 2);
    const uint innerColA = threadIdx.x % (BK / 2);
    constexpr uint rowStrideA = (NUM_THREADS * 2) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // =========================================================================
    // K-loop
    // =========================================================================
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        if (block_joint[kBlock] == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Float2 A-tile load: all 256 threads contribute, no bounds check needed
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            const float2 tmp = reinterpret_cast<const float2*>(
                &A[(innerRowA + offset) * K + innerColA * 2])[0];
            As[(innerColA * 2 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 2 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
        }

        // Float4 B-tile load: unchanged, all 256 threads contribute
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        uint8_t joint;
        if (laneId == 0) {
            joint = my_joints[kBlock];
        }
        joint = __shfl_sync(0xFFFFFFFF, joint, 0);

        if (joint != 0) {
            float regM[WMITER * TM];
            float regN[WNITER * TN];

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                if (!(joint & (1 << dotIdx))) {
                    continue;
                }

                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];
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
                    const float valM = regM[wSubRowIdx * TM + 0];
                    #pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        const uint resBase = (wSubRowIdx * TM + 0) * (WNITER * TN) + (wSubColIdx * TN);
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
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
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
                        &C_sub[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
