#pragma once

// K27: Ablation baseline — K20 with 32-row A granularity only.
// No block-level skip, no float4 inner loop.
// Isolates the effect of granularity change from K26's other optimizations.

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
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_optimized_v2_baseline(
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
    constexpr int MAX_K_BLOCKS = 1024;
    __shared__ uint8_t joint_smem[NUM_WARPS * MAX_K_BLOCKS];

    // Precompute per-warp joint patterns (same as K20/K26)
    {
        const int totalPatterns = NUM_WARPS * numKBlocks;
        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / numKBlocks;
            const int kBlock = i % numKBlocks;
            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;
            const int gWarpRow = cRow * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = cCol * NUM_WARPS_N + localWarpCol;
            const uint8_t a_pat = a_patterns[gWarpRow * numKBlocks + kBlock];
            const uint8_t b_pat = b_patterns[gWarpCol * numKBlocks + kBlock];
            joint_smem[i] = a_pat & b_pat;
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * numKBlocks;

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

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // rowStrideA=128 > BM=64: only threads with innerRowA < BM write
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
            }
        }

        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                reinterpret_cast<float4*>(
                    &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4*>(
                        &B[(innerRowB + offset) * N + innerColB * 4])[0];
            }
        }

        __syncthreads();

        // Per-warp skip: safe with NUM_WARPS_M=2 because we use if() not continue
        uint8_t joint;
        if (laneId == 0) joint = my_joints[kBlock];
        joint = __shfl_sync(0xFFFFFFFF, joint, 0);

        if (joint != 0) {
            float regM[WMITER * TM];
            float regN[WNITER * TN];

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                if (!(joint & (1 << dotIdx))) continue;

                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];
                }

                // Scalar B loads (no float4 — ablation baseline)
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        regN[wSubColIdx * TN + tn] = Bs[(dotIdx * BN) + warpCol * WN +
                            wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                    }
                }

                // Scalar accumulate (no float4 — ablation baseline)
                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    const float valM = regM[wSubRowIdx * TM + 0];
                    #pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        #pragma unroll
                        for (uint tn = 0; tn < TN; ++tn) {
                            threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) +
                                wSubColIdx * TN + tn] += valM * regN[wSubColIdx * TN + tn];
                        }
                    }
                }
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write back (float4 stores, same as K20/K26)
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
                    tmp.x = threadResults[i + 0]; tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2]; tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4*>(
                        &C_sub[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
