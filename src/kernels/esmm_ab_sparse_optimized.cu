#pragma once

// K20: Joint A+B sparsity with zero-overhead inner loop
// Direct bit checking (no offset array), sequential iteration

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
esmm_ab_sparse_optimized(
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

    // Shared memory: A tile, B tile, AND precomputed joint patterns
    __shared__ float As[BK * (BM + 1)];  // Padded to avoid bank conflicts
    __shared__ float Bs[BK * BN];
    __shared__ uint8_t joint_smem[NUM_WARPS * 1024];

    // ========================================================================
    // OPTIMIZATION: Precompute ALL joint patterns at kernel start
    // ========================================================================
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

            joint_smem[i] = a_pat & b_pat;  // PRECOMPUTE INTERSECTION
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * numKBlocks;

    // Standard GEMM setup
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

    // ========================================================================
    // K-LOOP with optimized inner loop (K21 style)
    // ========================================================================
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        uint8_t joint;
        if (laneId == 0) {
            joint = my_joints[kBlock];
        }
        joint = __shfl_sync(0xFFFFFFFF, joint, 0);

        if (joint == 0) {
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

        // ====================================================================
        // OPTIMIZED INNER LOOP: K21 style with direct bit checking
        // ====================================================================
        float regM[WMITER * TM];
        float regN[WNITER * TN];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // ALL 32 threads evaluate the SAME condition - zero divergence
            // Direct bit check - NO offset array computation
            if (!(joint & (1 << dotIdx))) {
                continue;
            }

            // Load A values into registers
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];
            }

            // Load B values into registers
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
                    #pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) + wSubColIdx * TN + resIdxN] +=
                            regM[wSubRowIdx * TM + 0] * regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write results back to C
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
