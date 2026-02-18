#pragma once


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
esmm_ab_32x32(
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
    // Each warp needs WMITER patterns per K-block (one per 32-row sub-tile)
    __shared__ uint8_t joint_smem[NUM_WARPS * WMITER * 1024];

    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;
        // Total patterns: NUM_WARPS warps × WMITER sub-tiles × numKBlocks
        const int totalPatterns = NUM_WARPS * WMITER * numKBlocks;

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / (WMITER * numKBlocks);
            const int remainder = i % (WMITER * numKBlocks);
            const int wSubRowIdx = remainder / numKBlocks;  // Which 32-row sub-tile (0-1)
            const int kBlock = remainder % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            const int gWarpRow = globalMBlock * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            // A pattern: indexed by 32-row tile
            // tileRow = (warpRow * WMITER) + wSubRowIdx
            const int tileRow = gWarpRow * WMITER + wSubRowIdx;
            const uint8_t a_pat = a_patterns[tileRow * numKBlocks + kBlock];

            // B pattern: indexed by warp column (already 8×WN granularity)
            const uint8_t b_pat = b_patterns[gWarpCol * numKBlocks + kBlock];

            joint_smem[i] = a_pat & b_pat;  // PRECOMPUTE INTERSECTION
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    // Pointer to this warp's patterns: WMITER sub-tiles × numKBlocks
    const uint8_t* my_joints = joint_smem + myWarpId * (WMITER * numKBlocks);

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

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Load all WMITER joint patterns for this K-block
        uint8_t joints[WMITER];
        if (laneId == 0) {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                joints[m] = my_joints[m * numKBlocks + kBlock];
            }
        }
        #pragma unroll
        for (int m = 0; m < WMITER; ++m) {
            joints[m] = __shfl_sync(0xFFFFFFFF, joints[m], 0);
        }

        // Check if ALL sub-tiles are zero for this K-block
        uint8_t any_nonzero = 0;
        #pragma unroll
        for (int m = 0; m < WMITER; ++m) {
            any_nonzero |= joints[m];
        }

        if (any_nonzero == 0) {
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

//        float regM[WMITER * TM];
//       float regN[WNITER * TN];

        // Iterate over WMITER (M-iterations)
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const uint8_t joint = joints[wSubRowIdx];  // Pattern for THIS 32-row sub-tile

            if (joint == 0) continue;  // Skip this 32-row sub-tile entirely

            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // Iterate over BK dot products
                #pragma unroll
                for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    // Check pattern for THIS specific 32×32 tile
                    if (!(joint & (1 << dotIdx))) {
                        continue;
                    }

                    // Load A value for this sub-tile and dotIdx
                    float a_val = As[(dotIdx * (BM + 1)) + warpRow * WM +
                                     wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];

                    // Load B values for this column iteration
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        float b_val = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                        threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) + wSubColIdx * TN + tn] += a_val * b_val;
                    }
                }
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write results back to C
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
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
