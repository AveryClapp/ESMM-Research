#pragma once

/*
 * ESMM AB-Sparse V2: Drop-in Replacement with All Optimizations
 * 
 * OPTIMIZATIONS:
 * 1. Precomputes joint patterns (A & B) in shared memory at kernel start
 * 2. Uses warp shuffle broadcast to eliminate bank conflicts
 * 3. Inlines LUT computation to avoid constant memory latency
 *
 * INTEGRATION: Just replace esmm_ab_sparse with esmm_ab_sparse_v2
 */

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
    esmm_ab_sparse_v2(
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
        __shared__ float As[BK * BM];
        __shared__ float Bs[BK * BN];
        __shared__ uint8_t joint_smem[NUM_WARPS * 1024];

        // ========================================================================
        // OPTIMIZATION 1: Precompute ALL joint patterns at kernel start
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
        // K-LOOP with all optimizations
        // ========================================================================
        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            const uint kBlock = bkIdx / BK;

            // OPTIMIZATION 2: Warp shuffle broadcast
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

            // Load A tile
#pragma unroll
            for (uint offset = 0; offset < BM; offset += rowStrideA) {
                float4 tmp = reinterpret_cast<const float4*>(
                        &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }

            // Load B tile
#pragma unroll
            for (uint offset = 0; offset < BK; offset += rowStrideB) {
                reinterpret_cast<float4*>(
                        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4*>(
                            &B[(innerRowB + offset) * N + innerColB * 4])[0];
            }

            __syncthreads();

            // OPTIMIZATION 3: Inline LUT computation
            uint8_t count = __popc(joint);
            uint8_t offsets[8];
            uint8_t cnt = 0;
#pragma unroll
            for (int bit = 0; bit < 8; bit++) {
                if (joint & (1 << bit)) {
                    offsets[cnt++] = bit;
                }
            }

            // Sparse FMA computation
            float regM[WMITER * TM];
            float regN[WNITER * TN];

#pragma unroll
            for (int sparse_idx = 0; sparse_idx < 8; ++sparse_idx) {
                if (sparse_idx >= count) break;

                const uint8_t dotIdx = offsets[sparse_idx];

#pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                    for (uint i = 0; i < TM; ++i) {
                        regM[wSubRowIdx * TM + i] = As[dotIdx * BM + warpRow * WM +
                            wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                    }
                }

#pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
                    for (uint j = 0; j < TN; ++j) {
                        regN[wSubColIdx * TN + j] = Bs[dotIdx * BN + warpCol * WN +
                            wSubColIdx * WSUBN + threadColInWarp * TN + j];
                    }
                }

#pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
                        for (uint i = 0; i < TM; ++i) {
#pragma unroll
                            for (uint j = 0; j < TN; ++j) {
                                threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) +
                                    wSubColIdx * TN + j] +=
                                    regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
                            }
                        }
                    }
                }
            }

            __syncthreads();
            A += BK;
            B += BK * N;
        }

        // Write results
#pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                float* C_sub = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
#pragma unroll
                for (uint i = 0; i < TM; ++i) {
#pragma unroll
                    for (uint j = 0; j < TN; ++j) {
                        C_sub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j] = 
                            threadResults[(wSubRowIdx * TM + i) * (WNITER * TN) + 
                            wSubColIdx * TN + j];
                    }
                }
            }
        }
    }

