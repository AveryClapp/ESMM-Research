#pragma once

// K26: Joint A+B sparsity, 8x32 tile granularity with double buffering
// Based on K21 (esmm_ab_8x32.cu) with:
//   - Double-buffered As and Bs shared memory
//   - cp.async for B loads (direct float4 copy)
//   - Synchronous A loads (register-based transpose)
//   - Pipeline: load(k+1) overlaps with compute(k) for B data

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
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
esmm_ab_8x32_db(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ a_patterns,  // [numTileRows x numKBlocks]
    const uint8_t* __restrict__ b_patterns,  // [numWarpCols x numKBlocks]
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

    // Double-buffered shared memory
    __shared__ float As[2][BK * (BM + 1)];  // Padded to avoid bank conflicts
    __shared__ float Bs[2][BK * BN];
    __shared__ uint8_t joint_smem[NUM_WARPS * WMITER * 1024];

    // ========================================================================
    // Precompute joint patterns (same as K21)
    // ========================================================================
    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;
        const int totalPatterns = NUM_WARPS * WMITER * numKBlocks;

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / (WMITER * numKBlocks);
            const int remainder = i % (WMITER * numKBlocks);
            const int wSubRowIdx = remainder / numKBlocks;
            const int kBlock = remainder % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            const int gWarpRow = globalMBlock * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            const int tileRow = gWarpRow * WMITER + wSubRowIdx;
            const uint8_t a_pat = a_patterns[tileRow * numKBlocks + kBlock];
            const uint8_t b_pat = b_patterns[gWarpCol * numKBlocks + kBlock];

            joint_smem[i] = a_pat & b_pat;
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * (WMITER * numKBlocks);

    // Standard GEMM setup
    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // Loading index computation (same as K21)
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    const uint totalKBlocks = K / BK;

    // ========================================================================
    // PROLOGUE: Load first tile into buffer 0
    // ========================================================================
    {
        // Load A tile 0 synchronously (with transpose)
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[0][(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
                As[0][(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
                As[0][(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
                As[0][(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
            }
        }

        // Load B tile 0 with cp.async (direct copy, no transpose)
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                __pipeline_memcpy_async(
                    &Bs[0][(innerRowB + offset) * BN + innerColB * 4],
                    &B[(innerRowB + offset) * N + innerColB * 4],
                    16  // sizeof(float4)
                );
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ========================================================================
    // MAIN LOOP: Double-buffered pipeline
    // ========================================================================
    for (uint kBlock = 0; kBlock < totalKBlocks - 1; kBlock++) {
        const int cur = kBlock % 2;
        const int next = 1 - cur;

        // ---- Load joint patterns for current tile ----
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

        // ---- Load NEXT tile: A synchronous, B async ----
        // A tile k+1 (synchronous with transpose)
        const float* A_next = A + (kBlock + 1) * BK;
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A_next[(innerRowA + offset) * K + innerColA * 4])[0];
                As[next][(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
                As[next][(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
                As[next][(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
                As[next][(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
            }
        }

        // B tile k+1 (async, direct copy)
        const float* B_next = B + (kBlock + 1) * BK * N;
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                __pipeline_memcpy_async(
                    &Bs[next][(innerRowB + offset) * BN + innerColB * 4],
                    &B_next[(innerRowB + offset) * N + innerColB * 4],
                    16
                );
            }
        }
        __pipeline_commit();

        // ---- Compute CURRENT tile from buf[cur] ----
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const uint8_t joint = joints[wSubRowIdx];

            if (joint == 0) continue;

            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    if (!(joint & (1 << dotIdx))) {
                        continue;
                    }

                    float a_val = As[cur][(dotIdx * (BM + 1)) + warpRow * WM +
                                     wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];

                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        float b_val = Bs[cur][(dotIdx * BN) + warpCol * WN +
                                         wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                        threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) +
                                      wSubColIdx * TN + tn] += a_val * b_val;
                    }
                }
            }
        }

        // ---- Wait for next tile loads and synchronize ----
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // ========================================================================
    // EPILOGUE: Compute last tile
    // ========================================================================
    {
        const uint lastKBlock = totalKBlocks - 1;
        const int lastBuf = lastKBlock % 2;

        uint8_t joints[WMITER];
        if (laneId == 0) {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                joints[m] = my_joints[m * numKBlocks + lastKBlock];
            }
        }
        #pragma unroll
        for (int m = 0; m < WMITER; ++m) {
            joints[m] = __shfl_sync(0xFFFFFFFF, joints[m], 0);
        }

        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const uint8_t joint = joints[wSubRowIdx];

            if (joint == 0) continue;

            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    if (!(joint & (1 << dotIdx))) {
                        continue;
                    }

                    float a_val = As[lastBuf][(dotIdx * (BM + 1)) + warpRow * WM +
                                     wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];

                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        float b_val = Bs[lastBuf][(dotIdx * BN) + warpCol * WN +
                                         wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                        threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) +
                                      wSubColIdx * TN + tn] += a_val * b_val;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Write results back to C (same as K21)
    // ========================================================================
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
