#pragma once

// K27: Fused Pipeline - A-mask computed on-the-fly from shared memory
// Based on K26 (double-buffered) with:
//   - NO separate A preprocessing pass (saves ~264us at 4096x4096)
//   - A-mask extracted from As[cur] shared memory each K-iteration
//   - Cached B patterns (computed once, reused across batches)
//   - Double-buffered As/Bs, cp.async for B loads
//   - Eliminates ~32KB joint_smem (total smem ~12KB vs ~44KB in K26)

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// Compute 8-bit A-mask from an 8×8 sub-tile in transposed shared memory.
// Each warp (32 lanes) covers 64 elements (8 rows × 8 cols), 2 per lane.
// Returns pattern where bit k is set if row k has any non-zero in the 8×8 tile.
__device__ __forceinline__ uint8_t compute_a_mask_from_smem(
    const float* As_buf, int mBase, int BM_PLUS_1)
{
    const int laneId = threadIdx.x % WARPSIZE;
    uint8_t threadPattern = 0;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int flatIdx = laneId * 2 + i;
        const int mRow = flatIdx / 8;   // 0-7
        const int kCol = flatIdx % 8;   // 0-7

        // As is stored transposed: As[kCol * (BM+1) + mRow]
        float val = As_buf[kCol * BM_PLUS_1 + mBase + mRow];
        if (val != 0.0f) {
            threadPattern |= (1 << kCol);
        }
    }

    // Warp OR-reduce: 5 rounds for 32 lanes
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
    }

    return threadPattern;
}

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_fused_pipeline(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ b_patterns,  // [numWarpCols x numKBlocks] (cached)
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
    constexpr uint NUM_WARPS_N = BN / WN;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // Double-buffered shared memory (NO joint_smem needed!)
    __shared__ float As[2][BK * (BM + 1)];  // Padded to avoid bank conflicts
    __shared__ float Bs[2][BK * BN];

    // Global warp column for B-pattern lookup
    const int gWarpCol = cCol * NUM_WARPS_N + warpCol;

    // Standard GEMM setup
    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // Loading index computation
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
    // MAIN LOOP: Double-buffered pipeline with fused A-mask
    // ========================================================================
    for (uint kBlock = 0; kBlock < totalKBlocks - 1; kBlock++) {
        const int cur = kBlock % 2;
        const int next = 1 - cur;

        // ---- Load NEXT tile: A synchronous, B async ----
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

        // ---- Load B-mask for current K-block ----
        uint8_t b_mask;
        if (laneId == 0) {
            b_mask = b_patterns[gWarpCol * numKBlocks + kBlock];
        }
        b_mask = __shfl_sync(0xFFFFFFFF, b_mask, 0);

        // ---- Compute joint masks from As[cur] and cached B-mask ----
        uint8_t joints[WMITER];
        if (b_mask == 0) {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                joints[m] = 0;
            }
        } else {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                const int mBase = warpRow * WM + m * WSUBM;
                uint8_t a_mask = compute_a_mask_from_smem(
                    As[cur], mBase, BM + 1);
                joints[m] = a_mask & b_mask;
            }
        }

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

        // Load B-mask for last K-block
        uint8_t b_mask;
        if (laneId == 0) {
            b_mask = b_patterns[gWarpCol * numKBlocks + lastKBlock];
        }
        b_mask = __shfl_sync(0xFFFFFFFF, b_mask, 0);

        // Compute joint masks from As[lastBuf]
        uint8_t joints[WMITER];
        if (b_mask == 0) {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                joints[m] = 0;
            }
        } else {
            #pragma unroll
            for (int m = 0; m < WMITER; ++m) {
                const int mBase = warpRow * WM + m * WSUBM;
                uint8_t a_mask = compute_a_mask_from_smem(
                    As[lastBuf], mBase, BM + 1);
                joints[m] = a_mask & b_mask;
            }
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
    // Write results back to C
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
