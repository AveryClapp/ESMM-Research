#pragma once

// K24: Fused A+B sparsity with persistent pattern extraction
// Combines preprocessing + computation in one kernel
// Key optimization: Extract patterns to SMEM once, then skip loads when joint==0

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
esmm_ab_fused(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    // Block and warp coordinates
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

    const int numKBlocks = K / BK;
    const int numTileRows = M / 8;  // 8-row granularity for A
    const int numWarpCols = N / WN;  // WN-column granularity for B

    // Shared memory for computation
    __shared__ float As[BK * (BM + 1)];
    __shared__ float Bs[BK * BN];

    // Shared memory for patterns
    // A patterns: One per 8-row tile (BM/8 tiles per block)
    // B patterns: One per WN-column warp (BN/WN warps per block)
    constexpr int A_TILES_PER_BLOCK = BM / 8;
    constexpr int B_WARPS_PER_BLOCK = BN / WN;

    __shared__ uint8_t a_patterns_smem[A_TILES_PER_BLOCK * 512];  // Max 512 K-blocks
    __shared__ uint8_t b_patterns_smem[B_WARPS_PER_BLOCK * 512];
    __shared__ uint8_t joint_smem[NUM_WARPS * WMITER * 512];

    // ========================================================================
    // PHASE 1: PATTERN EXTRACTION
    // ========================================================================
    {
        const int globalMBase = cRow * BM;
        const int globalNBase = cCol * BN;

        // Extract A patterns: Each 8-row tile scans its rows across all K
        for (int tileIdx = threadIdx.x / WARPSIZE; tileIdx < A_TILES_PER_BLOCK; tileIdx += NUM_WARPS) {
            const int tileLaneId = threadIdx.x % WARPSIZE;
            const int tileRowBase = globalMBase + tileIdx * 8;

            for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
                const int kBase = kBlock * BK;
                uint8_t threadPattern = 0;

                // Each thread scans 2 elements (8 rows × 8 cols / 32 threads = 2 elements)
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int flatIdx = tileLaneId * 2 + i;
                    const int mRow = flatIdx / BK;  // 0-7
                    const int kCol = flatIdx % BK;  // 0-7

                    const int globalM = tileRowBase + mRow;
                    const int globalK = kBase + kCol;

                    if (globalM < M && globalK < K) {
                        float val = A[globalM * K + globalK];
                        if (val != 0.0f) {
                            threadPattern |= (1 << kCol);
                        }
                    }
                }

                // Warp reduction (OR across all lanes)
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
                }

                if (tileLaneId == 0) {
                    a_patterns_smem[tileIdx * numKBlocks + kBlock] = threadPattern;
                }
            }
        }

        // Extract B patterns: Each WN-column warp scans its columns across all K
        for (int warpColIdx = threadIdx.x / WARPSIZE; warpColIdx < B_WARPS_PER_BLOCK; warpColIdx += NUM_WARPS) {
            const int warpLaneId = threadIdx.x % WARPSIZE;
            const int warpNBase = globalNBase + warpColIdx * WN;

            for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
                const int kBase = kBlock * BK;
                uint8_t threadPattern = 0;

                // Each thread scans 8 elements (8 rows × 32 cols / 32 threads = 8 elements)
                constexpr int ELEMENTS_PER_THREAD = (BK * WN) / WARPSIZE;
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                    const int flatIdx = warpLaneId * ELEMENTS_PER_THREAD + i;
                    const int kRow = flatIdx / WN;  // 0-7
                    const int nCol = flatIdx % WN;  // 0-31

                    const int globalK = kBase + kRow;
                    const int globalN = warpNBase + nCol;

                    if (globalK < K && globalN < N) {
                        float val = B[globalK * N + globalN];
                        if (val != 0.0f) {
                            threadPattern |= (1 << kRow);
                        }
                    }
                }

                // Warp reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
                }

                if (warpLaneId == 0) {
                    b_patterns_smem[warpColIdx * numKBlocks + kBlock] = threadPattern;
                }
            }
        }

        __syncthreads();

        // Compute joint patterns (A & B) for each warp's sub-tiles
        for (int i = threadIdx.x; i < NUM_WARPS * WMITER * numKBlocks; i += NUM_THREADS) {
            const int localWarpId = i / (WMITER * numKBlocks);
            const int remainder = i % (WMITER * numKBlocks);
            const int wSubRowIdx = remainder / numKBlocks;
            const int kBlock = remainder % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            // A pattern: Local tile index within this block
            const int aTileIdx = localWarpRow * WMITER + wSubRowIdx;
            const uint8_t a_pat = a_patterns_smem[aTileIdx * numKBlocks + kBlock];

            // B pattern: Local warp column within this block
            const uint8_t b_pat = b_patterns_smem[localWarpCol * numKBlocks + kBlock];

            joint_smem[i] = a_pat & b_pat;
        }

        __syncthreads();
    }

    // ========================================================================
    // PHASE 2: COMPUTATION WITH LOAD SKIPPING
    // ========================================================================

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * (WMITER * numKBlocks);

    // Set up global memory pointers
    const float* A_block = A + cRow * BM * K;
    const float* B_block = B + cCol * BN;
    float* C_warp = C + (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS * 4) / BN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Shared memory to coordinate block-wide skipping decision
    __shared__ bool blockNeedsKBlock[512];  // One per K-block

    // K-loop with load skipping
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Load all joint patterns for this K-block (per-warp)
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

        // Check if THIS WARP needs this K-block
        bool warpNeedsBlock = false;
        #pragma unroll
        for (int m = 0; m < WMITER; ++m) {
            if (joints[m] != 0) {
                warpNeedsBlock = true;
                break;
            }
        }

        // Cooperative decision: Block needs K-block if ANY warp needs it
        if (threadIdx.x < NUM_WARPS) {
            blockNeedsKBlock[threadIdx.x] = false;
        }
        __syncthreads();

        if (laneId == 0 && warpNeedsBlock) {
            blockNeedsKBlock[myWarpId] = true;
        }
        __syncthreads();

        bool anyWarpNeeds = false;
        for (int w = 0; w < NUM_WARPS && !anyWarpNeeds; w++) {
            if (blockNeedsKBlock[w]) {
                anyWarpNeeds = true;
            }
        }

        // Skip memory loads only if NO warp needs this K-block
        if (!anyWarpNeeds) {
            continue;
        }

        // Load A tile
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A_block[(innerRowA + offset) * K + bkIdx + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
            }
        }

        // Load B tile
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                reinterpret_cast<float4*>(
                    &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4*>(
                        &B_block[(bkIdx + innerRowB + offset) * N + innerColB * 4])[0];
            }
        }

        __syncthreads();

        // Inner loop: Fine-grained computation with per-sub-tile patterns
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const uint8_t joint = joints[wSubRowIdx];

            if (joint == 0) continue;  // Skip this 8-row sub-tile

            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    if (!(joint & (1 << dotIdx))) {
                        continue;  // Skip this K-element
                    }

                    // Load A value
                    float a_val = As[(dotIdx * (BM + 1)) + warpRow * WM +
                                     wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];

                    // Compute with B values
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        float b_val = Bs[(dotIdx * BN) + warpCol * WN +
                                        wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                        threadResults[(wSubRowIdx * TM + 0) * (WNITER * TN) +
                                     wSubColIdx * TN + tn] += a_val * b_val;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results back to C
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C_warp + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
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
