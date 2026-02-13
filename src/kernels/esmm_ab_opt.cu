#pragma once

// K30: Optimized Fused A+B Sparsity
//
// Improvements over K25 (esmm_ab_simple_fused):
//
// 1. CORRECTNESS FIX: K25 has a latent __syncthreads() deadlock. Each warp
//    independently evaluates `if (joint == 0) continue`, but tile loading
//    requires ALL threads to participate. If warp 0 skips (joint==0) while
//    warp 1 doesn't, warp 1 hits __syncthreads() while warp 0 has already
//    advanced to the next iteration → deadlock. This doesn't manifest with
//    uniform test patterns (--blockwise --pattern 11110000) but WILL break
//    with non-uniform real-world sparsity.
//
//    Fix: Precompute a block-level OR of all warp joints. Tile-level skip
//    uses block_joint (all threads see the same value → safe). Per-warp
//    fine-grained skip still uses individual warp patterns.
//
// 2. SHARED MEMORY: Reduced from ~16KB to ~10KB.
//    K25: As[4128] + Bs[4096] + joint_smem[8192] = 16,416 bytes
//    K30: As[4128] + Bs[4096] + block_joint[2048] = 10,272 bytes
//    Saves 6KB → better occupancy potential on A10G (100KB/SM).
//
// 3. TWO-LEVEL SKIP ARCHITECTURE:
//    Level 1 (tile): block_joint[kBlock] == 0 → skip entire tile load.
//                    Safe because all threads read the same shared memory value.
//    Level 2 (dotIdx): per-warp pattern from global memory (L1 cached) → skip
//                      individual BK rows. Warp-uniform via __shfl_sync.
//
// Preprocessors: Identical to K25 (proven correct).

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ============================================================================
// Preprocessor A: Coalesced row-scanning with ballot reduction
// Unchanged from K25. Produces one uint8_t per (TILE_M-row-tile, BK-column-block).
// Each bit indicates whether ANY row in the tile has a non-zero in that BK column.
// ============================================================================
template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_opt(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
    constexpr int K_CHUNK = WARP_SIZE;
    constexpr int BK_BLOCKS_PER_CHUNK = K_CHUNK / BK;

    const int numTiles = M / TILE_M;
    const int numKBlocks = K / BK;
    const int numKChunks = K / K_CHUNK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tileIdx = blockIdx.x;

    const int globalMBase = tileIdx * TILE_M;

    for (int kChunk = warpId; kChunk < numKChunks; kChunk += WARPS_PER_BLOCK) {
        const int globalKChunkBase = kChunk * K_CHUNK;

        uint32_t myBit = 0;

        #pragma unroll 4
        for (int mRow = 0; mRow < TILE_M; mRow++) {
            const int globalM = globalMBase + mRow;
            float val = A[globalM * K + globalKChunkBase + laneId];
            if (val != 0.0f) {
                myBit = 1;
            }
        }

        uint32_t ballot = __ballot_sync(0xFFFFFFFF, myBit);

        #pragma unroll
        for (int bkOffset = 0; bkOffset < BK_BLOCKS_PER_CHUNK; bkOffset++) {
            const int kBlock = kChunk * BK_BLOCKS_PER_CHUNK + bkOffset;
            uint8_t pattern = (ballot >> (bkOffset * BK)) & 0xFF;

            if (laneId == bkOffset) {
                patterns[tileIdx * numKBlocks + kBlock] = pattern;
            }
        }
    }
}

// ============================================================================
// Preprocessor B: Vectorized float4 scanning with warp OR reduction
// Unchanged from K25. Produces one uint8_t per (WN-column-block, BK-row-block).
// Each bit indicates whether ANY column in the WN block has a non-zero in that
// BK row.
// ============================================================================
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_opt(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;
    constexpr int FLOAT4_PER_ROW = WN / 4;
    constexpr int TOTAL_FLOAT4 = BK * FLOAT4_PER_ROW;
    constexpr int FLOAT4_PER_THREAD = TOTAL_FLOAT4 / WARP_SIZE;

    const int numNBlocks = N / WN;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int nBlock = blockIdx.x;
    if (nBlock >= numNBlocks) return;

    const int globalNBase = nBlock * WN;

    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            const int float4Idx = laneId + i * WARP_SIZE;
            const int kRow = float4Idx / FLOAT4_PER_ROW;
            const int nBase = (float4Idx % FLOAT4_PER_ROW) * 4;

            const int globalK = globalKBase + kRow;
            const int globalN = globalNBase + nBase;

            float4 vals = *reinterpret_cast<const float4*>(&B[globalK * N + globalN]);
            if (vals.x != 0.0f || vals.y != 0.0f || vals.z != 0.0f || vals.w != 0.0f) {
                threadPattern |= (1 << kRow);
            }
        }

        // Warp-level OR reduction (SM 80+ / Ampere+)
        threadPattern = __reduce_or_sync(0xFFFFFFFF, threadPattern);

        if (laneId == 0) {
            patterns[nBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// ============================================================================
// Compute kernel: Optimized with two-level skip and reduced shared memory
// ============================================================================
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_compute_opt(
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

    constexpr uint NUM_WARPS_M = BM / WM;
    constexpr uint NUM_WARPS_N = BN / WN;
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint threadColInWarp = (threadIdx.x % WARPSIZE) % (WSUBN / TN);
    const uint threadRowInWarp = (threadIdx.x % WARPSIZE) / (WSUBN / TN);

    // ------------------------------------------------------------------
    // Shared memory layout (10,272 bytes vs K25's 16,416 bytes)
    // ------------------------------------------------------------------
    __shared__ float As[BK * (BM + 1)];   // 4,128 bytes (BM+1 padding avoids bank conflicts)
    __shared__ float Bs[BK * BN];          // 4,096 bytes
    // Block-level tile-skip pattern. Supports K up to 16384 with BK=8.
    // Only 2,048 bytes vs K25's 8,192 bytes for joint_smem.
    constexpr int MAX_K_BLOCKS = 2048;
    __shared__ uint8_t block_joint[MAX_K_BLOCKS];

    // ------------------------------------------------------------------
    // Phase 1: Precompute block-level joint patterns
    //
    // block_joint[k] = OR over all warps of (a_pat[warp_row][k] & b_pat[warp_col][k])
    //
    // If block_joint[k] == 0, NO warp in this block needs tile k.
    // ALL threads will read the same value → tile-level skip is safe.
    //
    // Cost: NUM_WARPS_M * NUM_WARPS_N * 2 global reads per k-block per thread.
    // These are single-byte reads from small arrays (numKBlocks bytes each),
    // so they hit L1/L2 cache after the first thread in each warp reads them.
    // ------------------------------------------------------------------
    {
        for (int k = threadIdx.x; k < numKBlocks; k += NUM_THREADS) {
            uint8_t block_pat = 0;
            #pragma unroll
            for (int wr = 0; wr < (int)NUM_WARPS_M; wr++) {
                const int gRow = cRow * NUM_WARPS_M + wr;
                #pragma unroll
                for (int wc = 0; wc < (int)NUM_WARPS_N; wc++) {
                    const int gCol = cCol * NUM_WARPS_N + wc;
                    const uint8_t a_pat = a_patterns[gRow * numKBlocks + k];
                    const uint8_t b_pat = b_patterns[gCol * numKBlocks + k];
                    block_pat |= (a_pat & b_pat);
                }
            }
            block_joint[k] = block_pat;
        }
    }

    // Per-warp pattern pointers for Level 2 (inner-loop) skip.
    // These stay in global memory; lane 0 reads 2 bytes per tile, broadcasts
    // via __shfl_sync. Sequential access pattern → excellent L1 hit rate.
    const int gWarpRow = cRow * NUM_WARPS_M + warpRow;
    const int gWarpCol = cCol * NUM_WARPS_N + warpCol;
    const uint8_t* my_a_pats = a_patterns + gWarpRow * numKBlocks;
    const uint8_t* my_b_pats = b_patterns + gWarpCol * numKBlocks;

    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: Main GEMM loop with two-level skipping
    // ------------------------------------------------------------------
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

        // ==============================================================
        // Level 1: Block-level tile skip
        // All threads read block_joint from shared memory → uniform decision.
        // Safe with __syncthreads() because every thread takes the same branch.
        // ==============================================================
        if (block_joint[kBlock] == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // ---- Load A tile to shared memory (synchronous, with transpose) ----
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }

        // ---- Load B tile to shared memory (direct copy, row-major) ----
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // ==============================================================
        // Level 2: Per-warp inner-loop skip
        // Lane 0 reads this warp's A and B patterns from global memory,
        // computes the joint, and broadcasts to all lanes via shfl.
        // Cost: 2 global reads (L1 cached) + 1 shfl per tile per warp.
        // ==============================================================
        uint8_t my_joint;
        if (laneId == 0) {
            my_joint = my_a_pats[kBlock] & my_b_pats[kBlock];
        }
        my_joint = __shfl_sync(0xFFFFFFFF, my_joint, 0);

        // If this warp's joint is entirely zero (but another warp's wasn't,
        // which is why we still loaded the tile), skip compute entirely.
        if (my_joint != 0) {
            float regM[WMITER];
            float regN[WNITER * TN];

            #pragma unroll
            for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
                if (!(my_joint & (1 << dotIdx))) {
                    continue;
                }

                #pragma unroll
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    regM[wSubRowIdx] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                        wSubRowIdx * WSUBM + threadRowInWarp];
                }

                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float4 tmp0 = reinterpret_cast<const float4*>(
                        &Bs[(dotIdx * BN) + warpCol * WN +
                        wSubColIdx * WSUBN + threadColInWarp * TN])[0];
                    const float4 tmp1 = reinterpret_cast<const float4*>(
                        &Bs[(dotIdx * BN) + warpCol * WN +
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

                // Outer product accumulation (float4 vectorized)
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

                        acc_0.x += valM * reg_0.x;
                        acc_0.y += valM * reg_0.y;
                        acc_0.z += valM * reg_0.z;
                        acc_0.w += valM * reg_0.w;

                        acc_1.x += valM * reg_1.x;
                        acc_1.y += valM * reg_1.y;
                        acc_1.z += valM * reg_1.z;
                        acc_1.w += valM * reg_1.w;

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
                const int i = wSubRowIdx * (WNITER * TN) +
                    wSubColIdx * TN + resIdxN;
                tmp.x = threadResults[i + 0];
                tmp.y = threadResults[i + 1];
                tmp.z = threadResults[i + 2];
                tmp.w = threadResults[i + 3];
                reinterpret_cast<float4*>(
                    &C_sub[threadRowInWarp * N +
                    threadColInWarp * TN + resIdxN])[0] = tmp;
            }
        }
    }
}
