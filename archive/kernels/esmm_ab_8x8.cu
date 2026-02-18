#pragma once

// K30: 8×8 A-granularity variant (SpInfer-style)
// A-matrix: 8-row tiles (finest granularity)
// B-matrix: 8×32 (unchanged from K25)

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// A preprocessor with 8-row granularity
template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_8x8(
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
    if (tileIdx >= numTiles) return;

    const int globalMBase = tileIdx * TILE_M;

    for (int kChunk = warpId; kChunk < numKChunks; kChunk += WARPS_PER_BLOCK) {
        const int globalKChunkBase = kChunk * K_CHUNK;

        uint32_t myBit = 0;

        // Process TILE_M=8 rows
        #pragma unroll
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

// B preprocessor (unchanged from K25)
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_8x8(
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

        threadPattern = __reduce_or_sync(0xFFFFFFFF, threadPattern);

        if (laneId == 0) {
            patterns[nBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// Compute kernel with 8-row A-pattern granularity
// Each warp processes 64 rows but uses 8 different A-patterns (one per 8-row subtile)
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS,
          const int A_TILE_M>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_8x8_compute(
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

    // Number of A-pattern tiles per warp (WM / A_TILE_M = 64/8 = 8)
    constexpr uint A_TILES_PER_WARP = WM / A_TILE_M;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BK * (BM + 1)];
    __shared__ float Bs[BK * BN];

    // Joint patterns: one per A-tile per warp per K-block
    // For 8×8: 8 A-tiles per warp
    __shared__ uint8_t joint_smem[NUM_WARPS * A_TILES_PER_WARP * 1024];

    // Precompute joint patterns with 8-row A granularity
    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;

        const int totalPatterns = NUM_WARPS * A_TILES_PER_WARP * numKBlocks;

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / (A_TILES_PER_WARP * numKBlocks);
            const int rem = i % (A_TILES_PER_WARP * numKBlocks);
            const int aTileInWarp = rem / numKBlocks;
            const int kBlock = rem % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            // Global A-tile index (8-row granularity)
            const int gWarpMBase = globalMBlock * BM + localWarpRow * WM;
            const int gATileIdx = (gWarpMBase + aTileInWarp * A_TILE_M) / A_TILE_M;

            // Global B-tile index (unchanged - still WN granularity)
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            const uint8_t a_pat = a_patterns[gATileIdx * numKBlocks + kBlock];
            const uint8_t b_pat = b_patterns[gWarpCol * numKBlocks + kBlock];

            joint_smem[localWarpId * A_TILES_PER_WARP * numKBlocks + aTileInWarp * numKBlocks + kBlock] = a_pat & b_pat;
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * A_TILES_PER_WARP * numKBlocks;

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

        // Check if ANY A-tile has work (quick skip for entire K-block)
        uint8_t anyWork = 0;
        #pragma unroll
        for (int aTile = 0; aTile < A_TILES_PER_WARP; aTile++) {
            anyWork |= my_joints[aTile * numKBlocks + kBlock];
        }

        if (laneId == 0) {
            anyWork = anyWork;
        }
        anyWork = __shfl_sync(0xFFFFFFFF, anyWork, 0);

        if (anyWork == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load A and B tiles
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

        // Load patterns for this K-block (one per A-tile)
        uint8_t joints[A_TILES_PER_WARP];
        #pragma unroll
        for (int aTile = 0; aTile < A_TILES_PER_WARP; aTile++) {
            if (laneId == 0) {
                joints[aTile] = my_joints[aTile * numKBlocks + kBlock];
            }
            joints[aTile] = __shfl_sync(0xFFFFFFFF, joints[aTile], 0);
        }

        float regM[WMITER * TM];
        float regN[WNITER * TN];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load B registers once (same for all rows)
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

            // Keep original WMITER loop structure, but check per-thread A-tile pattern
            // Each thread knows which row it handles: wSubRowIdx * WSUBM + threadRowInWarp * TM
            // Map that row to its 8-row A-tile
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                // Which 8-row A-tile does this thread's row belong to?
                const int rowInWarp = wSubRowIdx * WSUBM + threadRowInWarp * TM;
                const int myATile = rowInWarp / A_TILE_M;

                if (!(joints[myATile] & (1 << dotIdx))) {
                    continue;
                }

                regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];

                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float valM = regM[wSubRowIdx];
                    const uint resBase = wSubRowIdx * (WNITER * TN) + (wSubColIdx * TN);
                    const uint nBase = wSubColIdx * TN;
                    #pragma unroll
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resBase + resIdxN] += valM * regN[nBase + resIdxN];
                    }
                }
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write results (unchanged from K25)
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
