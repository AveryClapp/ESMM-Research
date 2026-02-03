#pragma once

// K25: Simple Fused - Literally copy-paste preprocessor + K20 computation
// Just runs preprocessing once at start, stores in global, then computes

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// Copy-pasted from ab_preprocessor.cu
template <const int BK, const int TILE_M, const int NUM_THREADS>
__global__ void preprocess_a_inline(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

    const int numTiles = M / TILE_M;
    const int numKBlocks = K / BK;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    const int globalMBase = tileIdx * TILE_M;

    for (int kBlock = warpId; kBlock < numKBlocks; kBlock += WARPS_PER_BLOCK) {
        const int globalKBase = kBlock * BK;

        uint8_t threadPattern = 0;

        constexpr int ELEMENTS_PER_THREAD = (TILE_M * BK) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int flatIdx = laneId * ELEMENTS_PER_THREAD + i;
            const int mRow = flatIdx / BK;
            const int kCol = flatIdx % BK;

            const int globalM = globalMBase + mRow;
            const int globalK = globalKBase + kCol;

            if (globalM < M && globalK < K) {
                float val = A[globalM * K + globalK];
                if (val != 0.0f) {
                    threadPattern |= (1 << kCol);
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            threadPattern |= __shfl_xor_sync(0xFFFFFFFF, threadPattern, offset);
        }

        if (laneId == 0) {
            patterns[tileIdx * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// Copy-pasted from ab_preprocessor.cu
template <const int BK, const int WN, const int NUM_THREADS>
__global__ void preprocess_b_inline(
    int K, int N,
    const float* __restrict__ B,
    uint8_t* __restrict__ patterns
) {
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = NUM_THREADS / WARP_SIZE;

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

        constexpr int ELEMENTS_PER_THREAD = (BK * WN) / WARP_SIZE;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int baseFlatIdx = (i * 4 * WARP_SIZE) + (laneId * 4);

            const int kRowBase = baseFlatIdx / WN;
            const int nColBase = baseFlatIdx % WN;

            const float4* ptrB = reinterpret_cast<const float4*>(&B[globalKBase * N + globalNBase]);
            float4 vals = ptrB[baseFlatIdx / 4];

            auto process = [&](float v, int offset) {
                if (v != 0.0f) {
                    int currentFlatIdx = baseFlatIdx + offset;
                    int kRow = currentFlatIdx / WN;
                    threadPattern |= (1 << kRow);
                }
            };

            process(vals.x, 0);
            process(vals.y, 1);
            process(vals.z, 2);
            process(vals.w, 3);
        }

        threadPattern = __reduce_or_sync(0xFFFFFFFF, threadPattern);

        if (laneId == 0) {
            patterns[nBlock * numKBlocks + kBlock] = threadPattern;
        }
    }
}

// Copy-pasted from esmm_ab_sparse_optimized.cu (K20)
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_ab_compute_inline(
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
    __shared__ uint8_t joint_smem[NUM_WARPS * 1024];

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

        float regM[WMITER * TM];
        float regN[WNITER * TN];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Every thread executes this together
            if (!(joint & (1 << dotIdx))) {
                continue;
            }

            #pragma unroll WMITER
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
                const float valM = regM[wSubRowIdx];
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
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
