#pragma once

// K25-27: Joint skip experiments (baseline, skip-only, skip+FMA)
// Measure overhead of skip logic with identical preprocessing

#include <cuda_runtime.h>
#include <cstdint>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

// ============================================================================
// SHARED PREPROCESSOR (used by ALL variants)
// ============================================================================

__global__ void preprocess_joint_skip_kernel(
    const uint8_t* __restrict__ a_patterns,
    const uint8_t* __restrict__ b_patterns,
    uint8_t* __restrict__ joint_patterns,
    int numMWarps, int numNWarps, int numKBlocks
) {
    const int mWarp = blockIdx.y;
    const int nWarp = blockIdx.x;
    const int kBase = blockIdx.z * blockDim.x + threadIdx.x;

    if (mWarp >= numMWarps || nWarp >= numNWarps || kBase >= numKBlocks) return;

    const uint8_t a_pat = a_patterns[mWarp * numKBlocks + kBase];
    const uint8_t b_pat = b_patterns[nWarp * numKBlocks + kBase];
    const uint8_t joint = a_pat & b_pat;

    const int outIdx = mWarp * numNWarps * numKBlocks + nWarp * numKBlocks + kBase;
    joint_patterns[outIdx] = joint;
}

// ============================================================================
// K25: BASELINE - Dense computation (no joint sparsity)
// ============================================================================

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_joint_baseline(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

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

    // DENSE K-LOOP: No skipping, always compute
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
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

        // DENSE FMA: All 8 iterations
        float regM[WMITER * TM];
        float regN[WNITER * TN];

        #pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
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

// ============================================================================
// K26: SKIP_ONLY - Check patterns and skip, but NO FMA
// Measures pure overhead of skip logic
// ============================================================================

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_joint_skip_only(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ joint_patterns,
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
    constexpr uint NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;

    __shared__ uint8_t joint_smem[NUM_WARPS * 1024];

    // Load precomputed joint patterns
    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;
        const int totalPatterns = NUM_WARPS * numKBlocks;
        const int numNWarps = (gridDim.x * NUM_WARPS_N);

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / numKBlocks;
            const int kBlock = i % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            const int gWarpRow = globalMBlock * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            const int jointIdx = gWarpRow * numNWarps * numKBlocks + gWarpCol * numKBlocks + kBlock;
            joint_smem[i] = joint_patterns[jointIdx];
        }
        __syncthreads();
    }

    const int myWarpId = warpRow * NUM_WARPS_N + warpCol;
    const uint8_t* my_joints = joint_smem + myWarpId * numKBlocks;

    A += cRow * BM * K;
    B += cCol * BN;

    // K-LOOP: Check patterns, skip, but NO FMA
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // Warp shuffle broadcast
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

        // NO FMA COMPUTATION - just skip logic overhead
        // This measures: shuffle cost + branch cost + pointer arithmetic

        A += BK;
        B += BK * N;
    }

    // Write zeros to C (no computation was done)
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_sub = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                #pragma unroll
                for (uint j = 0; j < TN; ++j) {
                    C_sub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j] = 0.0f;
                }
            }
        }
    }
}

// ============================================================================
// K27: SKIP_FMA - Full joint sparse (check + skip + FMA)
// This is the REAL implementation we care about
// ============================================================================

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_joint_skip_fma(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ joint_patterns,
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

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    __shared__ uint8_t joint_smem[NUM_WARPS * 1024];

    // Load precomputed joint patterns
    {
        const uint globalMBlock = cRow;
        const uint globalNBlock = cCol;
        const int totalPatterns = NUM_WARPS * numKBlocks;
        const int numNWarps = (gridDim.x * NUM_WARPS_N);

        for (int i = threadIdx.x; i < totalPatterns; i += NUM_THREADS) {
            const int localWarpId = i / numKBlocks;
            const int kBlock = i % numKBlocks;

            const int localWarpRow = localWarpId / NUM_WARPS_N;
            const int localWarpCol = localWarpId % NUM_WARPS_N;

            const int gWarpRow = globalMBlock * NUM_WARPS_M + localWarpRow;
            const int gWarpCol = globalNBlock * NUM_WARPS_N + localWarpCol;

            const int jointIdx = gWarpRow * numNWarps * numKBlocks + gWarpCol * numKBlocks + kBlock;
            joint_smem[i] = joint_patterns[jointIdx];
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

    // K-LOOP: Check pattern, skip if zero, compute sparse FMA if non-zero
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

        // Load tiles
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        #pragma unroll
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }

        __syncthreads();

        // Compute sparse FMA
        uint8_t count = __popc(joint);
        uint8_t offsets[8];
        uint8_t cnt = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            if (joint & (1 << bit)) offsets[cnt++] = bit;
        }

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
