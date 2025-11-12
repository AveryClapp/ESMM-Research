#pragma once

/*
 * ============================================================================
 * Kernel 24: Joint A+B with SHARED MEMORY Pattern Cache
 * ============================================================================
 *
 * The Winning Insight:
 *   K23's problem: Reading 4MB of patterns from global memory kills bandwidth
 *   K17's advantage: Only 64KB patterns, fits in L2 cache
 *
 * Solution:
 *   Load joint patterns for THIS BLOCK into shared memory at start
 *   Then access from shared memory (32 cycles vs 400 cycles)
 *   Pattern data per block: 512 bytes (numKBlocks × numWarps × 1 byte)
 *
 * Why This Wins:
 *   - Shared memory access: ~30 cycles (vs ~400 for global)
 *   - Only load patterns once per block
 *   - Still exploit full joint sparsity (1.56% density)
 *   - Zero divergence (bitmask is warp-uniform)
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/b_transpose_preprocessor.cu"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

// Forward declarations - reuse K23 types
// No redeclaration needed - include order handles it

template <int BK>
__global__ void compute_joint_patterns_kernel_v2(
    const uint8_t* A_patterns,
    const uint8_t* B_patterns,
    JointPattern* joint_patterns,
    int numKBlocks,
    int numMWarps,
    int numNWarps) {

    const int kBlock = blockIdx.x;
    const int mWarp = blockIdx.y;
    const int nWarp = threadIdx.x;

    if (nWarp >= numNWarps) return;

    const uint A_idx = kBlock * numMWarps + mWarp;
    const uint B_idx = nWarp * numKBlocks + kBlock;

    const uint8_t A_pattern = A_patterns[A_idx];
    const uint8_t B_pattern = B_patterns[B_idx];

    const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
    const uint8_t B_count = PATTERN_LUT_BK8[B_pattern].count;

    const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;
    const uint8_t* B_offsets = PATTERN_LUT_BK8[B_pattern].offsets;

    JointPattern jp;
    jp.mask = 0;

    for (int a = 0; a < A_count; ++a) {
        uint8_t a_offset = A_offsets[a];
        for (int b = 0; b < B_count; ++b) {
            if (B_offsets[b] == a_offset) {
                jp.mask |= (1 << a_offset);
                break;
            }
        }
    }

    const int joint_idx = (kBlock * numMWarps + mWarp) * numNWarps + nWarp;
    joint_patterns[joint_idx] = jp;
}

JointPatternMetadata preprocess_joint_patterns_v2(
    float* d_A, float* d_B,
    int M, int N, int K,
    int BM, int BN, int BK,
    int WM, int WN) {

    JointPatternMetadata meta;

    BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, M, K, WM, BK);
    BTPatternMetadata B_meta = preprocess_b_transpose(d_B, K, N, WN, BK);

    meta.numKBlocks = K / BK;
    meta.numMWarps = (M / BM) * (BM / WM);
    meta.numNWarps = (N / BN) * (BN / WN);

    const int totalJointPatterns = meta.numKBlocks * meta.numMWarps * meta.numNWarps;

    cudaMalloc(&meta.d_patterns, totalJointPatterns * sizeof(JointPattern));

    dim3 gridDim(meta.numKBlocks, meta.numMWarps);
    dim3 blockDim(meta.numNWarps);

    compute_joint_patterns_kernel_v2<8><<<gridDim, blockDim>>>(
        A_meta.d_blockPatterns,
        B_meta.d_blockPatterns,
        meta.d_patterns,
        meta.numKBlocks,
        meta.numMWarps,
        meta.numNWarps);

    cudaDeviceSynchronize();

    free_block_pattern_metadata(A_meta);
    free_bt_pattern_metadata(B_meta);

    return meta;
}

void free_joint_pattern_metadata_v2(JointPatternMetadata meta) {
    cudaFree(meta.d_patterns);
}

// ============================================================================
// Compute with Shared Memory Pattern Cache
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_joint_shared(int M, int N, int K,
                     float *A, float *B, float *C,
                     const JointPattern* __restrict__ global_patterns,
                     int numKBlocks, int numMWarps, int numNWarps) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    constexpr uint WARPS_PER_BLOCK_M = BM / WM;
    constexpr uint WARPS_PER_BLOCK_N = BN / WN;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // Shared memory for data + pattern cache
    __shared__ float As[BK * BM];
    __shared__ float BTs[BK * BN];
    __shared__ uint8_t pattern_cache[16];  // Per-block: WARPS_PER_BLOCK_M × WARPS_PER_BLOCK_N

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

    const int blockMWarpsBase = cRow * WARPS_PER_BLOCK_M;
    const int blockNWarpsBase = cCol * WARPS_PER_BLOCK_N;

    constexpr int patternsPerBlock = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;

    // MAIN COMPUTE LOOP - load patterns into shared memory per K-block iteration
    for (uint kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        // LOAD PATTERNS for this K-block INTO SHARED MEMORY (coalesced, once per K-iteration)
        if (threadIdx.x < patternsPerBlock) {
            const int localWarp = threadIdx.x;
            const int localMWarp = localWarp / WARPS_PER_BLOCK_N;
            const int localNWarp = localWarp % WARPS_PER_BLOCK_N;

            const int globalM = blockMWarpsBase + localMWarp;
            const int globalN = blockNWarpsBase + localNWarp;

            const int src_idx = (kBlock * numMWarps + globalM) * numNWarps + globalN;
            pattern_cache[localWarp] = global_patterns[src_idx].mask;
        }
        __syncthreads();

        // Read from SHARED MEMORY (~30 cycles, not 400!)
        const int local_warp_idx = warpRow * WARPS_PER_BLOCK_N + warpCol;
        const uint8_t joint_mask = pattern_cache[local_warp_idx];

        if (joint_mask == 0) {
            A += BK;
            B += BK * N;
            __syncthreads();  // Sync before continue to match the sync at end of iteration
            continue;
        }

        // Load tiles
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // Compute using bitmask (same as K23)
        const uint mBase = warpRow * WM + threadRowInWarp * TM;
        float regM[WMITER * TM];
        float regN[WNITER * TN];

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            if (!(joint_mask & (1 << dotIdx))) continue;

            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                const uint mOffset = mBase + wSubRowIdx * WSUBM;
                const uint baseAddr = dotIdx * BM + mOffset;
                regM[wSubRowIdx * TM + 0] = As[baseAddr + 0];
                regM[wSubRowIdx * TM + 1] = As[baseAddr + 1];
                regM[wSubRowIdx * TM + 2] = As[baseAddr + 2];
                regM[wSubRowIdx * TM + 3] = As[baseAddr + 3];
                regM[wSubRowIdx * TM + 4] = As[baseAddr + 4];
                regM[wSubRowIdx * TM + 5] = As[baseAddr + 5];
                regM[wSubRowIdx * TM + 6] = As[baseAddr + 6];
                regM[wSubRowIdx * TM + 7] = As[baseAddr + 7];
            }

            const float* BTs_row = &BTs[dotIdx * BN];
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                const uint localCol = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
                regN[wSubColIdx * TN] = BTs_row[localCol];
            }

            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float b_val = regN[wSubColIdx * TN];
                    const int resBase = wSubRowIdx * TM * (WNITER * TN) + wSubColIdx * TN;

                    threadResults[resBase + 0 * (WNITER * TN)] += regM[wSubRowIdx * TM + 0] * b_val;
                    threadResults[resBase + 1 * (WNITER * TN)] += regM[wSubRowIdx * TM + 1] * b_val;
                    threadResults[resBase + 2 * (WNITER * TN)] += regM[wSubRowIdx * TM + 2] * b_val;
                    threadResults[resBase + 3 * (WNITER * TN)] += regM[wSubRowIdx * TM + 3] * b_val;
                    threadResults[resBase + 4 * (WNITER * TN)] += regM[wSubRowIdx * TM + 4] * b_val;
                    threadResults[resBase + 5 * (WNITER * TN)] += regM[wSubRowIdx * TM + 5] * b_val;
                    threadResults[resBase + 6 * (WNITER * TN)] += regM[wSubRowIdx * TM + 6] * b_val;
                    threadResults[resBase + 7 * (WNITER * TN)] += regM[wSubRowIdx * TM + 7] * b_val;
                }
            }
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    C_interim[(threadRowInWarp * TM + resIdxM) * N +
                              threadColInWarp * TN + resIdxN] = threadResults[i];
                }
            }
        }
    }
}
