#pragma once

/*
 * ============================================================================
 * Kernel 23: Joint A+B Sparsity with PRECOMPUTED Intersections
 * ============================================================================
 *
 * Revolutionary Idea:
 *   Instead of computing A∩B intersections at runtime, PRECOMPUTE them
 *   during preprocessing and encode as single joint patterns.
 *
 * Why This Should Win:
 *   - Zero runtime intersection overhead
 *   - Single pattern read (not two)
 *   - Single switch dispatch (not 2D)
 *   - Direct offset iteration (fully unrolled)
 *   - Still gets 64× compute reduction from joint sparsity
 *
 * Key Innovation:
 *   JointPattern struct stores:
 *   - count: number of joint non-zero K-indices
 *   - offsets[8]: the actual joint offsets
 *   - Precomputed during preprocessing, indexed by (kBlock, mBlock, nBlock)
 *
 * Performance Target:
 *   Should match K17 speed (~13 GFLOPS) while exploiting both A and B sparsity
 *   At 87.5% sparsity: 64× theoretical speedup vs dense
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

// ============================================================================
// Joint Pattern Structure (Compressed - 1 byte!)
// ============================================================================

// Ultra-compact: Use 8-bit mask for BK=8
// Bit i set = K-index i is non-zero in both A and B
struct JointPattern {
    uint8_t mask;  // Bitmask of joint non-zero indices
};

// ============================================================================
// Joint Preprocessing Kernel
// ============================================================================

template <int BK>
__global__ void compute_joint_patterns_kernel(
    const uint8_t* A_patterns,
    const uint8_t* B_patterns,
    JointPattern* joint_patterns,
    int numKBlocks,
    int numMWarps,
    int numNWarps) {

    // Each thread processes one (kBlock, mWarp, nWarp) combination
    const int kBlock = blockIdx.x;
    const int mWarp = blockIdx.y;
    const int nWarp = threadIdx.x;

    if (nWarp >= numNWarps) return;

    // Get A and B patterns
    const uint A_idx = kBlock * numMWarps + mWarp;
    const uint B_idx = nWarp * numKBlocks + kBlock;

    const uint8_t A_pattern = A_patterns[A_idx];
    const uint8_t B_pattern = B_patterns[B_idx];

    const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
    const uint8_t B_count = PATTERN_LUT_BK8[B_pattern].count;

    const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;
    const uint8_t* B_offsets = PATTERN_LUT_BK8[B_pattern].offsets;

    // Compute intersection as bitmask
    JointPattern jp;
    jp.mask = 0;

    for (int a = 0; a < A_count; ++a) {
        uint8_t a_offset = A_offsets[a];
        for (int b = 0; b < B_count; ++b) {
            if (B_offsets[b] == a_offset) {
                jp.mask |= (1 << a_offset);  // Set bit for this offset
                break;
            }
        }
    }

    // Store joint pattern
    const int joint_idx = (kBlock * numMWarps + mWarp) * numNWarps + nWarp;
    joint_patterns[joint_idx] = jp;
}

// ============================================================================
// Preprocessing Function
// ============================================================================

struct JointPatternMetadata {
    JointPattern* d_patterns;
    int numKBlocks;
    int numMWarps;
    int numNWarps;
};

JointPatternMetadata preprocess_joint_patterns(
    float* d_A, float* d_B,
    int M, int N, int K,
    int BM, int BN, int BK,
    int WM, int WN) {

    JointPatternMetadata meta;

    // First get A and B patterns separately
    BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, M, K, WM, BK);
    BTPatternMetadata B_meta = preprocess_b_transpose(d_B, K, N, WN, BK);

    meta.numKBlocks = K / BK;
    meta.numMWarps = (M / BM) * (BM / WM);  // Total M-warps across all blocks
    meta.numNWarps = (N / BN) * (BN / WN);  // Total N-warps across all blocks

    const int totalJointPatterns = meta.numKBlocks * meta.numMWarps * meta.numNWarps;

    cudaMalloc(&meta.d_patterns, totalJointPatterns * sizeof(JointPattern));

    // Launch kernel to compute joint patterns
    dim3 gridDim(meta.numKBlocks, meta.numMWarps);
    dim3 blockDim(meta.numNWarps);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    compute_joint_patterns_kernel<8><<<gridDim, blockDim>>>(
        A_meta.d_blockPatterns,
        B_meta.d_blockPatterns,
        meta.d_patterns,
        meta.numKBlocks,
        meta.numMWarps,
        meta.numNWarps);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Joint pattern preprocessing error: %s\n", cudaGetErrorString(error));
    }

    float metadataKB = totalJointPatterns * sizeof(JointPattern) / 1024.0f;
    printf("Joint pattern preprocessing: %d patterns (%.1f KB) in %.3f ms\n",
           totalJointPatterns, metadataKB, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free A and B patterns
    free_block_pattern_metadata(A_meta);
    free_bt_pattern_metadata(B_meta);

    return meta;
}

void free_joint_pattern_metadata(JointPatternMetadata meta) {
    cudaFree(meta.d_patterns);
}

// ============================================================================
// Compute Function with Joint Bitmask (ULTRA-COMPACT)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__device__ void compute_with_joint_bitmask(
    const uint8_t mask,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* BTs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Extract offsets from bitmask into array (done once per warp, shared across threads)
    uint8_t offsets[8];
    int offset_count = 0;

    #pragma unroll
    for (int i = 0; i < BK; ++i) {
        if (mask & (1 << i)) {
            offsets[offset_count++] = i;
        }
    }

    // Now iterate ONLY over non-zero offsets - WARP UNIFORM!
    const uint mBase = warpRow * WM + threadRowInWarp * TM;

    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        if (idx >= offset_count) break;  // Warp-uniform break

        const uint8_t dotIdx = offsets[idx];

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
}

// ============================================================================
// Main Kernel
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_joint_precomputed(int M, int N, int K,
                          float *A, float *B, float *C,
                          const JointPattern* __restrict__ joint_patterns,
                          int numKBlocks, int numMWarps, int numNWarps) {

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
    __shared__ float BTs[BK * BN];

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

    constexpr int WARPS_PER_BLOCK_M = BM / WM;
    constexpr int WARPS_PER_BLOCK_N = BN / WN;

    const int blockMWarpsBase = cRow * WARPS_PER_BLOCK_M;
    const int blockNWarpsBase = cCol * WARPS_PER_BLOCK_N;

    const int globalMWarp = blockMWarpsBase + warpRow;
    const int globalNWarp = blockNWarpsBase + warpCol;

    for (uint kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        // Single byte read - ULTRA-COMPACT! Use __ldg for texture cache
        const int joint_idx = (kBlock * numMWarps + globalMWarp) * numNWarps + globalNWarp;
        const uint8_t joint_mask = __ldg(&joint_patterns[joint_idx].mask);

        if (joint_mask == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load tiles
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            BTs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // Direct compute with bitmask - NO SWITCH!
        compute_with_joint_bitmask<BM, BN, BK, WM, WN, WNITER, TM, TN>(
            joint_mask, warpRow, warpCol, threadRowInWarp, threadColInWarp,
            As, BTs, threadResults);

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
