#pragma once

/*
 * ============================================================================
 * Kernel 22: B-Transpose + 8×8 Offset Templates (THE SYNTHESIS)
 * ============================================================================
 *
 * Strategy:
 *   Transpose B so B-sparsity is in K-dimension (same as A).
 *   Use offset lists for BOTH A and B^T in K-dimension.
 *   Template on (A_COUNT, B_COUNT) for fully unrolled loops.
 *
 * Why This Should Work:
 *   K21 problem: B-sparsity in N-dimension → scattered column access
 *   K20 insight: B-transpose → B-sparsity in K-dimension → coalesced row access
 *   K21 insight: 8×8 templates → zero-branch fully unrolled loops
 *
 *   COMBINE: Both A and B^T use offset lists in K-dimension
 *            Both maintain coalesced memory access
 *            Joint sparsity: skip K-iterations where BOTH are zero
 *
 * Architecture:
 *   - B^T is N×K (transposed preprocessing)
 *   - A patterns: per WM×BK block (K-dimension)
 *   - B^T patterns: per WN×BK block (K-dimension)
 *   - Template: 64 variants (8 A-counts × 8 B-counts)
 *   - Computation: only iterate over K-indices where both A AND B^T are non-zero
 *
 * Expected Performance:
 *   - 50% A + 50% B = 25% joint density → 4× theoretical speedup
 *   - All memory access coalesced (both A and B^T)
 *   - No dispatch overhead (templates)
 *   - Should approach K17 speeds with B-sparsity benefits
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_transpose_preprocessor.cu"
#include <cuda_runtime.h>

// ============================================================================
// Joint A+B^T Offset Computation (8×8 Templates)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT, const int B_COUNT>
__device__ __forceinline__ void compute_joint_offsets(
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    // Early exit for fully sparse
    if (A_COUNT == 0 || B_COUNT == 0) {
        return;
    }

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Iterate over A offsets and check if B also has them (no dynamic array)
    #pragma unroll
    for (int a_idx = 0; a_idx < A_COUNT; ++a_idx) {
        const uint8_t dotIdx = A_offsets[a_idx];

        // Check if B also has this offset
        bool b_has_offset = false;
        #pragma unroll
        for (int b_idx = 0; b_idx < B_COUNT; ++b_idx) {
            if (B_offsets[b_idx] == dotIdx) {
                b_has_offset = true;
                break;
            }
        }

        if (!b_has_offset) continue;

        // Load A values
        const uint mBase = warpRow * WM + threadRowInWarp * TM;

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

        // Load B values (BTs[dotIdx * BN + col])
        const float* BTs_row = &Bs[dotIdx * BN];
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint localCol = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;
            regN[wSubColIdx * TN] = BTs_row[localCol];
        }

        // Outer product (like btranspose)
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
// Dispatch (2D switch on A_COUNT and B_COUNT)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT>
__device__ __forceinline__ void dispatch_B_count(
    const uint8_t B_count,
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    switch(B_count) {
        case 0: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,0>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 1: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,1>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 2: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,2>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 3: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,3>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 4: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,4>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 5: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,5>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 6: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,6>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 7: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,7>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 8: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,8>(
            A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__device__ __forceinline__ void dispatch_A_B_count(
    const uint8_t A_count,
    const uint8_t B_count,
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    switch(A_count) {
        case 0: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,0>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 1: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,1>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 2: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,2>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 3: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,3>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 4: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,4>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 5: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,5>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 6: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,6>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 7: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,7>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 8: dispatch_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,8>(
            B_count, A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
    }
}

// ============================================================================
// Main Kernel
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_btranspose_offset(int M, int N, int K, float *A, float *B, float *C,
                           uint8_t* A_patterns, uint8_t* B_patterns) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

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

    const int numKBlocks = K / BK;
    const int numMBlocks = M / BM;

    const int warpLinearIdx = cRow * (NUM_WARPS / (BM / WM)) + warpRow;
    const uint globalColBlock = cCol * (BN / WN) + warpCol;

    for (uint kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        // Get A and B patterns first (like btranspose)
        const uint A_patternIdx = kBlock * numMBlocks * (NUM_WARPS / (BM / WM)) +
            warpLinearIdx * WMITER;
        const uint8_t A_pattern = A_patterns[A_patternIdx];
        const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;

        const uint B_patternIdx = globalColBlock * numKBlocks + kBlock;
        const uint8_t B_pattern = B_patterns[B_patternIdx];
        const uint8_t B_count = PATTERN_LUT_BK8[B_pattern].count;
        const uint8_t* B_offsets = PATTERN_LUT_BK8[B_pattern].offsets;

        // Skip if either A or B is fully sparse
        if (A_count == 0 || B_count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load A tile (column-major with padding)
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // Load B tile and transpose (like btranspose)
        for (int32_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();

        // Use switch on A_count only (B patterns checked inside)
        switch (A_count) {
            case 1: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,1,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 2: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,2,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 3: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,3,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 4: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,4,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 5: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,5,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 6: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,6,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 7: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,7,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
            case 8: compute_joint_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,8,8>(
                A_offsets, B_offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write results (like btranspose)
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
