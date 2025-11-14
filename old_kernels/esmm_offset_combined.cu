#pragma once

/*
 * ============================================================================
 * Kernel 20: ESMM Combined A+B Sparsity with Offset Lists (8x8 Templates)
 * ============================================================================
 *
 * Strategy:
 *   Use offset lists for BOTH A and B sparsity dimensions.
 *   Template on the COUNT of non-zero elements (0-8 for each dimension).
 *   64 specialized functions (8×8) with fully unrolled loops.
 *
 * Architecture:
 *   - A-Sparsity: Offset list per warp (BK × WM blocks)
 *   - B-Sparsity: Offset list per thread per iteration (BK × TN blocks)
 *   - Template specialization: compute_sparse_block<A_COUNT, B_COUNT>
 *   - Zero branches in inner loops (all unrolled at compile time)
 *
 * Benefits:
 *   - Compound sparsity: 50% A + 50% B = 75% work reduction
 *   - Perfect loop unrolling with #pragma unroll
 *   - Load only non-zero elements (both A and B)
 *   - Small code footprint: 64 functions × ~150 bytes = 9.6KB
 *
 * Expected Performance:
 *   - 50% sparsity: ~3-4× speedup over dense
 *   - 75% sparsity: ~8-12× speedup over dense
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include "../preprocessors/b_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

// ============================================================================
// Templated Sparse Compute (64 specializations: 8 A-counts × 8 B-counts)
// ============================================================================

#pragma nv_diag_suppress 128  // Suppress "loop is not reachable" for COUNT=0 instantiations
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT, const int B_COUNT>
__device__ __forceinline__ void compute_sparse_block_offsets(
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint wSubColIdx,
    const uint cCol, const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[B_COUNT > 0 ? B_COUNT : 1];  // Avoid zero-sized array

    // Early exit for fully sparse blocks
    if (A_COUNT == 0 || B_COUNT == 0) {
        return;
    }

    // Loop over non-zero A elements (fully unrolled)
    for (int a_idx = 0; a_idx < A_COUNT; ++a_idx) {
        const uint8_t dotIdx = A_offsets[a_idx];

        // Load A values for this K index
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx * TM] = As[(dotIdx * BM) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        // Load B values (only non-zero ones)
        const uint baseBAddr = (dotIdx * BN) + warpCol * WN +
            wSubColIdx * WSUBN + threadColInWarp * TN;

        #pragma unroll
        for (int b_idx = 0; b_idx < B_COUNT; ++b_idx) {
            const uint8_t n_offset = B_offsets[b_idx];
            regN[b_idx] = Bs[baseBAddr + n_offset];
        }

        // Compute: outer product of regM and regN
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            const float regM_val = regM[wSubRowIdx * TM];
            const int resRowBase = wSubRowIdx * (WNITER * TN);
            const int resColBase = wSubColIdx * TN;

            #pragma unroll
            for (int b_idx = 0; b_idx < B_COUNT; ++b_idx) {
                const uint8_t n_offset = B_offsets[b_idx];
                threadResults[resRowBase + resColBase + n_offset] += regM_val * regN[b_idx];
            }
        }
    }
}

// ============================================================================
// Dispatch Functions (2D switch on A_COUNT and B_COUNT)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int A_COUNT>
__device__ __forceinline__ void dispatch_on_B_count(
    const uint8_t B_count,
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint wSubColIdx,
    const uint cCol, const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    switch(B_count) {
        case 0: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,0>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 1: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,1>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 2: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,2>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 3: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,3>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 4: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,4>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 5: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,5>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 6: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,6>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 7: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,7>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 8: compute_sparse_block_offsets<BM,BN,BK,WM,WN,WNITER,TM,TN,A_COUNT,8>(
            A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__device__ __forceinline__ void dispatch_on_A_B_count(
    const uint8_t A_count,
    const uint8_t B_count,
    const uint8_t* A_offsets,
    const uint8_t* B_offsets,
    const uint wSubColIdx,
    const uint cCol, const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs,
    float* threadResults) {

    switch(A_count) {
        case 0: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,0>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 1: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,1>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 2: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,2>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 3: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,3>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 4: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,4>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 5: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,5>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 6: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,6>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 7: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,7>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
        case 8: dispatch_on_B_count<BM,BN,BK,WM,WN,WNITER,TM,TN,8>(
            B_count, A_offsets, B_offsets, wSubColIdx, cCol, warpRow, warpCol,
            threadRowInWarp, threadColInWarp, As, Bs, threadResults); break;
    }
}

// ============================================================================
// Main Kernel
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_offset_combined(int M, int N, int K, float *A, float *B, float *C,
                         uint8_t* A_patterns, uint8_t* B_patterns) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;

    // These are declared for potential future use
    [[maybe_unused]] const int threadCol = threadIdx.x % (BN / TN);
    [[maybe_unused]] const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpRow = (warpIdx * WM) / BM;
    const uint warpCol = (warpIdx * WM) % BM / WN;

    const uint threadRowInWarp = (threadIdx.x % WARPSIZE) / (WN / TN);
    const uint threadColInWarp = (threadIdx.x % WARPSIZE) % (WN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = NUM_THREADS / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = NUM_THREADS / BN;

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    const int numKBlocks = K / BK;
    const int numNBlocks = N / BK;
    const int numMBlocks = M / BM;

    const int warpLinearIdx = cRow * (NUM_WARPS / (BM / WM)) + warpRow;

    for (uint kBlock = 0; kBlock < numKBlocks; ++kBlock) {
        // Load A tile into shared memory
        #pragma unroll
        for (uint offset = 0; offset < BM; offset += strideA) {
            As[(innerColA * BM) + innerRowA + offset] =
                A[(innerRowA + offset) * K + innerColA];
        }

        // Load B tile into shared memory
        #pragma unroll
        for (uint offset = 0; offset < BK; offset += strideB) {
            Bs[(innerRowB + offset) * BN + innerColB] =
                B[(innerRowB + offset) * N + innerColB];
        }
        __syncthreads();

        // Get A sparsity pattern for this warp's M-block
        const uint A_patternIdx = kBlock * numMBlocks * (NUM_WARPS / (BM / WM)) +
            warpLinearIdx * WMITER;
        const uint8_t A_pattern = A_patterns[A_patternIdx];
        const uint8_t A_count = PATTERN_LUT_BK8[A_pattern].count;
        const uint8_t* A_offsets = PATTERN_LUT_BK8[A_pattern].offsets;

        // Process each WNITER iteration with its own B-pattern
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint globalColBase = cCol * BN + warpCol * WN +
                wSubColIdx * WSUBN + threadColInWarp * TN;
            const uint nBlockIdx = globalColBase >> 3;
            const uint B_patternIdx = kBlock * numNBlocks + nBlockIdx;
            const uint8_t B_pattern = B_patterns[B_patternIdx];
            const uint8_t B_count = PATTERN_LUT_BK8[B_pattern].count;
            const uint8_t* B_offsets = PATTERN_LUT_BK8[B_pattern].offsets;

            // Dispatch to specialized template (one of 64 variants)
            dispatch_on_A_B_count<BM, BN, BK, WM, WN, WNITER, TM, TN>(
                A_count, B_count, A_offsets, B_offsets, wSubColIdx,
                cCol, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                As, Bs, threadResults);
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write results to global memory
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const float* regTilePtr = threadResults +
                (wSubRowIdx * TM * WNITER * TN) + (wSubColIdx * TN);

            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    const uint globalRow = warpRow * WM + wSubRowIdx * WSUBM +
                        threadRowInWarp * TM + resIdxM;
                    const uint globalCol = warpCol * WN + wSubColIdx * WSUBN +
                        threadColInWarp * TN + resIdxN;
                    C[globalRow * N + globalCol] = regTilePtr[resIdxM * TN + resIdxN];
                }
            }
        }
    }
}
