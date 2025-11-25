#pragma once

/*
 * ============================================================================
 * Kernel 20: ESMM B-Sparse with Shared Memory Transpose (K20)
 * ============================================================================
 *
 * Architecture:
 *   - Load B tile into shared memory Bs[BK×BN]
 *   - Transpose in shared memory to Bs_T[BN×BK]
 *   - Check WN-granularity pattern ONCE per tile (outside K-loop)
 *   - Template dispatch on pattern → fully unrolled inner loops
 *   - Access Bs_T[n][k] where each warp handles WN consecutive columns
 *
 * Key Advantages:
 *   - Dispatch overhead: Once per tile (not per K-iteration)
 *   - Warp-uniform: All threads in warp use same pattern
 *   - No global transpose: Transpose happens in shared memory
 *   - Cache-friendly: Transposed data stays in shared memory
 *
 * Pattern Format:
 *   - Granularity: WN columns × BK rows
 *   - Storage: column-major, one pattern per WN columns
 *   - Size: (N/WN) × (K/BK) bytes = ~64 KB for 4096×4096
 *
 * Memory Layout:
 *   - Bs[BK][BN]: Original B tile (row-major in memory)
 *   - Bs_T[BN][BK]: Transposed B tile (column-major access)
 *   - As[BK][BM+1]: A tile (column-major with padding)
 *
 * Performance Target: ~6-7 ms (match A-sparse K16)
 */

#include "../../include/utils.cuh"
#include "../../include/metadata.cuh"
#include "../../include/pattern_lut.cuh"
#include "../preprocessors/b_preprocessor_wn.cu"
#include <cuda_runtime.h>

// ============================================================================
// Shared Memory Transpose: Bs[BK×BN] → Bs_T[BN×BK]
// ============================================================================

template <const int BK, const int BN, const int NUM_THREADS>
__device__ void transpose_shared_memory(
    const float* __restrict__ Bs,     // [BK][BN] row-major
    float* __restrict__ Bs_T          // [BN][BK] output
) {
    constexpr int TOTAL_ELEMENTS = BK * BN;
    constexpr int ELEMENTS_PER_THREAD = (TOTAL_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;

    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        const int flatIdx = tid + i * NUM_THREADS;
        if (flatIdx < TOTAL_ELEMENTS) {
            const int k = flatIdx / BN;  // K-position (0..BK-1)
            const int n = flatIdx % BN;  // N-position (0..BN-1)

            // Transpose: Bs[k][n] → Bs_T[n][k]
            Bs_T[n * BK + k] = Bs[k * BN + n];
        }
    }
}

// ============================================================================
// Sparse Computation: Template specialized on pattern count
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int SIZE>
__device__ void compute_sparse_tile_k20(
    const uint8_t* offsets,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    float* As, float* Bs_T,
    float* threadResults) {

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    float regM[WMITER * TM];
    float regN[WNITER * TN];

    // Iterate only over NON-ZERO K-positions (determined by pattern)
    #pragma unroll
    for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
        const uint8_t dotIdx = offsets[sparse_idx];

        // Load A tile column dotIdx
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                wSubRowIdx * WSUBM + threadRowInWarp * TM];
        }

        // Load from TRANSPOSED B: Bs_T[n][k] format
        // Bs_T is [BN×BK], so access as Bs_T[n * BK + k]
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            const uint nBase = warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN;

            // Load 8 elements from Bs_T: columns [nBase..nBase+7], row dotIdx
            regN[wSubColIdx * TN + 0] = Bs_T[(nBase + 0) * BK + dotIdx];
            regN[wSubColIdx * TN + 1] = Bs_T[(nBase + 1) * BK + dotIdx];
            regN[wSubColIdx * TN + 2] = Bs_T[(nBase + 2) * BK + dotIdx];
            regN[wSubColIdx * TN + 3] = Bs_T[(nBase + 3) * BK + dotIdx];
            regN[wSubColIdx * TN + 4] = Bs_T[(nBase + 4) * BK + dotIdx];
            regN[wSubColIdx * TN + 5] = Bs_T[(nBase + 5) * BK + dotIdx];
            regN[wSubColIdx * TN + 6] = Bs_T[(nBase + 6) * BK + dotIdx];
            regN[wSubColIdx * TN + 7] = Bs_T[(nBase + 7) * BK + dotIdx];
        }

        // Outer product: regM × regN → threadResults
        #pragma unroll
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
                    regM[wSubRowIdx], regN, threadResults);
            }
        }
    }
}

// ============================================================================
// Main Kernel: B-Sparse with Shared Memory Transpose
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    esmm_b_smem_transpose(int M, int N, int K, float *A, float *B, float *C,
                          const uint8_t* __restrict__ colPatterns,
                          const int numKBlocks) {

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

    // Shared memory: need space for both original and transposed
    // Alternative: use two separate buffers for clarity
    __shared__ float As[BK * (BM + 1)];
    __shared__ float Bs[BK * BN];
    __shared__ float Bs_T[BN * BK];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Each warp has a unique column block
    const uint globalWarpCol = cCol * (BN / WN) + warpCol;

    // Main K-loop over tiles
    for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const uint kBlock = bkIdx / BK;

        // CRITICAL: Load pattern ONCE per tile, outside inner loop
        // This is the key difference from TN-granularity approach
        const uint patternIdx = globalWarpCol * numKBlocks + kBlock;
        const uint8_t pattern = colPatterns[patternIdx];
        const uint8_t count = PATTERN_LUT_BK8[pattern].count;
        const uint8_t* offsets = PATTERN_LUT_BK8[pattern].offsets;

        // Early exit if entire tile is zero
        if (count == 0) {
            A += BK;
            B += BK * N;
            continue;
        }

        // Load A tile (column-major into shared memory)
        for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }

        // Load B tile (row-major into shared memory)
        for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(
                    &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                        &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        // Transpose B in shared memory: Bs[BK×BN] → Bs_T[BN×BK]
        transpose_shared_memory<BK, BN, NUM_THREADS>(Bs, Bs_T);
        __syncthreads();

        // Dispatch based on sparsity count (ONCE per tile, not per K-iteration!)
        switch (count) {
            case 1:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 1>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 2:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 2>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 3:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 3>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 4:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 4>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 5:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 5>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 6:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 6>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 7:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 7>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
            case 8:
            default:
                compute_sparse_tile_k20<BM, BN, BK, WM, WN, WNITER, TM, TN, 8>(
                    offsets, warpRow, warpCol, threadRowInWarp, threadColInWarp,
                    As, Bs_T, threadResults);
                break;
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Write results back to C
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp;
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4 *>(
                            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                            threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}
