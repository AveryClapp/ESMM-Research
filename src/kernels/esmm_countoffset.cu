#pragma once

/* ESMM Kernel with Count+Offset Metadata - Unrolled Dispatch */

#include "../../include/utils.cuh"
#include "../preprocessors/a_preprocessor_countoffset.cu"
#include <cuda_runtime.h>

/*
 * Uses count+offset metadata for direct, unrolled computation
 * Dispatches based on count to fully unrolled code paths
 *
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */

// Helper macros for unrolled computation at a specific offset
#define COMPUTE_AT_OFFSET(dotIdx) \
    do { \
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) { \
            regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM + \
                wSubRowIdx * WSUBM + threadRowInWarp * TM]; \
        } \
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) { \
            regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0]; \
            regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1]; \
            regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2]; \
            regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3]; \
            regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4]; \
            regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5]; \
            regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6]; \
            regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol * \
                WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7]; \
        } \
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) { \
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) { \
                multiply_dense(wSubRowIdx, wSubColIdx, WNITER, \
                    regM[wSubRowIdx], regN, threadResults); \
            } \
        } \
    } while(0)

template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_countoffset(int M, int N, int K, float *A, float *B, float *C,
	                 CountOffset* __restrict__ metadata) {

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

	const uint numKBlocks = K / BK;

	__shared__ float As[BM * BK];
	__shared__ float Bs[BN * BK];

	// Cache count+offset metadata for current tile rows
	__shared__ CountOffset tileMeta[BM];

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
	float regM[WMITER * TM] = {0.0};
	float regN[WNITER * TN] = {0.0};

	const uint globalRowBase = cRow * BM;

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
		const uint kBlock = bkIdx / BK;

		// Load metadata for current k-block
		for (uint i = threadIdx.x; i < BM; i += blockDim.x) {
			tileMeta[i] = metadata[(globalRowBase + i) * numKBlocks + kBlock];
		}
		__syncthreads();

		// Aggregate count+offsets for this warp's rows
		// We'll use the maximum count and union of offsets (simple heuristic)
		uint8_t maxCount = 0;
		uint8_t warpOffsets[4] = {0, 0, 0, 0};
		uint8_t warpMask = 0;

		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			const uint localRow = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM;
			if (localRow < BM) {
				const CountOffset& co = tileMeta[localRow];
				if (co.count > maxCount) {
					maxCount = co.count;
					for (int i = 0; i < 4; i++) {
						warpOffsets[i] = co.offsets[i];
					}
				}
				// Also build a bitmask for fallback
				for (int i = 0; i < co.count && i < 4; i++) {
					warpMask |= (1 << co.offsets[i]);
				}
			}
		}

		// Load A tile
		for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}

		// Load B tile
		for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
			reinterpret_cast<float4 *>(
				&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
				reinterpret_cast<const float4 *>(
					&B[(innerRowB + offset) * N + innerColB * 4])[0];
		}
		__syncthreads();

		// *** COUNT-BASED DISPATCH - FULLY UNROLLED! ***
		// Each case is completely unrolled with zero branches in the computation
		switch (maxCount) {
			case 0:
				// Skip entirely - all zeros
				break;

			case 1:
				// Exactly 1 multiply at offsets[0]
				COMPUTE_AT_OFFSET(warpOffsets[0]);
				break;

			case 2:
				// Exactly 2 multiplies at offsets[0] and offsets[1]
				COMPUTE_AT_OFFSET(warpOffsets[0]);
				COMPUTE_AT_OFFSET(warpOffsets[1]);
				break;

			case 3:
				// Exactly 3 multiplies
				COMPUTE_AT_OFFSET(warpOffsets[0]);
				COMPUTE_AT_OFFSET(warpOffsets[1]);
				COMPUTE_AT_OFFSET(warpOffsets[2]);
				break;

			case 4:
				// Exactly 4 multiplies
				COMPUTE_AT_OFFSET(warpOffsets[0]);
				COMPUTE_AT_OFFSET(warpOffsets[1]);
				COMPUTE_AT_OFFSET(warpOffsets[2]);
				COMPUTE_AT_OFFSET(warpOffsets[3]);
				break;

			default:
				// For count > 4, fall back to bitmask iteration
				// (Still better than always using bitmask!)
				for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
					if (!(warpMask & (1 << dotIdx))) continue;
					COMPUTE_AT_OFFSET(dotIdx);
				}
				break;
		}

		A += BK;
		B += BK * N;
		__syncthreads();
	}

	// Write results back
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

#undef COMPUTE_AT_OFFSET
