#pragma once

/* ESMM Kernel with Pattern-Specialized Computation - Zero Branch Overhead */

#include "../../include/utils.cuh"
#include "../../include/pattern_functions_bk8.cuh"
#include <cuda_runtime.h>

/*
 * Uses row-level bitmask metadata + pattern-specialized compute functions
 * for zero-overhead sparsity exploitation.
 *
 * Each warp dispatches to one of 256 precompiled functions based on its
 * sparsity pattern, eliminating ALL branches and mask checks in the inner loop.
 *
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension (MUST BE 8 for this version).
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_pattern_specialized(int M, int N, int K, float *A, float *B, float *C, uint8_t* A_LIST) {

	static_assert(BK == 8, "Pattern-specialized kernel only supports BK=8");

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

	// Only cache masks for current tile rows
	__shared__ uint8_t rowMasks[BM];

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

	const uint globalRowBase = cRow * BM;

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
		const uint kBlock = bkIdx / BK;

		// Load masks for current k-block only
		for (uint i = threadIdx.x; i < BM; i += blockDim.x) {
			rowMasks[i] = A_LIST[(globalRowBase + i) * numKBlocks + kBlock];
		}
		__syncthreads();

		uint8_t warpMask = 0;
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			const uint localRow = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM;
			if (localRow < BM) {
				warpMask |= rowMasks[localRow];
			}
		}

		// Load A tile (unconditionally - could optimize this too)
		for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}

		// Conditionally load B rows based on warp mask
		for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
			int bRow = innerRowB + offset;
			if (warpMask & (1 << bRow)) {
				reinterpret_cast<float4 *>(
					&Bs[bRow * BN + innerColB * 4])[0] =
					reinterpret_cast<const float4 *>(
						&B[bRow * N + innerColB * 4])[0];
			}
		}
		__syncthreads();

		// *** PATTERN-SPECIALIZED DISPATCH - ZERO OVERHEAD! ***
		// All threads in warp have same warpMask, so no divergence
		// Each pattern function has zero branches - just hardcoded loads/multiplies
		dispatch_pattern(
			warpMask,
			As, Bs, threadResults,
			warpRow, warpCol,
			threadRowInWarp, threadColInWarp,
			WM, WN, TM, TN,
			BM, BN, WMITER, WNITER,
			WSUBM, WSUBN
		);

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
