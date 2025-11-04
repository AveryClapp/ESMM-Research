#pragma once

/* ESMM Hybrid Kernel: Combines offset list (K13) with pattern-specialized (K18) */

#include "../../include/utils.cuh"
#include "../preprocessors/a_preprocessor_hybrid.cu"
#include <cuda_runtime.h>

/*
 * Hybrid kernel with two modes:
 *
 * MODE 1 - UNIFORM PATTERN (isUniform=true):
 *   - Uses global offset list (16 bytes total!)
 *   - Template SIZE parameter for compile-time unrolling
 *   - Zero metadata loads from global memory
 *   - Extremely fast for uniform sparsity
 *
 * MODE 2 - NON-UNIFORM PATTERN (isUniform=false):
 *   - Falls back to per-row bitmasks (1 byte/row/K-block)
 *   - Still better than count+offset (1 byte vs 5 bytes)
 *   - Handles arbitrary per-row variation
 */

// ============================================================================
// MODE 1: UNIFORM PATTERN KERNEL (Uses global offset list)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS, const int SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_hybrid_uniform(int M, int N, int K, float *A, float *B, float *C,
	                    const uint8_t* __restrict__ globalOffsets) {

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

	__shared__ float As[BM * BK];
	__shared__ float Bs[BN * BK];

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
	float regM[WMITER * TM];
	float regN[WNITER * TN];

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
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

		// *** MAGIC: Compile-time unrolled loop using global offsets ***
		// For SIZE=1: Unrolls to single iteration
		// For SIZE=4: Unrolls to 4 iterations
		// Zero runtime overhead!
		#pragma unroll
		for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
			const uint8_t dotIdx = globalOffsets[sparse_idx];

			#pragma unroll
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
					wSubRowIdx * WSUBM + threadRowInWarp * TM];
			}

			#pragma unroll
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
				regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
				regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
				regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
				regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
				regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
				regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
				regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
			}

			#pragma unroll
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				#pragma unroll
				for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
					multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
						regM[wSubRowIdx], regN, threadResults);
				}
			}
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

// ============================================================================
// MODE 2: NON-UNIFORM PATTERN KERNEL (Uses per-row bitmasks)
// ============================================================================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_hybrid_nonuniform(int M, int N, int K, float *A, float *B, float *C,
	                       const uint8_t* __restrict__ rowMasks) {

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
	__shared__ uint8_t tileMasks[BM];

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
	float regM[WMITER * TM];
	float regN[WNITER * TN];

	const uint globalRowBase = cRow * BM;

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
		const uint kBlock = bkIdx / BK;

		// Load masks for current k-block
		for (uint i = threadIdx.x; i < BM; i += blockDim.x) {
			tileMasks[i] = rowMasks[(globalRowBase + i) * numKBlocks + kBlock];
		}
		__syncthreads();

		// Aggregate mask for this thread's rows
		uint8_t threadMask = 0;
		#pragma unroll
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			const uint localRow = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM;
			if (localRow < BM) {
				threadMask |= tileMasks[localRow];
			}
		}

		// Early exit if all zeros
		if (threadMask == 0) {
			A += BK;
			B += BK * N;
			__syncthreads();
			continue;
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

		// Compute using bitmask
		for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
			if (!(threadMask & (1 << dotIdx))) continue;

			#pragma unroll
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
					wSubRowIdx * WSUBM + threadRowInWarp * TM];
			}

			#pragma unroll
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				regN[wSubColIdx * TN + 0] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
				regN[wSubColIdx * TN + 1] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
				regN[wSubColIdx * TN + 2] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
				regN[wSubColIdx * TN + 3] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
				regN[wSubColIdx * TN + 4] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
				regN[wSubColIdx * TN + 5] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
				regN[wSubColIdx * TN + 6] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
				regN[wSubColIdx * TN + 7] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
			}

			#pragma unroll
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				#pragma unroll
				for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
					multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
						regM[wSubRowIdx], regN, threadResults);
				}
			}
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
