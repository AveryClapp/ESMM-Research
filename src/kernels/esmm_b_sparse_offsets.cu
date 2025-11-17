#pragma once

#include "../../include/utils.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ESMM Kernel: B-Sparse Matrix Multiplication with Templated K-wise Offsets
// SIZE = number of non-zero K-rows to process (compile-time constant)
// b_offsets = array of SIZE elements indicating which K-rows (0-7) to process
template <const int BM, const int BN, const int BK, const int WM,
			const int WN, const int WNITER, const int TM, const int TN,
			const int NUM_THREADS, const int SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_b_sparse_offsets(int M, int N, int K, float *A, float *B, float *C,
						  const uint8_t* __restrict__ b_offsets) {
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
	__shared__ float Bs[BK * BN];

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

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
		// Load A into shared memory (column-major)
		for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}

		// Load B into shared memory (row-major)
		for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
			reinterpret_cast<float4 *>(
				&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
				reinterpret_cast<const float4 *>(
					&B[(innerRowB + offset) * N + innerColB * 4])[0];
		}
		__syncthreads();

		// This is good for right now, but assumes row-sparsity. Bitwise & with A
		#pragma unroll
		for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
			#pragma unroll
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				regM[wSubRowIdx] = As[dotIdx * BM + warpRow * WM +
					wSubRowIdx * WSUBM + threadRowInWarp * TM];
			}

			#pragma unroll
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint offset = 0; offset < TM; ++offset) {
				regN[wSubColIdx * TN + offset] = Bs[(dotIdx * BN) + warpCol * 
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + offset];
				}
			}

			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
					for (uint8_t sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
						const uint8_t offIdx = b_offsets[sparse_idx];
						const int regNBase = wSubColIdx * TN;
						const int threadResBase = wSubRowIdx * (WNITER * TN) + (wSubColIdx * TN);
						threadResults[threadResBase + offIdx] += regM[wSubRowIdx] * regN[regNBase + offIdx];
					}
				}
			}
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

