#pragma once

/* Kernel #12, Warptiling (break blocks down even further by controlling warps) */

#include "utils.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

//#include <unrolled_kernels.cuh>

/*
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
template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS, const int SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_offsets(int M, int N, int K, float *A, float *B, float *C, int* sparse_data) {
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

	__shared__ float As[BN * BK];
	__shared__ float Bs[BM * BK];

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
		for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {

			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}
		for (int8_t offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
			reinterpret_cast<float4 *>(
				&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
				reinterpret_cast<const float4 *>(
					&B[(innerRowB + offset) * N + innerColB * 4])[0];
		}
		__syncthreads();

		#pragma unroll
		for (int sparse_idx = 0; sparse_idx < SIZE; ++sparse_idx) {
			#pragma unroll
			for (int i = 0; i < 2; ++i) { 
				int dotIdx = sparse_data[sparse_idx] + (8 * i);
				for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
					regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
						wSubRowIdx * WSUBM + threadRowInWarp * TM];
				}
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
					#pragma unrol
					for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
						/* switch_table here*/
						multiply_dense(wSubRowIdx, wSubColIdx, WNITER,
						regM[wSubRowIdx], regN, threadResults);
					}
				}
			}
		}
		A += BK;
		B += BK * N;
		__syncthreads();
	}

	for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
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

