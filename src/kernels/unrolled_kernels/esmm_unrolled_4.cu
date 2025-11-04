#pragma once

/* Fully unrolled kernels for SIZE=6,4,2,1 with innermost loop unrolled */

#include "../../../include/utils.cuh"
#include <cuda_runtime.h>

// ============================================================================
// SIZE=4: dotIdx = 0,1,2,3
// ============================================================================
template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_unrolled_4(int M, int N, int K, float *A, float *B, float *C) {
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

		// Iterations 0-3 (dotIdx = 0,1,2,3)
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(0 * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(0 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 0] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 0];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 1] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 1];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 2] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 2];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 3] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 3];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 4] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 4];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 5] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 5];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 6] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 6];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 7] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 7];
			}
		}

		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(1 * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(1 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 0] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 0];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 1] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 1];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 2] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 2];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 3] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 3];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 4] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 4];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 5] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 5];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 6] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 6];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 7] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 7];
			}
		}

		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(2 * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(2 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 0] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 0];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 1] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 1];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 2] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 2];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 3] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 3];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 4] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 4];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 5] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 5];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 6] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 6];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 7] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 7];
			}
		}

		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(3 * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(3 * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 0] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 0];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 1] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 1];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 2] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 2];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 3] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 3];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 4] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 4];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 5] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 5];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 6] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 6];
				threadResults[(wSubRowIdx * TM) * (WNITER * TN) + wSubColIdx * TN + 7] += regM[wSubRowIdx] * regN[wSubColIdx * TN + 7];
			}
		}

		A += BK;
		B += BK * N;
		__syncthreads();
	}

	for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
					float4 tmp = reinterpret_cast<float4 *>(
						&C_interim[(threadRowInWarp * TM + resIdxM) * N +
								threadColInWarp * TN + resIdxN])[0];
					tmp.x = tmp.x + threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
												wSubColIdx * TN + resIdxN + 0];
					tmp.y = tmp.y + threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
												wSubColIdx * TN + resIdxN + 1];
					tmp.z = tmp.z + threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
												wSubColIdx * TN + resIdxN + 2];
					tmp.w = tmp.w + threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
												wSubColIdx * TN + resIdxN + 3];
					reinterpret_cast<float4 *>(
						&C_interim[(threadRowInWarp * TM + resIdxM) * N +
								threadColInWarp * TN + resIdxN])[0] = tmp;
				}
			}
		}
	}
}
