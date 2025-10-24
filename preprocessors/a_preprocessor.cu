#pragma once

/* Preprocessor for A matrix to encode horizontal sparsity */

#define MAX_DENSE 5 

#include "utils.cuh"
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	preprocess_A(int M, int N, int K, float *A, int *A_CTS, int* A_LIST) {
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint warpIdx = threadIdx.x / WARPSIZE;
	const uint warpCol = warpIdx % (BN / WN);
	const uint warpRow = warpIdx / (BN / WN);

	constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
	constexpr uint WSUBM = WM / WMITER;

	// Target 50% or lower sparsity
	constexpr uint MAX_SPARSE_OFFSETS = BK / 2;

	const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
	const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); 
	const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); 

	__shared__ float As[BN * BK];
	// Enough space to encode BK + 1 elements for each 32x8 block
	__shared__ int8_t denseList[(K / BK) * ((BK / 2) * WMITER + (1 * WMITER))];

	A += cRow * BM * K;

	const uint innerRowA = threadIdx.x / (BK / 4);
	const uint innerColA = threadIdx.x % (BK / 4);
	constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
	

	for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
		for (int32_t offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
			const float4 tmp = reinterpret_cast<const float4 *>(
				&A[(innerRowA + offset) * K + innerColA * 4])[0];
			As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
		}

		__syncthreads();

		int laneId = threadIdx.x % WARPSIZE;
		// Traverse 32x8 blocks and accumulate sparsity
		for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				regM[wSubRowIdx] = As[(dotIdx * BM) + warpRow * WM +
					wSubRowIdx * WSUBM + threadRowInWarp];
			}
			for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
				short active = static_cast<short>(__ballot_sync(0xFFFFFFFF, regM[wSubRowIdx]) > 0);
				if (active && laneId == 0) {
					const uint kBlockBase = (bkIdx / BK) * (BK * WMITER + WMITER);
					const uint countIdx = kBlockBase + wSubRowIdx * (1 + BK);
					uint8_t currentCount = denseList[countIdx];
					if (currentCount < MAX_SPARSE_OFFSETS) {
						const uint offsetIdx = countIdx + 1 + currentCount;
						denseList[offsetIdx] = dotIdx;
						denseList[countIdx]++;
					} else {
						denseList[countIdx] = -1;
						break;

					}
				}
			}
		}
		A += BK;
		__syncthreads();
	}

	// TODO:
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

