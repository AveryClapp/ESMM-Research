#pragma once

/* Fully unrolled SIZE=8 kernel - eliminates sparse_data array indirection */
/* Instead of: for(i=0; i<SIZE; i++) { dotIdx = sparse_data[i]; ... } */
/* We hardcode: dotIdx=0, dotIdx=1, ..., dotIdx=7 */

#include "../../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * UNROLLED SIZE=8: dotIdx values are 0,1,2,3,4,5,6,7 (fully dense)
 * To create smaller versions, remove iterations:
 * - SIZE=6: Keep dotIdx 0,1,2,3,4,5 (remove 6,7)
 * - SIZE=4: Keep dotIdx 0,1,2,3 (remove 4,5,6,7)  
 * - SIZE=2: Keep dotIdx 0,1 (remove 2,3,4,5,6,7)
 * - SIZE=1: Keep dotIdx 0 only (remove 1,2,3,4,5,6,7)
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
		const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_unrolled_8(int M, int N, int K, float *A, float *B, float *C) {
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
		// Load A and B tiles into shared memory
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

		// UNROLLED: Instead of for(sparse_idx=0; sparse_idx<SIZE; sparse_idx++)
		//           We hardcode all 8 iterations with dotIdx = 0,1,2,3,4,5,6,7

		// ============ ITERATION 0: dotIdx = 0 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(0 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(0 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 1: dotIdx = 1 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(1 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(1 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 2: dotIdx = 2 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(2 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(2 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 3: dotIdx = 3 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(3 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(3 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 4: dotIdx = 4 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(4 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(4 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 5: dotIdx = 5 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(5 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(5 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 6: dotIdx = 6 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(6 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(6 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		// ============ ITERATION 7: dotIdx = 7 ============
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[(7 * BM) + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			regN[wSubColIdx * TN + 0] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 0];
			regN[wSubColIdx * TN + 1] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 1];
			regN[wSubColIdx * TN + 2] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 2];
			regN[wSubColIdx * TN + 3] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 3];
			regN[wSubColIdx * TN + 4] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 4];
			regN[wSubColIdx * TN + 5] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 5];
			regN[wSubColIdx * TN + 6] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 6];
			regN[wSubColIdx * TN + 7] = Bs[(7 * BN) + warpCol * WN + 
				wSubColIdx * WSUBN + threadColInWarp * TN + 7];
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[(wSubRowIdx * TM) * (WNITER * TN) + 
						wSubColIdx * TN + resIdxN] += 
						regM[wSubRowIdx] * regN[wSubColIdx * TN + resIdxN];
				}
			}
		}

		A += BK;
		B += BK * N;
		__syncthreads();
	}

	// Write results back to global memory
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


