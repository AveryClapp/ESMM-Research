#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
	one_d_vec(int M, int N, int K, float *A,
					 float *B, float *C) {
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const uint totalResultsBlocktile = BM * BN;
	const uint numThreadsBlocktile = totalResultsBlocktile / (TM);

	assert(numThreadsBlocktile == blockDim.x);

	const int threadCol = threadIdx.x % BN;
	const int threadRow = threadIdx.x / BN;

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	const uint innerRowA = threadIdx.x / (BK / 4);
	const uint innerColA = threadIdx.x % (BK / 4);
  	const uint innerRowB = threadIdx.x / (BN / 4);
	const uint innerColB = threadIdx.x % (BN / 4);

	float threadResults[TM] = {0.0};

	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		if (innerRowA < BM) {
			float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
				
			As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
			As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
			As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
			As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
		}
		if (innerRowB < BK && innerColB * 4 < BN) {
			reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
		}
		__syncthreads();

		A += BK;
		B += BK * N;
			
		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			float bTmp = Bs[dotIdx * BN + threadCol];

			#pragma unroll
			for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
				threadResults[resIdxM] += As[dotIdx * BM + threadRow * TM + resIdxM] * bTmp;
			}
		}
		__syncthreads();
	}

	for (uint resIdx = 0; resIdx < TM; ++resIdx) {
			C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
}
