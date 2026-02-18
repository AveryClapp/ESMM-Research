#pragma once

#include "../../include/utils.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WARPSIZE 32

// Device function type for pattern-based computation
// Computes FMAs based on sparsity pattern
using PatternComputeFunc = void (*)(float regM_val, const float* regN, float* threadResults, int regNBase, int threadResBase);

// Device functions for each pattern case - these are non-template for simplicity
__device__ void compute_pattern_0x80(float regM_val, const float* regN, float* threadResults, int regNBase, int threadResBase) {
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
}

__device__ void compute_pattern_0xC0(float regM_val, const float* regN, float* threadResults, int regNBase, int threadResBase) {
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

__device__ void compute_pattern_0xF0(float regM_val, const float* regN, float* threadResults, int regNBase, int threadResBase) {
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__device__ void compute_pattern_default(float regM_val, const float* regN, float* threadResults, int regNBase, int threadResBase) {
	// Do nothing for unimplemented patterns
}

// Lookup table: maps all 256 possible pattern bytes to function pointers
__device__ __constant__ PatternComputeFunc pattern_lut[256] = {
	// 0x00 - 0x7F: default (no pattern or unimplemented)
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,

	// 0x80: 1 bit set (sparse pattern)
	compute_pattern_0x80, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,

	// 0xC0: 2 bits set (sparse pattern)
	compute_pattern_0xC0, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,

	// 0xF0: 4 bits set (dense pattern)
	compute_pattern_0xF0, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default,
	compute_pattern_default, compute_pattern_default, compute_pattern_default, compute_pattern_default
};

template <const int BM, const int BN, const int BK, const int WM,
			const int WN, const int WNITER, const int TM, const int TN,
			const int NUM_THREADS, const int SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
	esmm_b_fp_lut(int M, int N, int K, float *A, float *B, float *C,
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

	const uint8_t pattern = *b_offsets;


for (int32_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
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
	#pragma unroll
	for (int8_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
		#pragma unroll
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			regM[wSubRowIdx] = As[dotIdx * BM + warpRow * WM +
				wSubRowIdx * WSUBM + threadRowInWarp * TM];
		}
		#pragma unroll
		for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
			for (uint offset = 0; offset < TN; ++offset) {
				regN[wSubColIdx * TN + offset] = Bs[(dotIdx * BN) + warpCol *
					WN + wSubColIdx * WSUBN + threadColInWarp * TN + offset];
			}
		}
		for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
			for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
				const int regNBase = wSubColIdx * TN;
				const int threadResBase = wSubRowIdx * (WNITER * TN) + (wSubColIdx * TN);
				// Use LUT to dispatch to the appropriate pattern function
				pattern_lut[pattern](regM[wSubRowIdx], regN, threadResults, regNBase, threadResBase);
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

