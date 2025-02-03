#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include "../utils.cuh"

template <const int blockHeight, const int blockWidth, const int blockInner, const int TM>
__global__ void matMulBlocktiling(float* A, float* B, float* C, int N, int innerDim) {
	const int col = blockIdx.x;
	const int row = blockIdx.y;

	const int localCol = threadIdx.x % blockWidth;
	const int localRow = threadIdx.x / blockWidth;

	float* baseA = A + row * innerDim * blockHeight;
	float* baseB = B + col * blockWidth;
	float* baseC = C + row * N * blockHeight + col * blockWidth;

	float threadResults[TM] = {0.0f};

	__shared__ float As[blockHeight * blockInner];
	__shared__ float Bs[blockInner * blockWidth];

	assert(blockHeight * blockInner == blockDim.x);
	assert(blockWidth * blockInner == blockDim.x);
	const uint innerColA = threadIdx.x % blockInner; 
	const uint innerRowA = threadIdx.x / blockInner;
	const uint innerColB = threadIdx.x % blockWidth;
	const uint innerRowB = threadIdx.x / blockWidth;
	
		// Iterate over blocks
	for (int blockIndex = 0; blockIndex < innerDim; blockIndex += blockInner) {
		As[innerRowA * blockInner + innerColA] = A[innerRowA * innerDim + innerColA];
		Bs[innerRowB * blockWidth + innerColB] = B[innerRowB * N + innerColB];
		__syncthreads();

		float* currA = baseA + blockIndex;
		float* currB = baseB + blockIndex * N;
		for (int dotIdx = 0; dotIdx < blockInner; ++dotIdx) {
			float tmpB = Bs[dotIdx * blockWidth + localCol];
			for (int resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += As[(localRow * TM + resIdx) * blockInner + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}
	for (int resIdx = 0; resIdx < TM; ++resIdx) {
		baseC[(localRow * TM + resIdx) * N + localCol] = threadResults[resIdx];
	}
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, const float *A, const float *B, float *C) {
	const uint cRow = blockIdx.y;
	const uint cCol = blockIdx.x;

	const int threadCol = threadIdx.x % BN;
	const int threadRow = threadIdx.x / BN;

	__shared__ float As[BM * BK];
	__shared__ float Bs[BK * BN];

	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	assert(BM * BK == blockDim.x);
	assert(BN * BK == blockDim.x);
	const uint innerColA = threadIdx.x % BK;
	const uint innerRowA = threadIdx.x / BK;
	const uint innerColB = threadIdx.x % BN; 
	const uint innerRowB = threadIdx.x / BN;

	float threadResults[TM] = {0.0};

	for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
		As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
		Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
		__syncthreads();

		A += BK;
		B += BK * N;

		for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
			float tmpB = Bs[dotIdx * BN + threadCol];
			for (uint resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}

	// write out the results
	for (uint resIdx = 0; resIdx < TM; ++resIdx) {
		C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
}
