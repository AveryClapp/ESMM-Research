#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include "../utils.cuh"
/*
	Next Steps 
	- Kernel Profiling
	- 2d blocktiling
	- autotuning params of block size (we know B will be 32x8)
*/

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

		float* currA = baseA + blockIndex;
		float* currB = baseB + blockIndex * N;
		for (int dotIdx = 0; dotIdx < blockInner; ++dotIdx) {
			float tmpB = currB[dotIdx * N + localCol]; 
			for (int resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += currA[(localRow * TM + resIdx) * innerDim + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}
	//TODO potential for unrolling
	for (int resIdx = 0; resIdx < TM; ++resIdx) {
		baseC[(localRow * TM + resIdx) * N + localCol] = threadResults[resIdx];
	}
}
