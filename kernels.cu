#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.cuh"


/* ------------------ OLD KERNELS ------------------------- */
// Original Method for a 32x32 block of B -- OUT OF DATE
__global__ void basic32(float* A, float* B, float* C) {
	// Each thread handles one row of B
	int b_row = threadIdx.x;
	int row = blockIdx.y * blockDim.x + blockIdx.x;
	if (b_row < B_ROWS) {
		float a_element = A[row * A_COLS + b_row];
		for (int j = 0; j < B_COLS; j++) {
			int b_col = (threadIdx.x + j) % B_COLS;
			C[blockIdx.x * C_COLS + b_col] += a_element * B[b_row * B_COLS + b_col];
		}
	}
}

// Original Method for a 8x8 block of B, impossible to eleminate atomicAdd
// here (pigeonhole principle)
__global__ void basic8(float* A, float* B, float* C) {
	int b_row = threadIdx.x;
	if (b_row < B_ROWS) {
		float b_elements[8] = {
			B[b_row * B_COLS + 0],
			B[b_row * B_COLS + 1],
			B[b_row * B_COLS + 2],
			B[b_row * B_COLS + 3],
			B[b_row * B_COLS + 4],
			B[b_row * B_COLS + 5],
			B[b_row * B_COLS + 6],
			B[b_row * B_COLS + 7]
		};
		float a_element = A[blockIdx.x * A_COLS + b_row];
		atomicAdd(&C[blockIdx.x * C_COLS + 0], a_element * b_elements[0]);
		atomicAdd(&C[blockIdx.x * C_COLS + 1], a_element * b_elements[1]);
		atomicAdd(&C[blockIdx.x * C_COLS + 2], a_element * b_elements[2]);
		atomicAdd(&C[blockIdx.x * C_COLS + 3], a_element * b_elements[3]);
		atomicAdd(&C[blockIdx.x * C_COLS + 4], a_element * b_elements[4]);
		atomicAdd(&C[blockIdx.x * C_COLS + 5], a_element * b_elements[5]);
		atomicAdd(&C[blockIdx.x * C_COLS + 6], a_element * b_elements[6]);
		atomicAdd(&C[blockIdx.x * C_COLS + 7], a_element * b_elements[7]);
	}
}

/* ---------------------- NEW KERNELS --------------------------------------- */


template <const int blockHeight, const int blockWidth, const int blockInner, const int TM>
__global__ void matMulBlocktiling(float* A, float* B, float* C, int N, int innerDim) {
	const int col = blockIdx.x;
	const int row = blockIdx.y;

	// Designated col and row for each thread
	const int localCol = threadIdx.x % blockWidth; // Assign cols sequentially
	const int localRow = threadIdx.x / blockWidth; // Assign rows in batches

	A += row * innerDim * blockHeight; 
	B += col * blockWidth;
	C += row * innerDim * blockHeight + col * blockWidth;

	float threadResults[TM] = {0.0};

	//Iterate over column/row
	//128 % 8 == 0, so this is fine for the time being
	for (int blockIndex = 0; blockIndex < innerDim; blockIndex += blockInner) {
	    __syncthreads();
		
		A += blockInner;
		B += blockInner * N;

		for (int dotIdx = 0; dotIdx < blockInner; ++dotIdx) {
			float tmpB = B[dotIdx * blockWidth + localCol];
			for (uint resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += 
					A[(localRow * TM + resIdx) * blockInner + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}
	for (int resIdx = 0; resIdx < TM; ++resIdx) {
		C[(localRow * TM + resIdx) * N + localCol] = threadResults[resIdx];
	}
}


template <const int blockHeight, const int blockWidth, const int blockInner, const int TM>
__global__ void matMulBlocktilingTwo(float* A, float* B, float* C, int N, int innerDim) {
		const int col = blockIdx.x;
		const int row = blockIdx.y;

		// Thread indexing remains the same
		const int localCol = threadIdx.x % blockWidth;
		const int localRow = threadIdx.x / blockWidth;

		// Store base pointers
		float* baseA = A + row * innerDim * blockHeight;
		float* baseB = B + col * blockWidth;
		float* baseC = C + row * N * blockHeight + col * blockWidth;

		float threadResults[TM] = {0.0f};

		// Iterate over blocks
		for (int blockIndex = 0; blockIndex < innerDim; blockIndex += blockInner) {
				__syncthreads();

				// Calculate
				// current
				// block
				// pointers
				float* currA = baseA + blockIndex;
				float* currB = baseB + blockIndex * N;

				// Compute
				// dot
				// products
				for (int dotIdx = 0; dotIdx < blockInner; ++dotIdx) {
						float tmpB = currB[dotIdx * N + localCol];  // Fixed
																	// B
																	// indexing

						for (uint resIdx = 0; resIdx < TM; ++resIdx) {
								float tmpA = currA[(localRow * TM + resIdx) * innerDim + dotIdx];  // Fixed
																								   // A
																								   // indexing
								threadResults[resIdx] += tmpA * tmpB;
						}
				}
				__syncthreads();
		}

		// Write results
		for (int resIdx = 0; resIdx < TM; ++resIdx) {
				baseC[(localRow * TM + resIdx) * N + localCol] = threadResults[resIdx];
		}
}
