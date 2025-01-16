#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.cuh"

// Original Method for a 32x32 block of B
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

__global__ void matrixMultiplyKernel(float* A, float* B, float* C) {
	// Each thread handles one row of B
	int b_row = threadIdx.x;
	if (b_row < B_ROWS) {
		// TODO memory coalescing
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
		// How do we not use atomic add here?
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
