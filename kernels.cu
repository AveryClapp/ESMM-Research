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

// New method to calculate blocks of C (each thread goes row(A) x col(B) for one
// element). The problem is that b is (8x32),
// TODO try one shared memory variable with offsetting to see if that results in
// increased performance
__global__ void sequential(float* A, float* B, float* C, int blocksize) {
	extern __shared__ float shared_A[];
	extern __shared__ float shared_B[];	    

	const int row = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
	const int col = blockIdx.y * blockDim.x + (threadIdx.x % B_COLS);
	
	// Load elements of A into SMEM, the motivation here is that since A_COLS !=
	// B_COLS, we need to find a quick way to load A elements into SMEM. This
	// can be done by having each thread add 4 elements into SMEM [col, col+3]
	shared_A[threadIdx.x * 4 + 0] = A[row * A_COLS + col + 0];
	shared_A[threadIdx.x * 4 + 1] = A[row * A_COLS + col + 1];
	shared_A[threadIdx.x * 4 + 2] = A[row * A_COLS + col + 2];
	shared_A[threadIdx.x * 4 + 3] = A[row * A_COLS + col + 3];

	// Stop GPU until all threads are done adding to SMEM
	__syncthreads();

	// Load elements of B into SMEM.
	shared_B[threadIdx.x] = B[(threadIdx.x * B_COLS) + col];

	// Stop GPU until all threads are done adding to SMEM
	__syncthreads();

	//TODO modify this for a discrepancy in column sizes b/w A and B
	float tmp = 0.0;
	for (int i=0; i < inners; ++i) {
		tmp += A[row * inners + i] * B[i * columns + col]; 
	}
	C[row * columns + col] = tmp;
}
