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

// Matrix multiplication on tiled rectangles
__global__ void sequential_rectangles(float* A, float* B, float* C, int xDim, int yDim, int innerDim) {
	const int row = blockIdx.x * xDim + (threadIdx.x / yDim);
	const int col = blockIdx.y * yDim + (threadIdx.x % yDim);	
	// Perform dot product matrix multiplication on this
	float tmp = 0.0;
	for (int i = 0; i < innerDim; i++) {
		tmp += A[row * innerDim + i] * B[i * xDim + col]; 
	}
	C[row * yDim + col] = tmp;
}

//TODO WIP.
__global__ void sequential_smem(float* A, float* B, float* C) {
	extern __shared__ float SMEM[];
	float *sA = SMEM;
	float *sB = SMEM + (A_COLS * A_ROWS);

	const int row = blockIdx.x * blockDim.x + (threadIdx.x / A_COLS);
	const int col = blockIdx.y * blockDim.y + (threadIdx.x % B_COLS);
	
	sA[threadIdx.x] = A[row * A_COLS + col];
	__syncthreads();

	sB[threadIdx.x] = B[(threadIdx.x / B_COLS) * B_COLS + col];
	__syncthreads();

	float tmp = 0.0;
	for (int i=0; i < A_COLS; ++i) {
		tmp += sA[row * A_COLS + i] * sB[i * B_COLS + col]; 
	}

	C[row * C_COLS + col] = tmp;

	return;
}

/*
// rectangular 2,4
dim3 gridDim24(2,4);
dim3 blockDim24(16,8);

// rectangular 4,2
dim3 gridDim42(4,2);
dim3 blockDim42(8,16);
rblksz = blockDim.x
cblksz = blockDim.y
 */
__global__ void esmm_sequential_ns (int rows, int columns, int inners, 
									int rblksz, int cblksz, 
														const float *A, const float *B, float *C)
{
// change iteration order to output sequentially
	const int row = blockIdx.x * rblksz + (threadIdx.x / cblksz);
	const int col = blockIdx.y * cblksz + (threadIdx.x % cblksz);

	float tmp = 0.0;
	for (int i=0; i < inners; ++i)
	{
		tmp += A[row * inners + i] * B[i * columns + col]; 
	}
	C[row * columns + col] = tmp;
}


