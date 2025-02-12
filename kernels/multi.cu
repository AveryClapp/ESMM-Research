#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void esmm_shmem_multi(int rows, int columns, int inners, 
				int blocksize,
				const float *A, const float *B, float *C)
{
	const int row = blockIdx.x * blocksize;
	const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);
	int coloff = col % blocksize;

	extern __shared__ float sArea [];
	float* sA = sArea;  
	float* sB = sArea + blocksize * blocksize; 
	// Create an array to hold (blocksize) number of elements
	float tmpres[32] = {0.0}; 

	for (int inner=0; inner < inners; inner += blocksize)
	{
		for (int dotidx=0; dotidx<blocksize; dotidx++)
		{
			sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
			sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
		}
		__syncthreads();

		for (int i=0; i < blocksize; ++i)
		{
			float Btmp = sB[i * blocksize + coloff];
			for (int dotidx=0; dotidx < blocksize; dotidx++)
			{
				tmpres[dotidx] +=  sA[dotidx * blocksize + i] * Btmp;
			}
		}
		__syncthreads();
	}
	
	//Can unroll this since we have to know the tmpres array size at launch
	//anyways
	for (int dotidx=0; dotidx<blocksize; dotidx++)
	{
		C[(row + dotidx) * columns + col] = tmpres[dotidx];
	}
	return;
}
