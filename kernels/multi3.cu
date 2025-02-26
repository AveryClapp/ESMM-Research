#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

/* Device functions to handle 8 element long computations */
__device__ void full(int dotidx, int i, int blocksize, int coloff, float* tmpres, float* sA, float* sB) {
	tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+1)] * sB[(i+1) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+2)] * sB[(i+2) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+3)] * sB[(i+3) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+4)] * sB[(i+4) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+5)] * sB[(i+5) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+6)] * sB[(i+6) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+7)] * sB[(i+7) * blocksize + coloff];
}

__device__ void half_sparse(int dotidx, int i, int blocksize, int coloff, float* tmpres, float* sA, float* sB) {
	tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+1)] * sB[(i+1) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+2)] * sB[(i+2) * blocksize + coloff];
	tmpres[dotidx] +=  sA[dotidx * blocksize + (i+3)] * sB[(i+3) * blocksize + coloff];
}

__device__ void one(int dotidx, int i, int blocksize, int coloff, float* tmpres, float* sA, float* sB) {
	tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
}

__global__ void esmm_shmem_multi3 (int rows, int columns, int inners, 
				int blocksize,
				const float *A, const float *B, float *C)
{
	const int row = blockIdx.x * blocksize;
	const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

	int coloff = col % blocksize;

	extern __shared__ float sArea [];
	float* sA = sArea;  
	float* sB = sArea + blocksize * blocksize; 

	float tmpres[32] = {0.0}; // thread results

	for (int inner=0; inner < inners; inner += blocksize)
	{
		for (int dotidx=0; dotidx < blocksize; dotidx++)
		{
			sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
			sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
		}
		__syncthreads();

		for (int dotidx=0; dotidx < blocksize; dotidx++)
		{
			for (int i=0; i < blocksize; i += blocksize/4)
			{
				full(dotidx,i,blocksize,coloff,tmpres,sA,sB);
			}
		}
		__syncthreads();
	}

	for (int dotidx=0; dotidx<blocksize; dotidx++)
	{
		C[(row + dotidx) * columns + col] = tmpres[dotidx];
	}
	return;
}


