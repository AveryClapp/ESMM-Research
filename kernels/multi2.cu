#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void esmm_shmem_multi2 (int rows, int columns, int inners, 
				int blocksize,
				const float *A, const float *B, float *C)
{
		// 1-d array of threads
		const int row = blockIdx.x * blocksize;
		const int col = blockIdx.y * blocksize + (threadIdx.x % blocksize);

		int coloff = col % blocksize;

		extern __shared__ float sArea [];
		float* sA = sArea;  
		float* sB = sArea + blocksize * blocksize; 

		// RBTODO need to make dynamic
		float tmpres[8] = {0.0}; // thread results

		// for a block of A and B
		for (int inner=0; inner < inners; inner += blocksize)
		{
				// each
				// thread
				// loads MT
				// elements
				for (int dotidx=0; dotidx<blocksize; dotidx++)
				{
						// Load
						// lock
						// of
						// A
						// and
						// B
						// into
						// shared
						// memory
						sA[dotidx * blocksize + coloff] = A[(row + dotidx) * inners + inner + coloff];
						sB[dotidx * blocksize + coloff] = B[(inner + dotidx) * columns + col];
				}
				__syncthreads();

				for (int dotidx=0; dotidx < blocksize; dotidx++)
				{
						for (int i=0; i < blocksize; ++i)
						{
								tmpres[dotidx] +=  sA[dotidx * blocksize + i] * sB[i * blocksize + coloff];
						}
				}
				__syncthreads();
		}

		// each thread loads blocksize
		// elements
		for (int dotidx=0; dotidx<blocksize; dotidx++)
		{
				C[(row + dotidx) * columns + col] = tmpres[dotidx];
		}
		return;
}
