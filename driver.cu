#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.cuh"
#include "./kernels/basic.cu"
#include "./kernels/gmem_coalesce.cu"
#include "./kernels/smem_blocking.cu"
#include "./kernels/1D_Blocktiling.cu"
#include "./kernels/1D_Blocktiling.cu"
#include "./kernels/2D_Blocktiling.cu"
#include "./kernels/vectorized_blocktiling.cu"
#include "./kernels/warptiling.cu"
#include <chrono>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

#define SETUP \
	auto start = std::chrono::high_resolution_clock::now(); \
	auto end = std::chrono::high_resolution_clock::now(); \
	double total_time = 0.0f;
#define START start = std::chrono::high_resolution_clock::now();
#define END \
	end = std::chrono::high_resolution_clock::now(); \
	total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#define RESULTS(kernel) \
	std::cout << "Average Speed of Kernel " << kernel << " (" << runs << " runs): "\
	<< std::fixed << std::setprecision(4) \
	<< (total_time / runs) / 1000.0f << " ms" << std::endl;

/* Simple function to iterate over a kernel and get data for a specific (or all
 * of them) */
void collect_data(int runs, int kernel, int rows, int cols, int inners, int blocksize, float* d_A, float* d_B, float* d_C, float* h_C, float* h_C_cpu) {
	// Integer corresponds to the version of multi
	SETUP
	switch (kernel) {
		case 1: {
			dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
			dim3 blockDim(32, 32);
			for (int i = 0; i < runs; i++) {
				START
				basic<<<gridDim, blockDim>>>(rows,cols,inners,d_A,d_B,d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("Naive");
			break;
		}
		case 2: {
			dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
			dim3 blockDim(32, 32);
			for (int i = 0; i < runs; i++) {
				START
				gmem_coalesce<32><<<gridDim, blockDim>>>(rows,cols,inners,d_A,d_B,d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("GMEM Coalescing");		
			break;
		}
		case 3: {
			dim3 gridDim(CEIL_DIV(cols, 32), CEIL_DIV(rows, 32));
			dim3 blockDim(32, 32);
			for (int i = 0; i < runs; i++) {
				START
				smem_blocking<32><<<gridDim, blockDim>>>(rows,cols,inners,d_A,d_B,d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("SMEM Blocking");		
			break;
		}
		case 4: {
			// 1D Blocktiling
			constexpr int BM = 64;
			constexpr int BN = 64;
			constexpr int BK = 8;
			constexpr int TM = 8;
			dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
			dim3 blockDim(BN * BM / TM);
			for (int i = 0; i < runs; i++) {
				START
				one_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("1D Blocktiling")
			break;
		}
		case 5: {
			// 2D Blocktiling
			constexpr int BM = 128;
			constexpr int BN = 128;
			constexpr int BK = 8;
			constexpr int TM = 8;
			constexpr int TN = 8;
			dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
			dim3 blockDim(BM * BN / (TM * TN));
			for (int i = 0; i < runs; i++) {
				START
				two_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("2D Blocktiling")
			break;
		}
		case 6: {
			// Vectorized Blocktiling
			constexpr int BM = 128;
			constexpr int BN = 128;
			constexpr int BK = 8;
			constexpr int TM = 8;
			constexpr int TN = 8;
			dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
			dim3 blockDim(BM * BN / (TM * TN));
			for (int i = 0; i < runs; i++) {
				START
				vectorized_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("Vectorized Blocktiling")
			break;
		}
		case 7: {
			const uint K10_NUM_THREADS = 128;
			const uint K10_BN = 128;
			const uint K10_BM = 128;
			const uint K10_BK = 16;
			const uint K10_WN = 64;
			const uint K10_WM = 64;
			const uint K10_WNITER = 4;
			const uint K10_TN = 4;
			const uint K10_TM = 8;

  			dim3 blockDim(K10_NUM_THREADS);
    		constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

			constexpr uint K10_WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
			
			dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
			for (int i = 0; i < runs; i++) {
				START
				warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS> <<<gridDim, blockDim>>>(cols, rows, inners, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
			}
			RESULTS("Warptiling")
			break;

		}
		case 8: {
			cublasHandle_t handle;
    		cublasCreate(&handle);
	   		float alpha = 1.0f;
		    float beta = 0.0f;

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, &alpha, d_B, cols, d_A, inners, &beta, d_C, cols);
				
			for (int i = 0; i < runs; i++) {
				START
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, &alpha, d_B, cols, d_A, inners, &beta, d_C, cols);
				END
				cudaDeviceSynchronize();
				cudaCheckError(cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
				cudaMemset(d_C, 0, rows * cols * sizeof(float));
			}
			RESULTS("cuBLAS")	
			cublasDestroy(handle);	
			break;
		}
		default:
			// Run all kernels
			collect_data(runs, 1, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 2, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 3, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 4, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 5, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 6, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 7, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
			collect_data(runs, 8, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);
	}
}

int main() {
		// Setup 
		constexpr int rows = 1024;
		constexpr int cols = 1024;
		constexpr int inners = 1024;
		constexpr int blocksize = 32;
		// Allocate host matrices
		float *h_A = (float*)malloc(rows * cols * sizeof(float));
		float *h_B = (float*)malloc(rows * cols * sizeof(float));
		float *h_C = (float*)malloc(rows * cols * sizeof(float));
		float *h_C_cpu = (float*)malloc(rows * cols * sizeof(float));

		// Generate random data
		randomize_matrix(h_A, rows, cols);
		randomize_matrix(h_B, rows, cols);

		// Allocate device matrices
		float *d_A, *d_B, *d_C;
		cudaCheckError(cudaMalloc(&d_A, rows * cols * sizeof(float)));
		cudaCheckError(cudaMalloc(&d_B, rows * cols * sizeof(float)));
		cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

		// Copy random data to device matrices
		cudaCheckError(cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(d_B, h_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

		matrixMultiplyCPU(h_A, h_B, h_C_cpu, rows, cols);
		// Run kernels
		/* 0 - all
		 * 1 - Naive
		 * 2 - GMEM Coalescing
		 * 3 - SMEM Blocking
		 * 4 - 1d
		 * 5 - 2d
		 * 6 - vectorized
		 * 7 - warptiling
		 * 8 - cuBLAS
		*/
		collect_data(1, 0, rows, cols, inners, blocksize, d_A, d_B, d_C, h_C, h_C_cpu);


		free(h_A);
		free(h_B);
		free(h_C);
		free(h_C_cpu);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		return 0;
}

/*
		case 1: {
			for (int i = 0; i < runs; i++) {
				START
				esmm_shmem_multi<<<dim3(CEIL_DIV(rows, blocksize), CEIL_DIV(cols, blocksize)), dim3(blocksize), blocksize * blocksize * 2 * sizeof(float)>>>(rows, cols, inners, blocksize, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
				cudaMemset(d_C, 0, rows * cols * sizeof(float));
			}
			RESULTS("Multi")
			break;
		}
		case 2: {
			for (int i = 0; i < runs; i++) {
				START
				esmm_shmem_multi2<<<dim3(CEIL_DIV(rows, blocksize), CEIL_DIV(cols, blocksize)), dim3(blocksize), blocksize * blocksize * 2 * sizeof(float)>>>(rows, cols, inners, blocksize, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
				cudaMemset(d_C, 0, rows * cols * sizeof(float));
				}
			RESULTS("Multi2")
			break;
		}
		case 3: {
			for (int i = 0; i < runs; i++) {
				START
				esmm_shmem_multi3<<<dim3(CEIL_DIV(rows, blocksize), CEIL_DIV(cols, blocksize)), dim3(blocksize), blocksize * blocksize * 2 * sizeof(float)>>>(rows, cols, inners, blocksize, d_A, d_B, d_C);
				END
				cudaDeviceSynchronize();
				cudaMemset(d_C, 0, rows * cols * sizeof(float));
			}
			RESULTS("Multi3")
			break;
		}
collect_data(runs, 1, rows, cols, inners, blocksize, d_A, d_B, d_C);
			collect_data(runs, 2, rows, cols, inners, blocksize, d_A, d_B, d_C);
			collect_data(runs, 3, rows, cols, inners, blocksize, d_A, d_B, d_C);

*/
