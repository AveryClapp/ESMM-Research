#include "utils.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include "esmm_buffered.cu"

using std::cout;
using std::endl;
using std::cin;

#define WARPSIZE 32

const uint K11_NUM_THREADS = 128;
const uint K11_BN = 128;
const uint K11_BM = 128;
const uint K11_BK = 16;
const uint K11_WN = 32;
const uint K11_WM = 128;
const uint K11_WNITER = 1;
const uint K11_TN = 8;
const uint K11_TM = 8;


bool run_esmm_buffered(int N, int M, int K, float *d_A, float *d_B,
                    float *d_C, float *h_C, float *h_C_ref, int runs) {


  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // warpsubtile in warptile
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 blockDim(K11_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  // dim3 gridDim(CEIL_DIV(cols, BN), CEIL_DIV(rows, BM));
  // Initialize C to zeros
  cudaMemset(d_C, 0, N * M * sizeof(float));

  for (int i = 0; i < runs; i++) {
    esmm_buffered<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER, K11_TM, K11_TN, K11_NUM_THREADS>
        <<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  return verifyResults(h_C, h_C_ref, M * N);
}

int main(int argc, char *argv[]) {
    // Define Matrix Dims
    constexpr int rows = 1024;
    constexpr int cols = 1024;
    constexpr int inners = 1024;

    // Default values
    std::vector<int> kernel_choices = {6}; // Default to kernel 6
    int runs = 10;


    // Allocate host matrices
    float *h_A = (float *)malloc(rows * inners * sizeof(float));
    float *h_B = (float *)malloc(inners * cols * sizeof(float));
    float *h_C = (float *)malloc(rows * cols * sizeof(float));
    float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

    // Generate random data w/ given sparsity:
    std::vector<int> sparsity = stringToVector("11111111");

    randomize_matrix_with_pattern(h_A, rows, inners, sparsity);
    randomize_matrix(h_B, inners, cols);
    memset(h_C, 0, rows * cols * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, rows * inners * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, inners * cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

    cudaCheckError(cudaMemcpy(d_A, h_A, rows * inners * sizeof(float),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, inners * cols * sizeof(float),
                              cudaMemcpyHostToDevice));

    if (verbose) cout << "Generating CPU reference solution..." << endl;
    matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);

    int passed = 0;
    int total = kernel_choices.size();

    bool result = run_esmm_buffered(rows, cols, inners,
                                       d_A, d_B, d_C, h_C, h_C_ref, runs);

    if (!result) cout << "FAIL!" << endl;
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (passed == total) ? 0 : 1;
}
