#include "runners.cuh"
#include "utils.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using std::cout;
using std::endl;
using std::cin;


int main(int argc, char *argv[]) {
  // Define Matrix Dims
  constexpr int rows = 1024;
  constexpr int cols = 1024;
  constexpr int inners = 1024;
  int kernel_choice = 6;
  int runs = 1;

  // Parse command line arguments
  if (argc > 1) {
    kernel_choice = atoi(argv[1]);
  }

  if (argc > 2) {
    runs = atoi(argv[2]);
  }

  // Allocate host matrices
  float *h_A = (float *)malloc(rows * inners * sizeof(float));
  float *h_B = (float *)malloc(inners * cols * sizeof(float));
  float *h_C = (float *)malloc(rows * cols * sizeof(float));
  float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

  // Generate random data w/ given sparsity:
  std::vector<int> sparsity = stringToVector("10101010");

  // Generate A matrix
  randomize_matrix_with_pattern(h_A, rows, inners, sparsity);
  // Generate B matrix
  randomize_matrix(h_B, inners, cols);
  // Set h_C to zeros
  memset(h_C, 0, rows * cols * sizeof(float));

  // Allocate device matrices
  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc(&d_A, rows * inners * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_B, inners * cols * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

  // Copy random data to device matrices
  cudaCheckError(cudaMemcpy(d_A, h_A, rows * inners * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, inners * cols * sizeof(float),
                            cudaMemcpyHostToDevice));

  // Generate reference solution on CPU
  matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);

  // Initialize d_C to zeros
  cudaCheckError(cudaMemset(d_C, 0, rows * cols * sizeof(float)));

  // Determine the result of the matrix multiplication
  bool res = false;

  // Choose kernel based on input (in order of blogpost)
  switch (kernel_choice) {
  case 1: // Naive Implementation
    res = run_naive(rows, cols, inners, d_A, d_B, d_C, runs);
    if (print_res) cout << "Naive kernel status: " << res << endl;
    break;
  case 2: // Global Memory Coalescing
    res = run_gmem_coalesce(rows, cols, inners, d_A, d_B, d_C, runs);
    if (print_res) cout << "GMEM kernel status: " << res << endl;
    break;
  case 3: // Shared Memory Blocks
    res = run_smem_blocking(rows, cols, inners, d_A, d_B, d_C, runs);
    if (print_res) cout << "SMEM kernel status: " << res << endl;
    break;
  case 4: // One Dimensional Blocktiling
    res = run_one_blocktiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs) << std::endl;
    if (print_res) cout << "1D Blocktiling kernel status: " << res << endl;
    break;
  case 5: // Two Dimensional Blocktiling
    res = run_two_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
    if (print_res) cout << "2D Blocktiling kernel status: " << res << endl;
    break;
  case 6: // Vectorized Memory Accessing
    res = run_vectorized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "Vectorized kernel status: " << res << endl;
    break;
  case 7: // 1 Dimensional Vectorized Approach
    res = run_1d_vec(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "1D vectorized status: " << res << endl;
    break;
  case 8: // Basic Warptiling
    res = run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "Warptiling kernel status: " << res << endl;
    break;
  case 9: // 1-Dimensional Warptiling
    res = run_1d_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "1D Warptiling kernel status: " << res << endl;
    break;
  case 10: // Emergent Sparsity Matrix Multiplication (our kernel)
    res = run_esmm(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "ESMM kernel status: " << res << endl;
    break;
  case 11: // Experimental warpskipping approach to ESMM
    res = run_esmm_warpskipping(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
    if (print_res) cout << "ESMM Warpskipping kernel status: " << res << endl;
    break;
  case 12: // cuBlas
    run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
    break;
  default:
    std::cout << "Invalid kernel choice." << "\n";
    break;
  }

  // Clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
