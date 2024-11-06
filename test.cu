
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
      if (code != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
                        exit(code);
                            }
}

// Matrix dimensions
#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
      int row = blockIdx.y * blockDim.y + threadIdx.y;
          int col = blockIdx.x * blockDim.x + threadIdx.x;
              
              if (row < M && col < K) {
                        float sum = 0.0f;
                                for (int i = 0; i < N; i++) {
                                              sum += A[row * N + i] * B[i * K + col];
                                                      }
                                        C[row * K + col] = sum;
                                            }
}

// CPU matrix multiplication for verification
void matrixMultiplyCPU(float* A, float* B, float* C, int M, int N, int K) {
      for (int row = 0; row < M; row++) {
                for (int col = 0; col < K; col++) {
                              float sum = 0.0f;
                                          for (int i = 0; i < N; i++) {
                                                            sum += A[row * N + i] * B[i * K + col];
                                                                        }
                                                      C[row * K + col] = sum;
                                                              }
                    }
}

// Initialize matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
      for (int i = 0; i < rows * cols; i++) {
                matrix[i] = (float)(rand() % 100) / 100.0f;
                    }
}

// Verify results
bool verifyResults(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-5) {
      for (int i = 0; i < size; i++) {
                if (fabs(gpuResult[i] - cpuResult[i]) > tolerance) {
                              printf("Mismatch at position %d: GPU = %f, CPU = %f\n", 
                                                     i, gpuResult[i], cpuResult[i]);
                                          return false;
                                                  }
                    }
          return true;
}

int main() {
      // Matrix dimensions
      int M = 1024;  // rows of A
          int N = 1024;  // cols of A and rows of B
              int K = 1024;  // cols of B

                  // Allocate host memory
                  float *h_A = (float*)malloc(M * N * sizeof(float));
                      float *h_B = (float*)malloc(N * K * sizeof(float));
                          float *h_C = (float*)malloc(M * K * sizeof(float));
                              float *h_C_cpu = (float*)malloc(M * K * sizeof(float));

                                  // Initialize matrices
                                  initializeMatrix(h_A, M, N);
                                      initializeMatrix(h_B, N, K);

                                          // Allocate device memory
                                          float *d_A, *d_B, *d_C;
                                              cudaCheckError(cudaMalloc(&d_A, M * N * sizeof(float)));
                                                  cudaCheckError(cudaMalloc(&d_B, N * K * sizeof(float)));
                                                      cudaCheckError(cudaMalloc(&d_C, M * K * sizeof(float)));

                                                          // Copy data to device
                                                          cudaCheckError(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
                                                              cudaCheckError(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));

                                                                  // Set up grid and block dimensions
                                                                  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
                                                                      dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                                                                                           (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

                                                                          // Launch kernel
                                                                          matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                                                                              cudaCheckError(cudaGetLastError());
                                                                                  cudaCheckError(cudaDeviceSynchronize());

                                                                                      // Copy result back to host
                                                                                      cudaCheckError(cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));

                                                                                          // Compute CPU result
                                                                                          matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, N, K);

                                                                                              // Verify results
                                                                                              bool correct = verifyResults(h_C, h_C_cpu, M * K);
                                                                                                  printf("Matrix multiplication %s\n", correct ? "PASSED" : "FAILED");

                                                                                                      // Free memory
                                                                                                      free(h_A);
                                                                                                          free(h_B);
                                                                                                              free(h_C);
                                                                                                                  free(h_C_cpu);
                                                                                                                      cudaFree(d_A);
                                                                                                                          cudaFree(d_B);
                                                                                                                              cudaFree(d_C);

                                                                                                                        return 0;
}
