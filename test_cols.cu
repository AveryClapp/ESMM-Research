#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call)                                                        \
  do {                                                                        \
    cudaError_t error = call;                                               \
    if (error != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n",      \
          cudaGetErrorString(error), error, __FILE__, __LINE__);      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                       \
  } while (0)

// Matrix dimensions
const int A_ROWS = 1024;
const int A_COLS = 32;
const int B_ROWS = 32;
const int B_COLS = 8;
const int C_ROWS = A_ROWS;
const int C_COLS = B_COLS;

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int A_COLS, int B_COLS, int C_COLS) {
  // Calculate the row index this block is responsible for
  int row = blockIdx.x;

  // Each thread within the block handles one column of C
  int col = threadIdx.x;

  // Initialize the sum for C[row][col]
  float sum = 0.0f;

  // Iterate over the k-dimension (0 to A_COLS - 1)
  for(int k = 0; k < A_COLS; ++k) {
    float a_val = A[row * A_COLS + k];      // A[row][k]
    float b_val = B[k * B_COLS + col];      // B[k][col]
    sum += a_val * b_val;                    // Accumulate the product
  }

  // Write the computed sum to C[row][col]
  C[row * C_COLS + col] = sum;
}

int main() {
  // Host memory allocation
  size_t size_A = A_ROWS * A_COLS * sizeof(float);
  size_t size_B = B_ROWS * B_COLS * sizeof(float);
  size_t size_C = C_ROWS * C_COLS * sizeof(float);

  float *h_A = (float*)malloc(size_A);
  float *h_B = (float*)malloc(size_B);
  float *h_C = (float*)malloc(size_C);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrices.\n");
    exit(EXIT_FAILURE);
  }

  // Initialize matrices A and B with 1.0f
  for(int i = 0; i < A_ROWS * A_COLS; i++) {
    h_A[i] = 1.0f;
  }

  for(int i = 0; i < B_ROWS * B_COLS; i++) {
    h_B[i] = 1.0f;
  }

  // Initialize matrix C to 0.0f
  for(int i = 0; i < C_ROWS * C_COLS; i++) {
    h_C[i] = 0.0f;
  }

  // Device memory allocation
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

  // Copy matrices A and B to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
  // Initialize device matrix C to 0.0f
  CUDA_CHECK(cudaMemset(d_C, 0, size_C));

  // Define grid and block dimensions
  dim3 gridDim(C_ROWS, 1, 1);     // 1024 blocks, one per row of A/C
  dim3 blockDim(C_COLS, 1, 1);    // 8 threads per block, one per column of C

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Record the start event
  CUDA_CHECK(cudaEventRecord(start, 0));

  // Launch the kernel
  matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, A_COLS, B_COLS, C_COLS);

  // Check for any errors launching the kernel
  CUDA_CHECK(cudaGetLastError());

  // Record the stop event
  CUDA_CHECK(cudaEventRecord(stop, 0));

  // Wait for the stop event to complete
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate the elapsed time between the two events
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  // Copy result matrix C back to host
  CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  // Verification: Since A and B are filled with 1.0f, each element in C should be 32.0f
  bool correct = true;
  for(int i = 0; i < C_ROWS * C_COLS; i++) {
    if(h_C[i] != 32.0f) { // Since A and B are filled with 1.0f
      correct = false;
      printf("Error at index %d: Expected 32.0, Got %f\n", i, h_C[i]);
      break;
    }
  }

  if(correct) {
    printf("Matrix multiplication is correct.\n");
  } else {
    printf("Matrix multiplication is incorrect.\n");
  }

  // Print the elapsed time
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Destroy CUDA events
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Free device memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  // Reset the device and exit
  CUDA_CHECK(cudaDeviceReset());

  return 0;
}

