// sparse_matmul_updated.cu

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstdlib>
#include <fstream>
#include <cuda_runtime.h>

#define ROWS_A 4096 
#define COLS_A 32
#define ROWS_B COLS_A // 32
#define COLS_B 8
#define ROWS_C ROWS_A // 1024
#define COLS_C COLS_B // 8

// CUDA Kernels

// Regular Matrix Multiplication Kernel
__global__ void matMulRegular(const float *A, const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < ROWS_A && col < COLS_B) {
    float sum = 0.0f;
    for (int k = 0; k < COLS_A; ++k) {
      sum += A[row * COLS_A + k] * B[k * COLS_B + col];
    }
    C[row * COLS_B + col] = sum;
  }
}

// Kernel for sparsity pattern "10101010"
__global__ void matMul_10101010(const float *A, const float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < ROWS_C) {
    for (int col = 0; col < COLS_C; col += 2) { 
      float sum = 0.0f;
      for (int k = 0; k < COLS_A; ++k) {
        sum += A[row * COLS_A + k] * B[k * COLS_B + col];
      }
      C[row * COLS_C + col] = sum;
    }
  }
}

// Kernel for sparsity pattern "00100000"
__global__ void matMul_00100000(const float *A, const float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = 2; 
  if (row < ROWS_C) {
    float sum = 0.0f;
    for (int k = 0; k < COLS_A; ++k) {
      sum += A[row * COLS_A + k] * B[k * COLS_B + col];
    }
    C[row * COLS_C + col] = sum;
  }
}


__global__ void matMulDense(const float *A, const float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < ROWS_C) {
    for (int col = 0; col < COLS_C; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < COLS_A; ++k) {
        sum += A[row * COLS_A + k] * B[k * COLS_B + col];
      }
      C[row * COLS_C + col] = sum;
    }
  }
}

// Half-Dense Kernel (Pattern "01010101")
__global__ void matMul_01010101(const float *A, const float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < ROWS_C) {
    for (int col = 1; col < COLS_C; col += 2) { // Non-zero columns at odd indices
      float sum = 0.0f;
      for (int k = 0; k < COLS_A; ++k) {
        sum += A[row * COLS_A + k] * B[k * COLS_B + col];
      }
      C[row * COLS_C + col] = sum;
    }
  }
}

// Sparse Kernel (Pattern "00010000")
__global__ void matMul_00010000(const float *A, const float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = 3; // Only column 3 is non-zero
  if (row < ROWS_C) {
    float sum = 0.0f;
    for (int k = 0; k < COLS_A; ++k) {
      sum += A[row * COLS_A + k] * B[k * COLS_B + col];
    }
    C[row * COLS_C + col] = sum;
  }
}

// Initialize Matrix A with random values
void initializeMatrixA(float *h_A, int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
  }
}

// Initialize Matrix B based on the pattern
void initializeB(float *h_B, int rows, int cols, const std::string &pattern, std::vector<int> &nzColumns) {
  memset(h_B, 0, rows * cols * sizeof(float));

  nzColumns.clear();

  if (pattern == "dense" || pattern == "regular") {
    for (int col = 0; col < cols; ++col) {
      nzColumns.push_back(col);
      for (int row = 0; row < rows; ++row) {
        h_B[row * cols + col] = 1.0f; 
      }
    }
  } else if (pattern == "01010101") {
    for (int col = 0; col < cols; ++col) {
      if (col % 2 == 1) {
        nzColumns.push_back(col);
        for (int row = 0; row < rows; ++row) {
          h_B[row * cols + col] = 1.0f;
        }
      }
    }
  } else if (pattern == "10101010") {
    for (int col = 0; col < cols; ++col) {
      if (col % 2 == 0) {
        nzColumns.push_back(col);
        for (int row = 0; row < rows; ++row) {
          h_B[row * cols + col] = 1.0f; 
        }
      }
    }
  } else if (pattern == "00010000") {
    int nzCol = 3;
    nzColumns.push_back(nzCol);
    for (int row = 0; row < rows; ++row) {
      h_B[row * cols + nzCol] = 1.0f; 
    }
  } else if (pattern == "00100000") {
    int nzCol = 2;
    nzColumns.push_back(nzCol);
    for (int row = 0; row < rows; ++row) {
      h_B[row * cols + nzCol] = 1.0f; 
    }
  } else {
    std::cerr << "Unknown pattern: " << pattern << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Verify the results
void verifyResults(const float *A, const float *B, const float *C, const std::vector<int> &nzColumns) {
  bool correct = true;
  for (int row = 0; row < ROWS_C; ++row) {
    for (int col : nzColumns) {
      float sum = 0.0f;
      for (int k = 0; k < COLS_A; ++k) {
        sum += A[row * COLS_A + k] * B[k * COLS_B + col];
      }
      float diff = fabs(C[row * COLS_C + col] - sum);
      if (diff > 1e-4) {
        correct = false;
        std::cerr << "Mismatch at row " << row << ", col " << col << ": GPU result " << C[row * COLS_C + col]
          << ", CPU result " << sum << std::endl;
        break;
      }
    }
    if (!correct) break;
  }
  if (!correct) {
    std::cout << "Results verified: INCORRECT" << std::endl;
  }
}

// Main Function
int main() {
  srand(0); // For reproducibility

  std::vector<std::string> patterns = {"dense", "01010101", "10101010", "00010000", "00100000"};

  // Host matrices
  float *h_A = new float[ROWS_A * COLS_A];
  float *h_B = new float[ROWS_B * COLS_B];
  float *h_C = new float[ROWS_C * COLS_C];
  float *h_C_experimental = new float[ROWS_C * COLS_C];
  float *h_C_regular = new float[ROWS_C * COLS_C];

  // Map patterns to kernels
  typedef void (*MatMulKernel)(const float *, const float *, float *);
  std::map<std::string, MatMulKernel> kernelMap = {
    {"dense", matMulDense},
    {"01010101", matMul_01010101},
    {"10101010", matMul_10101010},
    {"00010000", matMul_00010000},
    {"00100000", matMul_00100000}
  };

  // Open CSV file for writing performance data
  std::ofstream csvFile("performance_data.csv");
  csvFile << "Pattern,ExecutionTime(ms)" << std::endl;

  for (size_t p = 0; p < patterns.size(); ++p) {
    cudaFree(0);
    std::string pattern = patterns[p];
    std::vector<int> nzColumns;

    initializeMatrixA(h_A, ROWS_A, COLS_A);
    initializeB(h_B, ROWS_B, COLS_B, pattern, nzColumns);

    size_t sizeA = ROWS_A * COLS_A * sizeof(float);
    size_t sizeB = ROWS_B * COLS_B * sizeof(float);
    size_t sizeC = ROWS_C * COLS_C * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    MatMulKernel selectedKernel = kernelMap[pattern];


    dim3 blockDim(256);
    dim3 gridDim((ROWS_C + blockDim.x - 1) / blockDim.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    selectedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float experimentalTime = 0;
    cudaEventElapsedTime(&experimentalTime, start, stop);

    cudaMemcpy(h_C_experimental, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Pattern: " << pattern << std::endl;
    verifyResults(h_A, h_B, h_C_experimental, nzColumns);
    
    cudaEventRecord(start);
    matMulRegular<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float regularTime = 0;
    cudaEventElapsedTime(&regularTime, start, stop);

    cudaMemcpy(h_C_regular, d_C, sizeC, cudaMemcpyDeviceToHost);
    verifyResults(h_A, h_B, h_C_regular, nzColumns);

    std::cout << "Experimental Kernel Time: " << experimentalTime << " ms" << std::endl;
    std::cout << "Regular Kernel Time: " << regularTime << " ms" << std::endl << std::endl;
    csvFile << pattern << "," << experimentalTime << "," << regularTime << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  csvFile.close();

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_experimental;
  delete[] h_C_regular;

  return 0;
}

