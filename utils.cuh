#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define PATTERN_LENGTH 8

using std::cout;
using std::endl;


// Forward decl
bool verify_preprocess_a(float* d_A, int rows, int cols, int inners, int runs);
struct PreprocessResult {
  int* d_list;
  int* h_list;
  int totalSize;
  int denseListSize;
  int numBlocks;
};

std::vector<int> parse_kernel_selection(const std::string& input) {
  std::vector<int> kernels;
  if (input == "all") {
    for (int i = 1; i <= 15; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  size_t dash_pos = input.find('-');
  if (dash_pos != std::string::npos) {
    int start = std::stoi(input.substr(0, dash_pos));
    int end = std::stoi(input.substr(dash_pos + 1));
    for (int i = start; i <= end && i <= 15; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  std::stringstream ss(input);
  std::string kernel_str;
  while (std::getline(ss, kernel_str, ',')) {
    int kernel = std::stoi(kernel_str);
    if (kernel >= 1 && kernel <= 15) {
      kernels.push_back(kernel);
    }
  }
  return kernels;
}

const char* get_kernel_name(int kernel_choice) {
  switch (kernel_choice) {
    case 0: return "A/B Preprocessors";
    case 1: return "Naive Implementation";
    case 2: return "Global Memory Coalescing";
    case 3: return "Shared Memory Blocks";
    case 4: return "One Dimensional Blocktiling";
    case 5: return "Two Dimensional Blocktiling";
    case 6: return "Vectorized Memory Accessing";
    case 7: return "1D Vectorized Approach";
    case 8: return "Basic Warptiling";
    case 9: return "1D Warptiling";
    case 10: return "Emergent Sparsity Matrix Multiplication (ESMM)";
    case 11: return "ESMM Warpskipping";
    case 12: return "ESMM Buffered";
    case 13: return "ESMM Offsets";
    case 14: return "ESMM Unrolled";
    case 15: return "cuBLAS";
    default: return "Unknown Kernel";
  }
}

void print_usage(const char* program_name) {
  cout << "Usage: " << program_name << " [kernel_choice] [runs] [options]" << endl;
  cout << "\nPreprocessing Verification:" << endl;
  cout << "  0, --preprocess       Run both A and B preprocessing verification" << endl;
  cout << "  0a, --preprocess-a    Run A matrix preprocessing verification" << endl;
  cout << "  0b, --preprocess-b    Run B matrix preprocessing verification" << endl;
  cout << "  [size] [runs]         Optional: matrix size (default 1024) and runs (default 10)" << endl;
  cout << "\nKernel_choice: " << endl;
  cout << "    Single kernel: 1-15 (run specific kernel)" << endl;
  cout << "    Multiple kernels: \"1,3,5\" (comma-separated, no spaces)" << endl;
  cout << "    Range: \"1-5\" (run kernels 1 through 5)" << endl;
  cout << "    All: \"all\" (run all kernels 1-15)" << endl;
  cout << "  runs: number of runs per kernel (default: 1)" << endl;
  cout << "  Options:" << endl;
  cout << "    --verbose, -v: Enable verbose output" << endl;
  cout << "    --no-check, -n: Skip result verification (performance-only mode)" << endl;
  cout << "    --check-results, -c: Enable result verification (default)" << endl;
  cout << "    --help, -h: Show this help message" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 6 10 --verbose --no-check" << endl;
  cout << "  " << program_name << " 1-5 1 --check-results" << endl;
  cout << "  " << program_name << " all 1 -v -n" << endl;
}


#define cudaCheckError(ans)                                                    \
{                                                                            \
  cudaAssert((ans), __FILE__, __LINE__);                                     \
}
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
        line);
    exit(code);
  }
}

#define SETUP                                                                  \
  auto start = std::chrono::high_resolution_clock::now();                      \
  auto end = std::chrono::high_resolution_clock::now();                        \
  double total_time = 0.0f;
#define START start = std::chrono::high_resolution_clock::now();
#define END                                                                    \
  end = std::chrono::high_resolution_clock::now();                             \
  total_time +=                                                                \
  std::chrono::duration_cast<std::chrono::microseconds>(end - start)       \
  .count();
#define RESULTS(kernel)                                                        \
  std::cout << "Average Speed of Kernel " << kernel << " (" << runs            \
  << " runs): " << std::fixed << std::setprecision(4)                \
  << (total_time / runs) / 1000.0f << " ms" << std::endl;


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line,
        cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  return;
}


void randomize_matrix_with_pattern(float *mat, int M, int N,
    std::string_view pattern) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int pattern_idx = (i * N + j) % PATTERN_LENGTH;
      if (pattern[pattern_idx] == '1') {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i * N + j] = tmp;
      } else {
        mat[i * N + j] = 0.0f;
      }
    }
  }
}

void randomize_matrix(float *mat, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
      tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
      mat[i * N + j] = tmp;
    }
  }
}

std::vector<int> stringToVector(const std::string &str) {
  std::vector<int> vec;
  vec.reserve(str.length());
  for (char c : str) {
    vec.push_back(c - '0');
  }
  return vec;
}

// CPU implementation of matrix multiplication for verification
void matrixMultiplyCPU(const float *A, const float *B, float *C, int M, int N,
    int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Function to verify GPU results against CPU results
bool verifyResults(const float *gpuResults, const float *cpuResults, int size,
    float tolerance = 1e-2) {
  int mismatchCount = 0;
  float maxDiff = 0.0f;
  int firstMismatchIdx = -1;

  for (int i = 0; i < size; i++) {
    float diff = std::abs(gpuResults[i] - cpuResults[i]);
    if (diff > tolerance) {
      mismatchCount++;
      if (firstMismatchIdx == -1)
        firstMismatchIdx = i;
      maxDiff = std::max(maxDiff, diff);
    }
  }

  if (mismatchCount > 0) {
    std::cout << "Verification failed: " << mismatchCount
      << " mismatches out of " << size << " elements." << std::endl;
    std::cout << "Max difference: " << maxDiff << std::endl;
    if (firstMismatchIdx >= 0) {
      int row = firstMismatchIdx / (size / sqrt(size));
      int col = firstMismatchIdx % (int)(size / sqrt(size));
      std::cout << "First mismatch at index " << firstMismatchIdx
        << " (row=" << row << ", col=" << col << "): "
        << "GPU=" << gpuResults[firstMismatchIdx]
        << ", CPU=" << cpuResults[firstMismatchIdx] << std::endl;
    }
    return false;
  }

  return true;
}

std::vector<int> computeExpandedIndices(std::string_view pattern) {
  std::vector<int> indices;
  int patternSize = pattern.size();

  std::vector<int> patternIndices;
  for (int i = 0; i < patternSize; i++) {
    if (pattern[i] == '1') {
      indices.push_back(i);
    }
  }

  return indices;
}

void computeReferencePreprocessing(float* A, int* h_ALIST_ref, int rows, int cols,  int BM, int BK, int WMITER, int WSUBM) {
  const int numKBlocks = cols / BK;
  const int numBlockRows = rows / BM;
  const int MAX_SPARSE_OFFSETS = BK / 2;

  // For each block
  for (int blockRow = 0; blockRow < numBlockRows; blockRow++) {
    for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {

      // For each sub-row within the block
      for (int subRow = 0; subRow < WMITER; subRow++) {

        int count = 0;
        int offsets[BK];

        // Check each dotIdx (column within K-block)
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
          bool hasNonZero = false;

          // Check all 32 rows in this sub-row block
          for (int threadRow = 0; threadRow < 32; threadRow++) {
            int globalRow = blockRow * BM + subRow * WSUBM + threadRow;
            int globalCol = kBlock * BK + dotIdx;

            if (A[globalRow * cols + globalCol] != 0.0f) {
              hasNonZero = true;
              break;
            }
          }

          if (hasNonZero) {
            if (count < MAX_SPARSE_OFFSETS) {
              offsets[count] = dotIdx;
            }
            count++;
          }
        }

        const int blockBase = blockRow * numKBlocks * (BK * WMITER + WMITER);
        const int kBlockBase = blockBase + kBlock * (BK * WMITER + WMITER);
        const int subRowBase = kBlockBase + subRow * (1 + BK);


        if (count > MAX_SPARSE_OFFSETS) {
          h_ALIST_ref[subRowBase] = -1;  // Dense marker
        } else {
          h_ALIST_ref[subRowBase] = count;
          for (int i = 0; i < count; i++) {
            h_ALIST_ref[subRowBase + 1 + i] = offsets[i];
          }
        }
      }
    }
  }
}

bool verifyPreprocessResults(int* h_ALIST, int* h_ALIST_ref, int totalSize) {
  int* gpu = (int*)h_ALIST;
  int* cpu = (int*)h_ALIST_ref;

  bool allMatch = true;
  int errorCount = 0;
  const int MAX_ERRORS_TO_PRINT = 10;
  for (int i = 0; i < totalSize; i++) {
    if (gpu[i] != cpu[i]) {
      if (errorCount < MAX_ERRORS_TO_PRINT) {
        printf("Mismatch at index %d: GPU=%d, CPU=%d\n", 
            i, (int)gpu[i], (int)cpu[i]);
      }
      errorCount++;
      allMatch = false;
    }
  }
  if (allMatch) {
    printf("✓ VERIFICATION PASSED - All values match!\n");
  } else {
    printf("✗ VERIFICATION FAILED - %d mismatches found\n", errorCount);
    if (errorCount > MAX_ERRORS_TO_PRINT) {
      printf("  (showing first %d errors)\n", MAX_ERRORS_TO_PRINT);
    }
  }
  return allMatch;
}

bool handle_preprocessing_commands(int argc, char** argv) {
  if (argc < 2) return false;

  std::string arg = argv[1];
  if (arg != "0" && arg != "0a" && arg != "0b" && 
      arg != "--preprocess" && arg != "--preprocess-a" && arg != "--preprocess-b") {
    return false;
  }

  int size = (argc >= 3) ? atoi(argv[2]) : 1024;
  int runs = (argc >= 4) ? atoi(argv[3]) : 1;
  bool success = false;

  if (arg == "0a" || arg == "--preprocess-a") {
    printf("=== A Matrix Preprocessing Verification ===\n");
    printf("Size: %dx%d", size, size);

    // Setup test matrix
    float *d_A;
    cudaMalloc(&d_A, size * size * sizeof(float));
    float* h_A = (float*)malloc(size * size * sizeof(float));
    randomize_matrix_with_pattern(h_A, size, size, "10000000");
    cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    free(h_A);

    // Run verification (handles everything)
    success = verify_preprocess_a(d_A, size, size, size, runs);

    cudaFree(d_A);
    printf("\n%s\n", success ? "✓ PASSED" : "✗ FAILED");
  }
  else if (arg == "0b" || arg == "--preprocess-b") {
    printf("=== B Matrix Preprocessing ===\n(Not implemented)\n");
  }
  else {
    printf("=== Preprocessing Verification ===\n(Not implemented)\n");
  }

  exit(success ? 0 : 1);
}

//TODO: Replace these with unrolled loops file
__forceinline__ __device__ void multiply_dense(int wSubRowIdx, int wSubColIdx,
    int WNITER, float regM_val, float* regN,
    float* threadResults) {
  const int regNBase = wSubColIdx * 8;
  const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
  threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
  threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
  threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
  threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
  threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
  threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
  threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
  threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_threefourths(int wSubRowIdx, int wSubColIdx,
    int WNITER, float regM_val, float* regN,
    float* threadResults) {
  const int regNBase = wSubColIdx * 8;
  const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
  threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
  threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
  threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
  threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
  threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
  threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_half(int wSubRowIdx, int wSubColIdx,
    int WNITER, float regM_val, float* regN,
    float* threadResults) {
  const int regNBase = wSubColIdx * 8;
  const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
  threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
  threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
  threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
  threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}


__forceinline__ __device__ void multiply_quarter(int wSubRowIdx, int wSubColIdx,
    int WNITER, float regM_val, float* regN,
    float* threadResults) {
  const int regNBase = wSubColIdx * 8;
  const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
  threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
  threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

__forceinline__ __device__ void multiply_eighth(int wSubRowIdx, int wSubColIdx,
    int WNITER, float regM_val, float* regN,
    float* threadResults) {
  const int regNBase = wSubColIdx * 8;
  const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
  threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
}

