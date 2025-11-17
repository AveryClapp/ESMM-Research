#pragma once

#include "preprocess_params.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define PATTERN_LENGTH 8

using std::cout;
using std::endl;


// Forward decl - removed with old preprocessors
// bool verify_preprocess_a(float* d_A, int rows, int cols, int inners, int runs, bool check);
struct PreprocessResult {
  int* d_list;
  int* h_list;
  int totalSize;
};

std::vector<int> parse_kernel_selection(const std::string& input) {
  std::vector<int> kernels;
  if (input == "all") {
    for (int i = 1; i <= 26; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  size_t dash_pos = input.find('-');
  if (dash_pos != std::string::npos) {
    int start = std::stoi(input.substr(0, dash_pos));
    int end = std::stoi(input.substr(dash_pos + 1));
    for (int i = start; i <= end && i <= 26; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  std::stringstream ss(input);
  std::string kernel_str;
  while (std::getline(ss, kernel_str, ',')) {
    int kernel = std::stoi(kernel_str);
    if (kernel >= 1 && kernel <= 26) {
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
    case 16: return "ESMM Pattern-Specialized (Zero Overhead)";
    case 17: return "ESMM Block-wise Uniform (Per-Warp Pattern Encoding)";
    case 18: return "ESMM Combined A+B Sparsity";
    case 19: return "ESMM Joint A+B Sparsity (B-Transpose + Intersection)";
    case 20: return "ESMM B-Transpose (Warp-Uniform B-Sparsity)";
    case 21: return "ESMM A+B Offset Lists (8x8 Templated)";
    case 22: return "ESMM B-Transpose + 8x8 Offset Templates (Joint K-Sparsity)";
    case 23: return "ESMM Joint Precomputed (Zero-Overhead Joint A+B Sparsity)";
    case 24: return "ESMM Joint Shared Memory Pattern Cache";
    case 25: return "ESMM Block-wise Uniform (Large Tiles BM=256)";
    case 26: return "ESMM Block-wise Uniform (Square Tiles BM=BN=256)";
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
  cout << "    Single kernel: 1-24 (run specific kernel)" << endl;
  cout << "    Multiple kernels: \"1,3,5\" (comma-separated, no spaces)" << endl;
  cout << "    Range: \"1-5\" (run kernels 1 through 5)" << endl;
  cout << "    All: \"all\" (run all kernels 1-24)" << endl;
  cout << "  runs: number of runs per kernel (default: 1)" << endl;
  cout << "  Options:" << endl;
  cout << "    --verbose, -v: Enable verbose output" << endl;
  cout << "    --no-check, -n: Skip result verification (performance-only mode)" << endl;
  cout << "    --check-results, -c: Enable result verification (default)" << endl;
  cout << "    --random, -r: Use random unstructured sparsity (default: pattern-based)" << endl;
  cout << "    --help, -h: Show this help message" << endl;
  cout << endl;
  cout << "Sparsity Modes:" << endl;
  cout << "  Pattern-based (default): Uses repeating pattern \"11110000\" (50% sparsity)" << endl;
  cout << "  Random (--random, -r): Uses truly random unstructured sparsity (50% default)" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 6 10 --verbose --no-check" << endl;
  cout << "  " << program_name << " 1-5 1 --check-results" << endl;
  cout << "  " << program_name << " all 1 -v -n" << endl;
  cout << "  " << program_name << " 13 1 --random --verbose  # Test with random sparsity" << endl;
  cout << "  " << program_name << " 10 5 -r -v  # Short form with random sparsity" << endl;
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
      int pattern_idx = (j) % PATTERN_LENGTH;
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

// Generate sparsity along K-dimension (rows) for B matrix
// This allows skipping entire K-blocks when rows are zero
void randomize_matrix_B_kdim_pattern(float *mat, int K, int N,
    std::string_view pattern) {
  for (int k = 0; k < K; k++) {
    int pattern_idx = k % PATTERN_LENGTH;  // Pattern along K (rows)
    bool row_is_zero = (pattern[pattern_idx] == '0');

    for (int n = 0; n < N; n++) {
      if (row_is_zero) {
        mat[k * N + n] = 0.0f;  // Entire row zero
      } else {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[k * N + n] = tmp;
      }
    }
  }
}

// Generate sparsity along N-dimension (columns) for B matrix
// This allows skipping specific columns within TN blocks
void randomize_matrix_B_ndim_pattern(float *mat, int K, int N,
    std::string_view pattern) {
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      int pattern_idx = n % PATTERN_LENGTH;  // Pattern along N (columns)
      bool col_is_zero = (pattern[pattern_idx] == '0');

      if (col_is_zero) {
        mat[k * N + n] = 0.0f;  // Column zero
      } else {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[k * N + n] = tmp;
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

// ============================================================================
// UNSTRUCTURED RANDOM SPARSITY GENERATION
// ============================================================================

/*
 * Generate matrix with completely random unstructured sparsity
 *
 * Each element has an independent probability of being zero.
 * No repeating patterns - truly random sparsity distribution.
 *
 * Parameters:
 *   mat: Output matrix (M×N)
 *   M, N: Matrix dimensions
 *   sparsity_percent: Target sparsity percentage (0-100)
 *                     50.0 = 50% of elements are zero
 *   seed: Random seed for reproducibility (optional, 0 = use time)
 */
void randomize_matrix_unstructured(float *mat, int M, int N,
                                   float sparsity_percent,
                                   unsigned int seed = 0) {
  // Set random seed for reproducibility
  if (seed != 0) {
    srand(seed);
  }

  const float sparsity_threshold = sparsity_percent / 100.0f;
  int zero_count = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      // Random probability for each element
      float prob = (float)rand() / (float)RAND_MAX;

      if (prob < sparsity_threshold) {
        // Element is zero (sparse)
        mat[i * N + j] = 0.0f;
        zero_count++;
      } else {
        // Element is non-zero (dense)
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i * N + j] = tmp;
      }
    }
  }

  // Print actual sparsity achieved
  float actual_sparsity = 100.0f * zero_count / (M * N);
  if (fabs(actual_sparsity - sparsity_percent) > 2.0f) {
    printf("  Note: Requested %.1f%% sparsity, achieved %.2f%% (%d/%d zeros)\n",
           sparsity_percent, actual_sparsity, zero_count, M * N);
  }
}

/*
 * Generate A matrix with unstructured random sparsity in K-dimension
 *
 * For A (M×K): Each element has independent probability of being zero.
 * Good for testing joint sparsity with random patterns.
 *
 * Parameters:
 *   mat: Output matrix (M×K)
 *   M, K: Matrix dimensions
 *   sparsity_percent: Target sparsity percentage (0-100)
 *   seed: Random seed for reproducibility
 */
void randomize_matrix_A_unstructured(float *mat, int M, int K,
                                     float sparsity_percent,
                                     unsigned int seed = 0) {
  randomize_matrix_unstructured(mat, M, K, sparsity_percent, seed);
  printf("Generated A (M×K) with %.1f%% unstructured sparsity\n", sparsity_percent);
}

/*
 * Generate B matrix with unstructured random sparsity in K-dimension
 *
 * For B (K×N): Each element has independent probability of being zero.
 * K-dimension sparsity allows joint sparsity exploitation with A.
 *
 * Parameters:
 *   mat: Output matrix (K×N)
 *   K, N: Matrix dimensions
 *   sparsity_percent: Target sparsity percentage (0-100)
 *   seed: Random seed for reproducibility
 */
void randomize_matrix_B_unstructured(float *mat, int K, int N,
                                     float sparsity_percent,
                                     unsigned int seed = 0) {
  randomize_matrix_unstructured(mat, K, N, sparsity_percent, seed);
  printf("Generated B (K×N) with %.1f%% unstructured sparsity\n", sparsity_percent);
}

/*
 * Generate BOTH A and B with independent unstructured random sparsity
 *
 * Creates truly random sparsity patterns for joint A+B testing.
 * Each matrix gets independent random sparsity.
 *
 * Expected joint density: (1-sparsity_A) × (1-sparsity_B)
 * Example: 50% × 50% = 25% joint density → 75% joint sparsity
 *
 * Parameters:
 *   mat_A: Output A matrix (M×K)
 *   mat_B: Output B matrix (K×N)
 *   M, N, K: Matrix dimensions
 *   sparsity_A: Target sparsity for A (0-100)
 *   sparsity_B: Target sparsity for B (0-100)
 *   seed: Random seed for reproducibility
 */
void randomize_matrices_joint_unstructured(float *mat_A, float *mat_B,
                                           int M, int N, int K,
                                           float sparsity_A,
                                           float sparsity_B,
                                           unsigned int seed = 0) {
  printf("Generating unstructured random sparsity:\n");

  // Generate A with independent random sparsity
  randomize_matrix_A_unstructured(mat_A, M, K, sparsity_A, seed);

  // Generate B with independent random sparsity (different seed)
  randomize_matrix_B_unstructured(mat_B, K, N, sparsity_B, seed ? seed + 1 : 0);

  // Calculate expected joint sparsity
  float density_A = (100.0f - sparsity_A) / 100.0f;
  float density_B = (100.0f - sparsity_B) / 100.0f;
  float joint_density = density_A * density_B;
  float joint_sparsity = 100.0f * (1.0f - joint_density);

  printf("Expected joint sparsity: %.1f%% × %.1f%% = %.2f%% joint density (%.2f%% joint sparsity)\n",
         sparsity_A, sparsity_B, joint_density * 100.0f, joint_sparsity);
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

uint8_t computeExpandedIndicesBits(std::string_view pattern) {
  std::vector<int> indices;
  int patternSize = pattern.size();
  // 00000000
  uint8_t resp = 0;
  std::vector<int> patternIndices;
  for (int i = 0; i < patternSize; i++) {
    if (pattern[i] == '1') {
      resp <<= 1;
    }
  }

  return resp ^ 0xFF;
}
void computeReferencePreprocessing(float* A, int* h_ALIST_ref, int rows, int cols,  int BM, int BK, int WMITER, int WSUBM) {
  using P = PreprocessParams;

  const int numKBlocks = cols / P::BK;
  const int numBlockRows = rows / P::BM;

  // Total number of masks and ints
  const int numMasks = numKBlocks * P::NUM_WARP_ROWS * P::WMITER;
  const int numInts = (numMasks + 3) / 4;

  // Temporary buffer for masks
  uint8_t* masks = new uint8_t[numMasks];
  memset(masks, 0, numMasks);

  for (int blockRow = 0; blockRow < numBlockRows; blockRow++) {
    for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
      for (int warpRow = 0; warpRow < P::NUM_WARP_ROWS; warpRow++) {
        for (int subRow = 0; subRow < P::WMITER; subRow++) {
          uint8_t mask = 0;

          // Check each column in the K-block
          for (int dotIdx = 0; dotIdx < P::BK; dotIdx++) {
            bool hasNonZero = false;
            // Check all 32 rows for this warp sub-row
            for (int threadRow = 0; threadRow < 32; threadRow++) {
              int globalRow = blockRow * P::BM + warpRow * (P::BM / P::NUM_WARP_ROWS) + subRow * P::WSUBM + threadRow;
              int globalCol = kBlock * P::BK + dotIdx;

              if (A[globalRow * cols + globalCol] != 0.0f) {
                hasNonZero = true;
                break;
              }
            }

            // Set bit if any value in this column is non-zero
            if (hasNonZero) {
              mask |= (1 << dotIdx);
            }
          }

          // Store mask in temporary buffer
          const int maskIdx = kBlock * P::NUM_WARP_ROWS * P::WMITER + warpRow * P::WMITER + subRow;
          masks[maskIdx] = mask;
        }
      }
    }

    // Pack masks into ints (4 masks per int)
    const int blockOffset = blockRow * numInts;
    for (int i = 0; i < numInts; i++) {
      int packed = 0;
      for (int j = 0; j < 4 && (i * 4 + j) < numMasks; j++) {
        packed |= (masks[i * 4 + j] << (j * 8));
      }
      h_ALIST_ref[blockOffset + i] = packed;
    }
  }

  delete[] masks;
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
          i, gpu[i], cpu[i]);
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

// Removed - old preprocessor verification (K16/K17/K19 deleted)
/*
bool handle_preprocessing_commands(int argc, char** argv, int size, std::string_view sparsity) {
  std::string arg = argv[1];
  bool success = false;

  if (arg == "0a" || arg == "--preprocess-a") {
    printf("=== A Matrix Preprocessing Verification ===\n");
    printf("Size: %dx%d\n", size, size);
    constexpr int runs = 1;
    float *d_A;
    cudaMalloc(&d_A, size * size * sizeof(float));
    float* h_A = (float*)malloc(size * size * sizeof(float));
    randomize_matrix_with_pattern(h_A, size, size, sparsity);
    cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    free(h_A);
    bool verify = !(argc > 2 && strcmp(argv[2], "-n") == 0);
    success = verify_preprocess_a(d_A, size, size, size, runs, verify);

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
*/


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

