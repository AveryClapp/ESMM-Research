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

std::vector<int> parse_kernel_selection(const std::string& input) {
  std::vector<int> kernels;
  if (input == "all") {
    for (int i = 1; i <= 30; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  size_t dash_pos = input.find('-');
  if (dash_pos != std::string::npos) {
    int start = std::stoi(input.substr(0, dash_pos));
    int end = std::stoi(input.substr(dash_pos + 1));
    for (int i = start; i <= end && i <= 30; i++) {
      kernels.push_back(i);
    }
    return kernels;
  }
  std::stringstream ss(input);
  std::string kernel_str;
  while (std::getline(ss, kernel_str, ',')) {
    int kernel = std::stoi(kernel_str);
    if (kernel >= 1 && kernel <= 30) {
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
    case 13: return "ESMM B-Sparse Function Pointer LUT";
    case 14: return "ESMM A-Sparse Unrolled (Pattern-Based Offset Skipping)";
    case 15: return "cuBLAS";
    case 16: return "ESMM A-Sparse Block-wise (Warp-Granularity Patterns)";
    case 17: return "ESMM B-Sparse Warp-Granularity (32-col, Zero-Divergence)";
    case 18: return "ESMM B-Sparse TN-Granularity (8-col, Per Thread-Group)";
    case 19: return "ESMM B-Sparse Warp-Uniform Pattern (WN-granularity, Zero-Divergence)";
    case 20: return "ESMM A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop, K21 Style)";
    case 21: return "ESMM A+B Sparse - 8x32 GRANULARITY";
    case 22: return "ESMM A+B Sparse - 32x32 GRANULARITY";
    case 23: return "ESMM A Sparse - Block-wise Skipping";
    case 24: return "ESMM A+B Fused (Persistent Pattern Extraction)";
    case 25: return "ESMM A+B Simple Fused (Preprocessing + K20)";
    case 26: return "ESMM A+B 8x32 Double-Buffered";
    case 27: return "ESMM A+B Fused Pipeline (8x32, No A Preprocess)";
    case 28: return "ESMM A+B Fused Pipeline BRANCHLESS (No per-dotIdx branch)";
    case 29: return "ESMM A+B 16x8 Granularity (16-row A-pattern)";
    case 30: return "ESMM A+B 8x8 Granularity (SpInfer-style 8-row A-pattern)";
    default: return "Unknown Kernel";
  }
}

void print_usage(const char* program_name) {
  cout << "Usage: " << program_name << " [kernel_choice] [runs] [options]" << endl;
  cout << "\nKernel_choice: " << endl;
  cout << "    Single kernel: 1-30 (run specific kernel)" << endl;
  cout << "    Multiple kernels: \"1,3,5\" (comma-separated, no spaces)" << endl;
  cout << "    Range: \"1-5\" (run kernels 1 through 5)" << endl;
  cout << "    All: \"all\" (run all kernels 1-25)" << endl;
  cout << "  runs: number of runs per kernel (default: 1)" << endl;
  cout << "  Options:" << endl;
  cout << "    --verbose, -v: Enable verbose output" << endl;
  cout << "    --no-check, -n: Skip result verification (performance-only mode)" << endl;
  cout << "    --check-results, -c: Enable result verification (default)" << endl;
  cout << "    --random, -r: Use random unstructured sparsity (default: pattern-based)" << endl;
  cout << "    --blockwise, -b: Use block-level sparsity (warp-uniform patterns at BK=8 granularity)" << endl;
  cout << "    --size <N>, -s <N>: Set matrix dimensions to NxNxN (default: 4096)" << endl;
  cout << "    --pattern <P>, -p <P>: Set 8-bit sparsity pattern for both A and B (default: \"11000000\")" << endl;
  cout << "    --pattern-a <P>, -pa <P>: Set 8-bit sparsity pattern for A matrix only" << endl;
  cout << "    --pattern-b <P>, -pb <P>: Set 8-bit sparsity pattern for B matrix only" << endl;
  cout << "    --help, -h: Show this help message" << endl;
  cout << endl;
  cout << "Sparsity Modes:" << endl;
  cout << "  Pattern-based (default): Uses repeating 8-bit pattern \"11000000\" (25% sparsity)" << endl;
  cout << "  Random (--random, -r): Uses truly random unstructured sparsity (37.5% default)" << endl;
  cout << "  Block-level (--blockwise, -b): Block-level sparsity with warp-uniform patterns" << endl;
  cout << "    - Each BK=8 block is fully dense or fully zero" << endl;
  cout << "    - All warps share same K-block pattern (zero divergence)" << endl;
  cout << "    - Ideal for joint A+B sparsity experiments (K22-27)" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << program_name << " 6 10 --verbose --no-check" << endl;
  cout << "  " << program_name << " 1-5 1 --check-results" << endl;
  cout << "  " << program_name << " all 1 -v -n" << endl;
  cout << "  " << program_name << " 13 1 --random --verbose  # Test with random sparsity" << endl;
  cout << "  " << program_name << " 10 5 -r -v  # Short form with random sparsity" << endl;
  cout << "  " << program_name << " 17 1 --size 2048 --pattern 10101010  # Custom size and pattern" << endl;
  cout << "  " << program_name << " 22 5 -s 8192 -p 11110000 -v  # 50% sparsity both A&B, 8192x8192" << endl;
  cout << "  " << program_name << " 24 1 -b -pa 11100000 -pb 11110000 -v  # 62.5% A, 50% B block-level sparsity" << endl;
  cout << "  " << program_name << " 24-27 1 -b --pattern-a 11111000 --pattern-b 11000000 -v  # 37.5% A, 25% B" << endl;
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
}


std::vector<int> stringToVector(const std::string &str) {
  std::vector<int> vec;
  vec.reserve(str.length());
  for (char c : str) {
    vec.push_back(c - '0');
  }
  return vec;
}


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
  uint8_t resp = 0;
  int patternSize = pattern.size();

  // Convert binary string to byte (MSB first)
  // Example: "10000000" -> 0x80, "11000000" -> 0xC0, "11110000" -> 0xF0
  for (int i = 0; i < patternSize && i < 8; i++) {
    if (pattern[i] == '1') {
      resp |= (1 << (7 - i));
    }
  }

  return resp;
}



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


template <int BK = 8, int WM = 64>
void randomize_matrix_A_blocklevel(float *mat, int M, int K,
                                          float block_sparsity_percent,
                                          unsigned int seed = 0) {
    if (seed != 0) srand(seed);
    
    const int numKBlocks = K / BK;
    const int numMBlocks = M / WM;
    const float sparsity_threshold = block_sparsity_percent / 100.0f;

    for (int mBlock = 0; mBlock < numMBlocks; mBlock++) {
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            uint8_t tile_pattern = 0;
            for (int bit = 0; bit < BK; bit++) {
                if ((float)rand() / RAND_MAX >= sparsity_threshold) {
                    tile_pattern |= (1 << bit);
                }
            }
            
            for (int row = 0; row < WM; row++) {
                const int globalM = mBlock * WM + row;
                if (globalM >= M) break;
                
                for (int k = 0; k < BK; k++) {
                    const int globalK = kBlock * BK + k;
                    if (globalK >= K) break;
                    
                    const bool is_dense = (tile_pattern & (1 << k));
                    if (is_dense) {
                        float val = (float)(rand() % 5) + 0.01f * (rand() % 5);
                        mat[globalM * K + globalK] = (rand() % 2) ? val : -val;
                    } else {
                        mat[globalM * K + globalK] = 0.0f;
                    }
                }
            }
        }
    }
}

template <int BK = 8, int WN = 32>
void randomize_matrix_B_blocklevel_fixed(float *mat, int K, int N,
                                          float block_sparsity_percent,
                                          unsigned int seed = 0) {
    if (seed != 0) srand(seed);

    const int numKBlocks = K / BK;
    const int numNBlocks = N / WN;
    const float sparsity_threshold = block_sparsity_percent / 100.0f;

    std::vector<uint8_t> k_patterns(numKBlocks);
    for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
        uint8_t pattern = 0;
        for (int bit = 0; bit < BK; bit++) {
            if ((float)rand() / RAND_MAX >= sparsity_threshold) {
                pattern |= (1 << bit);
            }
        }
        k_patterns[kBlock] = pattern;
    }

    for (int nBlock = 0; nBlock < numNBlocks; nBlock++) {
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            const uint8_t tile_pattern = k_patterns[kBlock];  // SAME for all N-blocks

            for (int col = 0; col < WN; col++) {
                const int globalN = nBlock * WN + col;
                if (globalN >= N) break;
                for (int k = 0; k < BK; k++) {
                    const int globalK = kBlock * BK + k;
                    if (globalK >= K) break;

                    const bool is_dense = (tile_pattern & (1 << k));
                    if (is_dense) {
                        float val = (float)(rand() % 5) + 0.01f * (rand() % 5);
                        mat[globalK * N + globalN] = (rand() % 2) ? val : -val;
                    } else {
                        mat[globalK * N + globalN] = 0.0f;
                    }
                }
            }
        }
    }
}

/*
 * Generate A matrix with 8-row granularity random sparsity
 */
template <int BK = 8, int TILE_M = 8>
void randomize_matrix_A(float *mat, int M, int K,
                              float block_sparsity_percent,
                              unsigned int seed = 0) {
    if (seed != 0) srand(seed);

    const int numKBlocks = K / BK;
    const int numTileRows = M / TILE_M;
    const float sparsity_threshold = block_sparsity_percent / 100.0f;

    for (int tileRow = 0; tileRow < numTileRows; tileRow++) {
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            // Generate independent random pattern for EACH 8-row tile
            uint8_t tile_pattern = 0;
            for (int bit = 0; bit < BK; bit++) {
                if ((float)rand() / RAND_MAX >= sparsity_threshold) {
                    tile_pattern |= (1 << bit);
                }
            }

            // Apply pattern to these 8 rows
            for (int row = 0; row < TILE_M; row++) {
                const int globalM = tileRow * TILE_M + row;
                if (globalM >= M) break;

                for (int k = 0; k < BK; k++) {
                    const int globalK = kBlock * BK + k;
                    if (globalK >= K) break;

                    const bool is_dense = (tile_pattern & (1 << k));
                    if (is_dense) {
                        float val = (float)(rand() % 5) + 0.01f * (rand() % 5);
                        mat[globalM * K + globalK] = (rand() % 2) ? val : -val;
                    } else {
                        mat[globalM * K + globalK] = 0.0f;
                    }
                }
            }
        }
    }
}



/*
 * Generate A matrix with FIXED PATTERN block-level sparsity
 *
 * Unlike randomize_matrix_A_blocklevel which generates random patterns,
 * this function applies the SAME user-specified pattern to ALL K-blocks.
 */
template <int BK = 8, int WM = 32>
void randomize_matrix_A_blocklevel_pattern(float *mat, int M, int K,
                                           std::string_view pattern,
                                           unsigned int seed = 0) {
    if (seed != 0) srand(seed);

    // Convert pattern string to uint8_t bitmask
    uint8_t fixed_pattern = computeExpandedIndicesBits(pattern);

    const int numKBlocks = K / BK;
    const int numMBlocks = M / WM;

    for (int mBlock = 0; mBlock < numMBlocks; mBlock++) {
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            // Use FIXED pattern for ALL K-blocks (not random)
            // Apply pattern to all WM rows in this M-block
            for (int row = 0; row < WM; row++) {
                const int globalM = mBlock * WM + row;
                if (globalM >= M) break;

                for (int k = 0; k < BK; k++) {
                    const int globalK = kBlock * BK + k;
                    if (globalK >= K) break;

                    const bool is_dense = (fixed_pattern & (1 << (7 - k)));  // MSB first
                    if (is_dense) {
                        float val = (float)(rand() % 5) + 0.01f * (rand() % 5);
                        mat[globalM * K + globalK] = (rand() % 2) ? val : -val;
                    } else {
                        mat[globalM * K + globalK] = 0.0f;
                    }
                }
            }
        }
    }
}

/*
 * Generate B matrix with FIXED PATTERN block-level sparsity
 *
 * Applies the SAME user-specified pattern to ALL K-blocks in B matrix.
 */
template <int BK = 8, int WN = 32>
void randomize_matrix_B_blocklevel_pattern(float *mat, int K, int N,
                                           std::string_view pattern,
                                           unsigned int seed = 0) {
    if (seed != 0) srand(seed);

    // Convert pattern string to uint8_t bitmask
    uint8_t fixed_pattern = computeExpandedIndicesBits(pattern);

    const int numKBlocks = K / BK;
    const int numNBlocks = N / WN;

    for (int nBlock = 0; nBlock < numNBlocks; nBlock++) {
        for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
            for (int col = 0; col < WN; col++) {
                const int globalN = nBlock * WN + col;
                if (globalN >= N) break;

                for (int k = 0; k < BK; k++) {
                    const int globalK = kBlock * BK + k;
                    if (globalK >= K) break;

                    const bool is_dense = (fixed_pattern & (1 << (7 - k)));  // MSB first
                    if (is_dense) {
                        float val = (float)(rand() % 5) + 0.01f * (rand() % 5);
                        mat[globalK * N + globalN] = (rand() % 2) ? val : -val;
                    } else {
                        mat[globalK * N + globalN] = 0.0f;
                    }
                }
            }
        }
    }
}
