#pragma once

#include <cuda_runtime.h>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define PATTERN_LENGTH 8

using std::cout;
using std::endl;

static const std::set<int> VALID_KERNEL_SET = {14, 15, 16, 17, 20, 21, 25};

std::vector<int> parse_kernel_selection(const std::string& input) {
  std::vector<int> kernels;
  if (input == "all") {
    for (int k : VALID_KERNEL_SET) {
      kernels.push_back(k);
    }
    return kernels;
  }
  size_t dash_pos = input.find('-');
  if (dash_pos != std::string::npos) {
    int start = std::stoi(input.substr(0, dash_pos));
    int end = std::stoi(input.substr(dash_pos + 1));
    for (int i = start; i <= end; i++) {
      if (VALID_KERNEL_SET.count(i)) {
        kernels.push_back(i);
      }
    }
    return kernels;
  }
  std::stringstream ss(input);
  std::string kernel_str;
  while (std::getline(ss, kernel_str, ',')) {
    int kernel = std::stoi(kernel_str);
    if (VALID_KERNEL_SET.count(kernel)) {
      kernels.push_back(kernel);
    }
  }
  return kernels;
}

const char* get_kernel_name(int kernel_choice) {
  switch (kernel_choice) {
    case 14: return "ESMM A-Only Sparse (Offset List, Templated Dispatch)";
    case 15: return "cuBLAS";
    case 16: return "ESMM A-Sparse Block-wise (Warp-Granularity Patterns)";
    case 17: return "ESMM B-Sparse Warp-Granularity (32-col, Zero-Divergence)";
    case 20: return "ESMM A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop)";
    case 21: return "ESMM A+B Sparse - 8x32 Granularity";
    case 25: return "ESMM A+B Simple Fused (Main Contribution)";
    default: return "Unknown Kernel";
  }
}

void print_usage(const char* program_name) {
  cout << "Usage: " << program_name << " [kernel_choice] [runs] [options]" << endl;
  cout << "\nAvailable kernels: 14, 15, 16, 17, 20, 21, 25" << endl;
  cout << "  14: A-Only Sparse (Offset List, Templated Dispatch)" << endl;
  cout << "  15: cuBLAS baseline" << endl;
  cout << "  16: A-Sparse Block-wise (Warp-Granularity Patterns)" << endl;
  cout << "  17: B-Sparse Warp-Granularity (32-col, Zero-Divergence)" << endl;
  cout << "  20: A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop)" << endl;
  cout << "  21: A+B Sparse - 8x32 Granularity" << endl;
  cout << "  25: A+B Simple Fused (Main Contribution)" << endl;
  cout << "\nKernel selection formats:" << endl;
  cout << "  Single:   25          (run K25)" << endl;
  cout << "  Multiple: \"20,21,25\" (comma-separated)" << endl;
  cout << "  Range:    \"14-17\"    (kernels in range that are valid)" << endl;
  cout << "  All:      all         (all 7 kernels)" << endl;
  cout << "\nOptions:" << endl;
  cout << "  --verbose, -v       Enable verbose output" << endl;
  cout << "  --no-check, -n      Skip result verification (performance-only mode)" << endl;
  cout << "  --random, -r        Use random unstructured sparsity" << endl;
  cout << "  --blockwise, -b     Use block-level sparsity (recommended for K20/21/25)" << endl;
  cout << "  --size <N>, -s <N>  Set matrix dimensions NxNxN (default: 4096)" << endl;
  cout << "  --pattern <P>       Set 8-bit sparsity pattern for A and B (default: \"11000000\")" << endl;
  cout << "  --pattern-a <P>     Set 8-bit sparsity pattern for A matrix only" << endl;
  cout << "  --pattern-b <P>     Set 8-bit sparsity pattern for B matrix only" << endl;
  cout << "  --help, -h          Show this help message" << endl;
  cout << "\nExamples:" << endl;
  cout << "  " << program_name << " 25 10 --blockwise --pattern 11110000 --size 4096 --verbose" << endl;
  cout << "  " << program_name << " 17 1 --size 2048 --pattern 11110000" << endl;
  cout << "  " << program_name << " 14 3 --pattern 11110000 --size 1024" << endl;
  cout << "  " << program_name << " \"20,21,25\" 5 --blockwise --pattern 11110000 --no-check" << endl;
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



