#include <iostream>
#include <set>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>

// Minimal utils
void cudaCheckError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void randomize_matrix_with_pattern(float *mat, int rows, int cols, std::string_view pattern) {
    int pattern_len = pattern.length();
    for (int i = 0; i < rows * cols; i++) {
        int col = i % cols;
        int pattern_idx = col % pattern_len;
        if (pattern[pattern_idx] == '1') {
            mat[i] = static_cast<float>(rand()) / RAND_MAX;
        } else {
            mat[i] = 0.0f;
        }
    }
}

// Include preprocessor
template <const int BK, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
preprocess_A_rowlevel(int M, int N, int K, float *A, uint8_t* A_LIST) {
    constexpr int WARP_SIZE = 32;
    const uint numKBlocks = K / BK;
    const uint warpId = threadIdx.x / WARP_SIZE;
    const uint laneId = threadIdx.x % WARP_SIZE;
    constexpr int ROWS_PER_WARP = WARP_SIZE / BK;
    constexpr int THREADS_PER_ROW = BK;
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * (NUM_THREADS / WARP_SIZE);

    for (uint rowBlockBase = blockIdx.x * ROWS_PER_BLOCK;
         rowBlockBase < M;
         rowBlockBase += gridDim.x * ROWS_PER_BLOCK) {

        const uint localRowInWarp = laneId / THREADS_PER_ROW;
        const uint threadPosInRow = laneId % THREADS_PER_ROW;
        const uint row = rowBlockBase + warpId * ROWS_PER_WARP + localRowInWarp;

        if (row >= M) return;

        #pragma unroll 16
        for (uint kBlock = 0; kBlock < numKBlocks; kBlock++) {
            const uint kOffset = kBlock * BK + threadPosInRow;
            const float val = __ldg(&A[row * K + kOffset]);
            const uint32_t ballot = __ballot_sync(0xffffffff, val != 0.0f);

            uint8_t mask;
            if constexpr (BK == 8) {
                const uint shift = localRowInWarp * 8;
                mask = (ballot >> shift) & 0xFF;
            } else if constexpr (BK == 16) {
                const uint shift = localRowInWarp * 16;
                mask = (ballot >> shift) & 0xFFFF;
            }

            if (threadPosInRow == 0) {
                A_LIST[row * numKBlocks + kBlock] = mask;
            }
        }
    }
}

int main() {
    constexpr int rows = 1024;
    constexpr int inners = 1024;
    constexpr int BK = 8;
    constexpr std::string_view sparsity = "11110000";

    // Generate matrix with pattern
    float *h_A = (float *)malloc(rows * inners * sizeof(float));
    randomize_matrix_with_pattern(h_A, rows, inners, sparsity);

    // Copy to device
    float *d_A;
    cudaMalloc(&d_A, rows * inners * sizeof(float));
    cudaMemcpy(d_A, h_A, rows * inners * sizeof(float), cudaMemcpyHostToDevice);

    // Run preprocessing
    const int numKBlocks = inners / BK;
    const int totalSize = rows * numKBlocks;
    uint8_t* d_list;
    cudaMalloc(&d_list, totalSize * sizeof(uint8_t));
    cudaMemset(d_list, 0, totalSize * sizeof(uint8_t));

    constexpr int NUM_THREADS = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int ROWS_PER_WARP = WARP_SIZE / BK;
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * (NUM_THREADS / WARP_SIZE);

    dim3 blockDim(NUM_THREADS);
    dim3 gridDim((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    preprocess_A_rowlevel<BK, NUM_THREADS><<<gridDim, blockDim>>>(rows, rows, inners, d_A, d_list);
    cudaDeviceSynchronize();

    // Copy back masks
    uint8_t* h_list = (uint8_t*)malloc(totalSize * sizeof(uint8_t));
    cudaMemcpy(h_list, d_list, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Analyze patterns
    std::set<int> unique_patterns;
    int pattern_counts[256] = {0};

    for (int i = 0; i < totalSize; i++) {
        unique_patterns.insert(h_list[i]);
        pattern_counts[h_list[i]]++;
    }

    std::cout << "Total unique patterns found: " << unique_patterns.size() << std::endl;
    std::cout << "\nPatterns and their frequencies:" << std::endl;
    for (int pattern : unique_patterns) {
        std::cout << "  Pattern " << pattern << " (0b";
        for (int bit = 7; bit >= 0; bit--) {
            std::cout << ((pattern >> bit) & 1);
        }
        std::cout << ", 0x" << std::hex << pattern << std::dec << "): "
                  << pattern_counts[pattern] << " occurrences" << std::endl;
    }

    std::cout << "\nGenerated patterns in code: 128, 192, 240, 252, 255" << std::endl;

    free(h_A);
    free(h_list);
    cudaFree(d_A);
    cudaFree(d_list);

    return 0;
}
