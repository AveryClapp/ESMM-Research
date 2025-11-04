#pragma once

/* Row-Level Preprocessor: GPU-based pattern encoding per row */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include <cuda_runtime.h>

/*
 * GPU Preprocessing Kernel for Row-Level Patterns
 *
 * Each thread processes one row across all K-blocks
 * Generates 8-bit pattern for each (row, k-block) pair
 */
template <const int BK>
__global__ void preprocess_rowlevel_patterns(
    int M, int K,
    const float* __restrict__ A,
    uint8_t* __restrict__ rowPatterns
) {
    const int numKBlocks = K / BK;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    // Process all K-blocks for this row
    for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
        uint8_t pattern = 0;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            const int col = kBlock * BK + k;
            if (A[row * K + col] != 0.0f) {
                pattern |= (1 << k);
            }
        }

        rowPatterns[row * numKBlocks + kBlock] = pattern;
    }
}

/*
 * GPU-based row-level preprocessing
 */
RowLevelMetadata analyze_sparsity_pattern_rowlevel_gpu(float* d_A, int M, int K, int BK) {
    RowLevelMetadata meta;
    const int numKBlocks = K / BK;
    meta.totalSize = M * numKBlocks;

    // Allocate device memory for patterns
    cudaMalloc(&meta.d_list, meta.totalSize * sizeof(uint8_t));
    cudaMemset(meta.d_list, 0, meta.totalSize * sizeof(uint8_t));
    meta.h_list = nullptr; // Don't need host copy

    // Launch preprocessing kernel
    constexpr int THREADS_PER_BLOCK = 256;
    const int numBlocks = CEIL_DIV(M, THREADS_PER_BLOCK);

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(numBlocks);

    if (BK == 8) {
        preprocess_rowlevel_patterns<8><<<gridDim, blockDim>>>(M, K, d_A, meta.d_list);
    } else if (BK == 16) {
        preprocess_rowlevel_patterns<16><<<gridDim, blockDim>>>(M, K, d_A, meta.d_list);
    } else {
        printf("Error: Unsupported BK=%d\\n", BK);
    }

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Row-level preprocessing kernel error: %s\\n", cudaGetErrorString(error));
    }

    float metadataKB = meta.totalSize / 1024.0f;
    printf("Row-level GPU preprocessing: %d patterns (%.1f KB metadata)\\n",
           meta.totalSize, metadataKB);

    return meta;
}
