#pragma once

/*
 * Joint A+B Sparsity Preprocessing
 * 
 * Precomputes the INTERSECTION of A and B patterns offline.
 * Main kernel just looks up one pattern instead of two + intersection.
 */

#include "../../include/metadata.cuh"
#include "../../include/utils.cuh"
#include "../../include/pattern_lut.cuh"
#include <cuda_runtime.h>

/*
 * Compute Joint Pattern: A ∩ B
 * 
 * For each (M-block, N-block, K-block), compute:
 *   joint_pattern = A_pattern & B_pattern  (bitwise AND)
 * 
 * This gives us the K-indices where BOTH A and B are non-zero.
 */
__global__ void compute_joint_patterns_kernel(
    const uint8_t* __restrict__ a_patterns,  // [numMBlocks × numKBlocks]
    const uint8_t* __restrict__ b_patterns,  // [numNBlocks × numKBlocks]
    uint8_t* __restrict__ joint_patterns,     // [numMBlocks × numNBlocks × numKBlocks]
    int numMBlocks, int numNBlocks, int numKBlocks) {
    
    // 3D grid: (M-blocks, N-blocks, K-blocks)
    const int mBlock = blockIdx.x;
    const int nBlock = blockIdx.y;
    const int kBlock = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (mBlock >= numMBlocks || nBlock >= numNBlocks || kBlock >= numKBlocks) return;
    
    // Read A and B patterns
    const uint8_t a_pat = a_patterns[mBlock * numKBlocks + kBlock];
    const uint8_t b_pat = b_patterns[nBlock * numKBlocks + kBlock];
    
    // Bitwise AND to get intersection
    const uint8_t joint_pat = a_pat & b_pat;
    
    // Store joint pattern
    const int outIdx = (mBlock * numNBlocks + nBlock) * numKBlocks + kBlock;
    joint_patterns[outIdx] = joint_pat;
}

struct JointPatternMetadata {
    uint8_t* d_jointPatterns;  // [numMBlocks × numNBlocks × numKBlocks]
    int numMBlocks;
    int numNBlocks;
    int numKBlocks;
};

void free_joint_pattern_metadata(JointPatternMetadata& meta) {
    if (meta.d_jointPatterns) {
        cudaFree(meta.d_jointPatterns);
        meta.d_jointPatterns = nullptr;
    }
}

JointPatternMetadata preprocess_joint_patterns(
    uint8_t* d_a_patterns,  // A patterns: [numMBlocks × numKBlocks]
    uint8_t* d_b_patterns,  // B patterns: [numNBlocks × numKBlocks]
    int M, int N, int K, int WM, int WN, int BK) {
    
    JointPatternMetadata meta;
    meta.numMBlocks = M / WM;
    meta.numNBlocks = N / WN;
    meta.numKBlocks = K / BK;
    
    // Allocate joint patterns: [M-blocks × N-blocks × K-blocks]
    const size_t totalPatterns = (size_t)meta.numMBlocks * meta.numNBlocks * meta.numKBlocks;
    cudaMalloc(&meta.d_jointPatterns, totalPatterns * sizeof(uint8_t));
    
    // Launch kernel
    dim3 blockDim(256);
    dim3 gridDim(meta.numMBlocks, meta.numNBlocks, CEIL_DIV(meta.numKBlocks, 256));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    compute_joint_patterns_kernel<<<gridDim, blockDim>>>(
        d_a_patterns, d_b_patterns, meta.d_jointPatterns,
        meta.numMBlocks, meta.numNBlocks, meta.numKBlocks);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Joint pattern preprocessing error: %s\n", cudaGetErrorString(error));
    }
    
    float sizeMB = totalPatterns / (1024.0f * 1024.0f);
    printf("Joint pattern preprocessing: %.3f ms (%.2f MB metadata)\n", 
           elapsed_ms, sizeMB);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return meta;
}

