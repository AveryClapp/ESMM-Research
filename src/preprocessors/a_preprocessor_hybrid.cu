#pragma once

/* Hybrid Preprocessor: Detects uniform patterns and generates offset lists */

#include "../../include/utils.cuh"
#include <cuda_runtime.h>
#include <map>

/*
 * Analyzes A matrix to detect dominant sparsity patterns.
 *
 * Strategy:
 * 1. Build histogram of 8-bit patterns across all rows
 * 2. If one pattern dominates (>90%), use global offset list
 * 3. Otherwise, use per-row bitmasks for flexibility
 */

struct HybridMetadata {
    bool isUniform;          // True if uniform pattern across rows
    uint8_t dominantPattern; // The dominant 8-bit pattern (if uniform)
    uint8_t patternCount;    // Number of non-zeros in dominant pattern
    uint8_t offsets[8];      // Offsets for dominant pattern
    uint8_t* d_rowMasks;     // Per-row masks (if non-uniform)
    int totalRows;
    int numKBlocks;
};

/*
 * Host-side preprocessing to detect pattern uniformity
 * Analyzes matrix A and determines if we can use global offset list
 */
HybridMetadata analyze_sparsity_pattern(float* h_A, int M, int K, int BK) {
    HybridMetadata meta;
    meta.d_rowMasks = nullptr;
    meta.totalRows = M;
    meta.numKBlocks = K / BK;

    // Build histogram of patterns
    std::map<uint8_t, int> patternHistogram;
    int totalBlocks = M * meta.numKBlocks;

    for (int row = 0; row < M; row++) {
        for (int kBlock = 0; kBlock < meta.numKBlocks; kBlock++) {
            uint8_t pattern = 0;
            for (int k = 0; k < BK; k++) {
                int col = kBlock * BK + k;
                if (h_A[row * K + col] != 0.0f) {
                    pattern |= (1 << k);
                }
            }
            patternHistogram[pattern]++;
        }
    }

    // Find dominant pattern
    uint8_t dominantPattern = 0;
    int maxCount = 0;
    for (const auto& entry : patternHistogram) {
        if (entry.second > maxCount) {
            maxCount = entry.second;
            dominantPattern = entry.first;
        }
    }

    // Check if pattern is sufficiently uniform (>90% of blocks)
    float uniformity = (float)maxCount / totalBlocks;
    meta.isUniform = (uniformity > 0.90f);
    meta.dominantPattern = dominantPattern;

    if (meta.isUniform) {
        // Extract offsets from dominant pattern
        meta.patternCount = 0;
        for (int i = 0; i < BK; i++) {
            if (dominantPattern & (1 << i)) {
                meta.offsets[meta.patternCount++] = i;
            }
        }

        printf("Detected uniform pattern: 0x%02x (%.1f%% uniform, count=%d)\n",
               dominantPattern, uniformity * 100.0f, meta.patternCount);
    } else {
        // Need per-row masks - allocate device memory
        cudaMalloc(&meta.d_rowMasks, M * meta.numKBlocks * sizeof(uint8_t));

        // Build per-row masks on host
        uint8_t* h_rowMasks = (uint8_t*)malloc(M * meta.numKBlocks * sizeof(uint8_t));
        for (int row = 0; row < M; row++) {
            for (int kBlock = 0; kBlock < meta.numKBlocks; kBlock++) {
                uint8_t pattern = 0;
                for (int k = 0; k < BK; k++) {
                    int col = kBlock * BK + k;
                    if (h_A[row * K + col] != 0.0f) {
                        pattern |= (1 << k);
                    }
                }
                h_rowMasks[row * meta.numKBlocks + kBlock] = pattern;
            }
        }

        // Copy to device
        cudaMemcpy(meta.d_rowMasks, h_rowMasks, M * meta.numKBlocks * sizeof(uint8_t),
                   cudaMemcpyHostToDevice);
        free(h_rowMasks);

        printf("Non-uniform pattern detected (%.1f%% uniformity) - using per-row masks\n",
               uniformity * 100.0f);
    }

    return meta;
}

void free_hybrid_metadata(HybridMetadata& meta) {
    if (meta.d_rowMasks) {
        cudaFree(meta.d_rowMasks);
        meta.d_rowMasks = nullptr;
    }
}
