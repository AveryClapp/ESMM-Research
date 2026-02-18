#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define WARPSIZE 32

/*
 * ============================================================================
 * ESMM Research Project - Sparsity Metadata Structures
 * ============================================================================
 *
 * This file contains all metadata structures used for sparse matrix encoding.
 * Each structure is optimized for a specific sparsity pattern characteristic.
 */

// ============================================================================
// Generic Preprocessing Result (defined in utils.cuh)
// ============================================================================
// PreprocessResult is defined in utils.cuh for backward compatibility


struct BlockPatternMetadata {
    uint8_t* d_blockPatterns;  // Pattern for each 8×32 block
    int numWarpRows;           // Number of 32-row chunks (M / WM)
    int numKBlocks;            // Number of 8-element K-blocks (K / BK)
};

inline void free_block_pattern_metadata(BlockPatternMetadata& meta) {
    if (meta.d_blockPatterns) {
        cudaFree(meta.d_blockPatterns);
        meta.d_blockPatterns = nullptr;
    }
}

struct BMatrixPatternMetadata {
    uint8_t* d_blockPatterns;  // Pattern for each BK×TN block
    int numKBlocks;            // Number of K-blocks (K / BK)
    int numNBlocks;            // Number of N-blocks (N / TN)
};

inline void free_b_pattern_metadata(BMatrixPatternMetadata& meta) {
    if (meta.d_blockPatterns) {
        cudaFree(meta.d_blockPatterns);
        meta.d_blockPatterns = nullptr;
    }
}

