#pragma once

#include <cuda_runtime.h>

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

// ============================================================================
// Block-wise Pattern Encoding (Kernel 20: ESMM Block-wise Uniform)
// ============================================================================
// Use case: Each 8×32 block has uniform sparsity pattern
// Memory: 1 byte per block (~64 KB for 4096×4096)
// Benefit: Compile-time unrolling with warp-uniform execution

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

// ============================================================================
// Row-level Bitmask Encoding (Kernel 17: ESMM Row-Level)
// ============================================================================
// Use case: Pattern varies per row, but repeats within 8-element blocks
// Memory: 1 byte per row per K-block
// Benefit: Fine-grained control, no global uniformity required

struct RowLevelMetadata {
    uint8_t* d_list;    // Bitmask for each row per K-block
    uint8_t* h_list;    // Host copy (optional)
    int totalSize;      // M * (K / BK)
};

inline void free_rowlevel_metadata(RowLevelMetadata& result) {
    if (result.d_list) cudaFree(result.d_list);
    if (result.h_list) free(result.h_list);
    result.d_list = nullptr;
    result.h_list = nullptr;
}

// ============================================================================
// Count+Offset Encoding (Kernel 19: ESMM Count+Offset)
// ============================================================================
// Use case: Sparse patterns where we need explicit position lists
// Memory: 5 bytes per row per K-block (1 count + 4 offsets)
// Benefit: Efficient for very sparse matrices (1-4 non-zeros per block)

struct CountOffsetMetadata {
    uint8_t count;      // Number of non-zero elements (0-8)
    uint8_t offsets[4]; // Positions of first 4 non-zeros
};

struct CountOffsetResult {
    CountOffsetMetadata* d_list;  // Device metadata array
    CountOffsetMetadata* h_list;  // Host metadata array (optional)
    int totalSize;                // M * (K / BK)
};

inline void free_countoffset_result(CountOffsetResult& result) {
    if (result.d_list) cudaFree(result.d_list);
    if (result.h_list) free(result.h_list);
    result.d_list = nullptr;
    result.h_list = nullptr;
}

// ============================================================================
// Pattern-Specialized Encoding (Kernel 18: ESMM Pattern-Specialized)
// ============================================================================
// Use case: Entire matrix has single known pattern at compile time
// Memory: 0 bytes (pattern hardcoded in kernel)
// Benefit: Zero runtime overhead, maximum performance

// No metadata structure needed - pattern is compile-time constant

// ============================================================================
// B-Matrix Block-wise Pattern Encoding (For B-Sparsity)
// ============================================================================
// Use case: Each BK×TN block (8×8) has uniform sparsity pattern
// Memory: 1 byte per block
// Benefit: Skip computation and output writes for zero column groups

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
