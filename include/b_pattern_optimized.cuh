#pragma once

/*
 * Optimized B-pattern specialized multiply functions
 * Handles 5 most common patterns with zero overhead
 */

// Pattern 0x01: 10000000 (12.5% dense - 1 FMA)
__forceinline__ __device__ void multiply_b_pattern_0x01(
    int wSubRowIdx, int wSubColIdx, int WNITER,
    float regM_val, float* regN, float* threadResults) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
    threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
}

// Pattern 0x03: 11000000 (25% dense - 2 FMAs)
__forceinline__ __device__ void multiply_b_pattern_0x03(
    int wSubRowIdx, int wSubColIdx, int WNITER,
    float regM_val, float* regN, float* threadResults) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
    threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
    threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

// Pattern 0x0F: 11110000 (50% dense - 4 FMAs)
__forceinline__ __device__ void multiply_b_pattern_0x0F(
    int wSubRowIdx, int wSubColIdx, int WNITER,
    float regM_val, float* regN, float* threadResults) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
    threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
    threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
    threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
    threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

// Pattern 0x3F: 11111100 (75% dense - 6 FMAs)
__forceinline__ __device__ void multiply_b_pattern_0x3F(
    int wSubRowIdx, int wSubColIdx, int WNITER,
    float regM_val, float* regN, float* threadResults) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
    threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
    threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
    threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
    threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
    threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
    threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

// Pattern 0xFF: 11111111 (100% dense - 8 FMAs)
__forceinline__ __device__ void multiply_b_pattern_0xFF(
    int wSubRowIdx, int wSubColIdx, int WNITER,
    float regM_val, float* regN, float* threadResults) {
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

// Optimized dispatcher for 5 common patterns
__forceinline__ __device__ void dispatch_b_pattern_optimized(
    const uint8_t B_pattern,
    const int wSubRowIdx, const int wSubColIdx, const int WNITER,
    const float regM_val, float* regN, float* threadResults) {

    switch (B_pattern) {
        case 0x00:
            // All zeros - no-op
            break;
        case 0x01:
            multiply_b_pattern_0x01(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
        case 0x03:
            multiply_b_pattern_0x03(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
        case 0x0F:
            multiply_b_pattern_0x0F(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
        case 0x3F:
            multiply_b_pattern_0x3F(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
        case 0xFF:
            multiply_b_pattern_0xFF(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
        default:
            // Fallback to dense for unknown patterns
            multiply_b_pattern_0xFF(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults);
            break;
    }
}
