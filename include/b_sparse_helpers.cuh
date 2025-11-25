#pragma once

/*
 * Shared helper functions for B-sparsity kernels
 * These dispatch functions are used by both WN-granularity and TN-granularity variants
 */

__forceinline__ __device__ void multiply_offsets_1(int wSubRowIdx, int wSubColIdx,
        int WNITER, float regM_val,
        float* regN, float* threadResults,
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 1 Op using offsetList[0]
    int off0 = offsetList[0];
    threadResults[threadResBase + off0] += regM_val * regN[regNBase + off0];
}

__forceinline__ __device__ void multiply_offsets_2(int wSubRowIdx, int wSubColIdx,
        int WNITER, float regM_val,
        float* regN, float* threadResults,
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 2 Ops
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
}

__forceinline__ __device__ void multiply_offsets_4(int wSubRowIdx, int wSubColIdx,
        int WNITER, float regM_val,
        float* regN, float* threadResults,
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 4 Ops
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
    threadResults[threadResBase + offsetList[2]] += regM_val * regN[regNBase + offsetList[2]];
    threadResults[threadResBase + offsetList[3]] += regM_val * regN[regNBase + offsetList[3]];
}

__forceinline__ __device__ void multiply_offsets_8(int wSubRowIdx, int wSubColIdx,
        int WNITER, float regM_val,
        float* regN, float* threadResults,
        const uint8_t* offsetList) {
    const int regNBase = wSubColIdx * 8;
    const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);

    // 8 Ops - dense case
    threadResults[threadResBase + offsetList[0]] += regM_val * regN[regNBase + offsetList[0]];
    threadResults[threadResBase + offsetList[1]] += regM_val * regN[regNBase + offsetList[1]];
    threadResults[threadResBase + offsetList[2]] += regM_val * regN[regNBase + offsetList[2]];
    threadResults[threadResBase + offsetList[3]] += regM_val * regN[regNBase + offsetList[3]];
    threadResults[threadResBase + offsetList[4]] += regM_val * regN[regNBase + offsetList[4]];
    threadResults[threadResBase + offsetList[5]] += regM_val * regN[regNBase + offsetList[5]];
    threadResults[threadResBase + offsetList[6]] += regM_val * regN[regNBase + offsetList[6]];
    threadResults[threadResBase + offsetList[7]] += regM_val * regN[regNBase + offsetList[7]];
}

__forceinline__ __device__ void dispatch_multiply(int mode, int wSubRowIdx, int wSubColIdx,
        int WNITER, float regM_val,
        float* regN, float* threadResults,
        const uint8_t* offsetList) {
    switch (mode) {
        case 1:
            multiply_offsets_1(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 2:
            multiply_offsets_2(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 4:
            multiply_offsets_4(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
        case 8:
            multiply_offsets_8(wSubRowIdx, wSubColIdx, WNITER, regM_val, regN, threadResults, offsetList);
            break;
    }
}
