// Minimal B-pattern dispatch (5 common patterns only)
// For faster compilation during experimentation

__forceinline__ __device__ void dispatch_b_pattern(
    const uint8_t B_pattern,
    const int wSubRowIdx, const int wSubColIdx, const int WNITER,
    const float regM_val, float* regN, float* threadResults) {

    switch (B_pattern) {
        case 0x00:
            // Pattern 0: all zeros, no-op
            break;
        case 0x01:
            // Pattern: 10000000 (12.5% dense)
            multiply_pattern_1(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
        case 0x03:
            // Pattern: 11000000 (25% dense)
            multiply_pattern_3(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
        case 0x0F:
            // Pattern: 11110000 (50% dense)
            multiply_pattern_15(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
        case 0x3F:
            // Pattern: 11111100 (75% dense)
            multiply_pattern_63(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
        case 0xFF:
            // Pattern: 11111111 (100% dense)
            multiply_pattern_255(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
        default:
            // Unsupported pattern - fall back to dense
            multiply_pattern_255(wSubRowIdx, wSubColIdx, WNITER,
                regM_val, regN, threadResults);
            break;
    }
}
