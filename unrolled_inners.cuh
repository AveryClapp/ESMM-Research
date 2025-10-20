#pragma once

// Auto-generated sparsity pattern kernels
// Each kernel handles a specific 8-bit sparsity pattern (0x00 to 0xFF)
// Bit i (LSB=0) indicates whether element i should be processed
// Pattern 0x00 = fully sparse (no operations)
// Pattern 0xFF = fully dense (all 8 operations)

__forceinline__ __device__ void multiply_pattern_1(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
}

__forceinline__ __device__ void multiply_pattern_2(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

__forceinline__ __device__ void multiply_pattern_3(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
}

__forceinline__ __device__ void multiply_pattern_4(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
}

__forceinline__ __device__ void multiply_pattern_5(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
}

__forceinline__ __device__ void multiply_pattern_6(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
}

__forceinline__ __device__ void multiply_pattern_7(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00000111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
}

__forceinline__ __device__ void multiply_pattern_8(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_9(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_10(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_11(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_12(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_13(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_14(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_15(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00001111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
}

__forceinline__ __device__ void multiply_pattern_16(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_17(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_18(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_19(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_20(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_21(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_22(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_23(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00010111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_24(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_25(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_26(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_27(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_28(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_29(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_30(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_31(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00011111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
}

__forceinline__ __device__ void multiply_pattern_32(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_33(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_34(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_35(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_36(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_37(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_38(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_39(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00100111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_40(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_41(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_42(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_43(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_44(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_45(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_46(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_47(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00101111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_48(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_49(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_50(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_51(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_52(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_53(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_54(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_55(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00110111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_56(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_57(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_58(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_59(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_60(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_61(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_62(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_63(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b00111111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
}

__forceinline__ __device__ void multiply_pattern_64(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_65(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_66(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_67(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_68(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_69(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_70(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_71(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01000111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_72(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_73(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_74(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_75(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_76(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_77(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_78(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_79(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01001111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_80(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_81(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_82(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_83(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_84(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_85(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_86(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_87(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01010111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_88(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_89(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_90(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_91(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_92(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_93(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_94(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_95(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01011111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_96(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_97(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_98(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_99(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_100(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_101(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_102(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_103(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01100111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_104(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_105(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_106(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_107(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_108(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_109(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_110(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_111(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01101111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_112(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_113(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_114(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_115(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_116(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_117(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_118(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_119(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01110111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_120(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_121(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_122(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_123(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_124(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_125(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_126(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_127(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b01111111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
}

__forceinline__ __device__ void multiply_pattern_128(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_129(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_130(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_131(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_132(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_133(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_134(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_135(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10000111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_136(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_137(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_138(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_139(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_140(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_141(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_142(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_143(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10001111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_144(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_145(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_146(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_147(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_148(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_149(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_150(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_151(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10010111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_152(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_153(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_154(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_155(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_156(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_157(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_158(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_159(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10011111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_160(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_161(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_162(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_163(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_164(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_165(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_166(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_167(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10100111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_168(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_169(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_170(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_171(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_172(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_173(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_174(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_175(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10101111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_176(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_177(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_178(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_179(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_180(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_181(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_182(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_183(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10110111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_184(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_185(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_186(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_187(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_188(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_189(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_190(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_191(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b10111111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_192(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_193(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_194(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_195(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_196(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_197(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_198(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_199(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11000111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_200(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_201(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_202(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_203(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_204(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_205(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_206(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_207(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11001111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_208(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_209(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_210(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_211(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_212(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_213(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_214(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_215(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11010111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_216(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_217(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_218(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_219(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_220(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_221(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_222(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_223(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11011111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_224(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_225(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_226(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_227(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_228(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_229(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_230(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_231(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11100111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_232(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_233(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_234(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_235(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_236(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_237(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_238(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_239(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11101111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_240(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_241(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_242(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_243(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_244(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_245(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_246(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_247(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11110111
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_248(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111000
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_249(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111001
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_250(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111010
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_251(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111011
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_252(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111100
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_253(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111101
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 0] += regM_val * regN[regNBase + 0];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_254(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111110
	const int regNBase = wSubColIdx * 8;
	const int threadResBase = wSubRowIdx * (WNITER * 8) + (wSubColIdx * 8);
	threadResults[threadResBase + 1] += regM_val * regN[regNBase + 1];
	threadResults[threadResBase + 2] += regM_val * regN[regNBase + 2];
	threadResults[threadResBase + 3] += regM_val * regN[regNBase + 3];
	threadResults[threadResBase + 4] += regM_val * regN[regNBase + 4];
	threadResults[threadResBase + 5] += regM_val * regN[regNBase + 5];
	threadResults[threadResBase + 6] += regM_val * regN[regNBase + 6];
	threadResults[threadResBase + 7] += regM_val * regN[regNBase + 7];
}

__forceinline__ __device__ void multiply_pattern_255(int wSubRowIdx, int wSubColIdx,
				int WNITER, float regM_val, float* regN,
						float* threadResults) {{
	// Pattern: 0b11111111 - fully dense
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

#endif // SPARSITY_KERNELS_CUH


