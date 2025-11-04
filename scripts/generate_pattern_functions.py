#!/usr/bin/env python3
"""
Generate pattern-specialized compute functions for BK=8 sparse GEMM.
Each function handles one specific 8-bit sparsity pattern with zero branches.
"""

# Specific patterns to generate (in binary -> decimal):
# Common sparse patterns:
# 0b00001111 -> 15  (lower 4 bits - matches "11110000" string pattern)
# 0b10000000 -> 128, 0b11000000 -> 192, 0b11110000 -> 240
# 0b11111100 -> 252, 0b11111111 -> 255
PATTERNS_TO_GENERATE = [15, 128, 192, 240, 252, 255]

def generate_pattern_function(pattern: int) -> str:
    """Generate optimized compute function for a specific 8-bit pattern."""
    active_cols = [i for i in range(8) if (pattern >> i) & 1]

    code = f"""
// Pattern {pattern:#04x} (0b{pattern:08b}) - {len(active_cols)} active columns
__device__ inline void compute_pattern_{pattern}(
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ threadResults,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    const uint WM, const uint WN, const uint TM, const uint TN,
    const uint BM, const uint BN, const uint WMITER, const uint WNITER,
    const uint WSUBM, const uint WSUBN
) {{
"""

    if len(active_cols) == 0:
        code += "    // Empty pattern - no computation needed\n"
    else:
        # Generate code for each active column
        # Allocate register arrays once outside the column loop
        code += "    // Temporary register storage (reused for each column)\n"
        code += "    float regM[8];  // Support up to WMITER=8, TM=1\n"
        code += "    float regN[32];  // Support up to WNITER=4, TN=8\n\n"

        for col_idx in active_cols:
            code += f"    // Column {col_idx}\n"
            code += "    {\n"

            # Load from A (simplified for TM=1, WMITER=1)
            code += "        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {\n"
            code += f"            regM[wSubRowIdx] = As[({col_idx} * BM) + warpRow * WM + \n"
            code += "                wSubRowIdx * WSUBM + threadRowInWarp * TM];\n"
            code += "        }\n\n"

            # Load from B (across WNITER iterations)
            code += "        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {\n"
            code += "            #pragma unroll\n"
            code += "            for (uint tn = 0; tn < TN; ++tn) {\n"
            code += f"                regN[wSubColIdx * TN + tn] = Bs[({col_idx} * BN) + warpCol * WN + \n"
            code += "                    wSubColIdx * WSUBN + threadColInWarp * TN + tn];\n"
            code += "            }\n"
            code += "        }\n\n"

            # Multiply-accumulate
            code += "        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {\n"
            code += "            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {\n"
            code += "                #pragma unroll\n"
            code += "                for (uint tn = 0; tn < TN; ++tn) {\n"
            code += "                    const uint idx = (wSubRowIdx * TM) * (WNITER * TN) + \n"
            code += "                        wSubColIdx * TN + tn;\n"
            code += "                    threadResults[idx] += regM[wSubRowIdx] * regN[wSubColIdx * TN + tn];\n"
            code += "                }\n"
            code += "            }\n"
            code += "        }\n"
            code += "    }\n\n"

    code += "}\n"
    return code


def generate_generic_fallback() -> str:
    """Generate generic fallback computation for unhandled patterns."""
    code = """
// Generic fallback for patterns not explicitly specialized
__device__ inline void compute_pattern_generic(
    const uint8_t pattern,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ threadResults,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    const uint WM, const uint WN, const uint TM, const uint TN,
    const uint BM, const uint BN, const uint WMITER, const uint WNITER,
    const uint WSUBM, const uint WSUBN
) {
    // Generic implementation with branch to handle any pattern
    float regM[8];  // Support up to WMITER=8, TM=1
    float regN[32];  // Support up to WNITER=4, TN=8

    for (int col = 0; col < 8; ++col) {
        if ((pattern >> col) & 1) {
            // Load from A
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                regM[wSubRowIdx] = As[(col * BM) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM];
            }

            // Load from B
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                #pragma unroll
                for (uint tn = 0; tn < TN; ++tn) {
                    regN[wSubColIdx * TN + tn] = Bs[(col * BN) + warpCol * WN +
                        wSubColIdx * WSUBN + threadColInWarp * TN + tn];
                }
            }

            // Multiply-accumulate
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    #pragma unroll
                    for (uint tn = 0; tn < TN; ++tn) {
                        const uint idx = (wSubRowIdx * TM) * (WNITER * TN) +
                            wSubColIdx * TN + tn;
                        threadResults[idx] += regM[wSubRowIdx] * regN[wSubColIdx * TN + tn];
                    }
                }
            }
        }
    }
}
"""
    return code


def generate_dispatch_table() -> str:
    """Generate switch statement for pattern dispatch."""
    code = """
__device__ inline void dispatch_pattern(
    const uint8_t pattern,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    float* __restrict__ threadResults,
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp,
    const uint WM, const uint WN, const uint TM, const uint TN,
    const uint BM, const uint BN, const uint WMITER, const uint WNITER,
    const uint WSUBM, const uint WSUBN
) {
    switch(pattern) {
"""

    for pattern in PATTERNS_TO_GENERATE:
        code += f"        case {pattern}: compute_pattern_{pattern}(As, Bs, threadResults, "
        code += "warpRow, warpCol, threadRowInWarp, threadColInWarp, "
        code += "WM, WN, TM, TN, BM, BN, WMITER, WNITER, WSUBM, WSUBN); break;\n"

    code += """        default:
            // Fallback to generic computation for unhandled patterns
            compute_pattern_generic(pattern, As, Bs, threadResults,
                warpRow, warpCol, threadRowInWarp, threadColInWarp,
                WM, WN, TM, TN, BM, BN, WMITER, WNITER, WSUBM, WSUBN);
            break;
    }
}
"""
    return code


def generate_header() -> str:
    """Generate file header."""
    return """#pragma once

/* Auto-generated pattern-specialized compute functions for BK=8 */
/* Generated by generate_pattern_functions.py */

#include <cuda_runtime.h>

// Each pattern function handles one specific 8-bit sparsity pattern
// with completely unrolled computation and zero branches.
"""


def main():
    """Generate complete CUDA header file with specific pattern functions."""
    output_file = "pattern_functions_bk8.cuh"

    print(f"Generating {output_file}...")
    print(f"  Patterns: {[f'{p:#04x} (0b{p:08b})' for p in PATTERNS_TO_GENERATE]}")

    with open(output_file, 'w') as f:
        f.write(generate_header())
        f.write("\n// ============================================================================\n")
        f.write(f"// PATTERN-SPECIALIZED COMPUTE FUNCTIONS ({len(PATTERNS_TO_GENERATE)} total)\n")
        f.write("// ============================================================================\n")

        for pattern in PATTERNS_TO_GENERATE:
            print(f"  Generating pattern {pattern:#04x} (0b{pattern:08b})...")
            f.write(generate_pattern_function(pattern))

        f.write("\n// ============================================================================\n")
        f.write("// GENERIC FALLBACK FUNCTION\n")
        f.write("// ============================================================================\n")
        f.write(generate_generic_fallback())

        f.write("\n// ============================================================================\n")
        f.write("// DISPATCH FUNCTION\n")
        f.write("// ============================================================================\n")
        f.write(generate_dispatch_table())

    print(f"✓ Generated {output_file} with {len(PATTERNS_TO_GENERATE)} pattern functions + generic fallback")


if __name__ == "__main__":
    main()
