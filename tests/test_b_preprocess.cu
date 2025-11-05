#include "../include/utils.cuh"
#include "../include/metadata.cuh"
#include "../src/preprocessors/b_preprocessor_hybrid.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <string_view>

using std::cout;
using std::endl;

bool test_pattern(const char* pattern_str, int K, int N, int BK, int TN) {
    std::string_view pattern(pattern_str);

    // Calculate expected pattern from string
    uint8_t expectedPattern = 0;
    for (int i = 0; i < 8 && i < pattern.length(); i++) {
        if (pattern[i] == '1') {
            expectedPattern |= (1 << i);
        }
    }

    cout << "========================================" << endl;
    cout << "Testing pattern: \"" << pattern << "\"" << endl;
    cout << "Expected encoding: 0b";
    for (int bit = 7; bit >= 0; bit--) {
        cout << ((expectedPattern >> bit) & 1);
    }
    cout << " (0x" << std::hex << (int)expectedPattern << std::dec << ")" << endl;

    // Allocate and initialize matrices
    float *h_B = (float *)malloc(K * N * sizeof(float));
    randomize_matrix_with_pattern(h_B, K, N, pattern);

    // Copy to device
    float *d_B;
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run B preprocessing
    BMatrixPatternMetadata meta = analyze_b_sparsity_pattern_gpu(d_B, K, N, BK, TN);

    // Copy patterns back to host for verification
    const int totalBlocks = meta.numKBlocks * meta.numNBlocks;
    uint8_t *h_patterns = (uint8_t *)malloc(totalBlocks * sizeof(uint8_t));
    cudaMemcpy(h_patterns, meta.d_blockPatterns, totalBlocks * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    // Verify patterns
    int correct = 0;
    int total = 0;

    for (int kb = 0; kb < meta.numKBlocks; kb++) {
        for (int nb = 0; nb < meta.numNBlocks; nb++) {
            const int idx = kb * meta.numNBlocks + nb;
            const uint8_t patternVal = h_patterns[idx];
            total++;

            if (patternVal == expectedPattern) {
                correct++;
            } else if (kb == 0 && nb < 3) {  // Print first few mismatches
                cout << "  Mismatch at block [" << kb << "][" << nb << "]: "
                     << "got 0x" << std::hex << (int)patternVal
                     << ", expected 0x" << (int)expectedPattern << std::dec << endl;
            }
        }
    }

    bool passed = (correct == total);
    cout << "Results: " << correct << "/" << total << " blocks correct";
    if (passed) {
        cout << " ✓" << endl;
    } else {
        cout << " ✗" << endl;
    }

    // Sample a few patterns
    cout << "Sample patterns (first 4 blocks):" << endl;
    for (int i = 0; i < 4 && i < totalBlocks; i++) {
        cout << "  Block " << i << ": 0b";
        for (int bit = 7; bit >= 0; bit--) {
            cout << ((h_patterns[i] >> bit) & 1);
        }
        cout << " (0x" << std::hex << (int)h_patterns[i] << std::dec << ")" << endl;
    }

    // Cleanup
    free(h_B);
    free(h_patterns);
    cudaFree(d_B);
    free_b_pattern_metadata(meta);

    return passed;
}

int main() {
    constexpr int K = 1024;
    constexpr int N = 1024;
    constexpr int BK = 8;
    constexpr int TN = 8;

    cout << "Testing B-matrix preprocessing on multiple patterns" << endl;
    cout << "Matrix size: K=" << K << ", N=" << N << endl;
    cout << "Block size: BK=" << BK << ", TN=" << TN << endl;
    cout << endl;

    // Test multiple patterns
    const char* patterns[] = {
        "11000000",  // 25% dense
        "11110000",  // 50% dense
        "11111100",  // 75% dense
        "11111111"   // 100% dense
    };

    int passed = 0;
    int total = sizeof(patterns) / sizeof(patterns[0]);

    for (int i = 0; i < total; i++) {
        if (test_pattern(patterns[i], K, N, BK, TN)) {
            passed++;
        }
        cout << endl;
    }

    cout << "========================================" << endl;
    cout << "Overall: " << passed << "/" << total << " patterns passed" << endl;

    return (passed == total) ? 0 : 1;
}
