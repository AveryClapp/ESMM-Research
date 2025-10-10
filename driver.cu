#include "utils.cuh"
#include "runners.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <string_view>

using std::cout;
using std::endl;
using std::cin;

bool run_single_kernel(int kernel_choice, int rows, int cols, int inners, 
                      float* d_A, float* d_B, float* d_C, 
                      float* h_C, float* h_C_ref, int runs, 
                      bool verbose, bool check_results, std::string_view pattern) {
    bool res = false;

    // Reset d_C to zeros before each kernel
    cudaCheckError(cudaMemset(d_C, 0, rows * cols * sizeof(float)));

    if (verbose) {
        cout << "Running kernel " << kernel_choice << ": " << get_kernel_name(kernel_choice);
        if (!check_results) {
            cout << " (Performance-only mode)";
        }
        cout << endl;
    }

    switch (kernel_choice){
    case 1: // Naive Implementation
        res = run_naive(rows, cols, inners, d_A, d_B, d_C, runs);
        break;
    case 2: // Global Memory Coalescing
        res = run_gmem_coalesce(rows, cols, inners, d_A, d_B, d_C, runs);
        break;
    case 3: // Shared Memory Blocks
        res = run_smem_blocking(rows, cols, inners, d_A, d_B, d_C, runs);
        break;
    case 4: // One Dimensional Blocktiling
        res = run_one_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
        break;
    case 5: // Two Dimensional Blocktiling
        res = run_two_blocktiling(rows, cols, inners, d_A, d_B, d_C, runs);
        break;
    case 6: // Vectorized Memory Accessing
        if (check_results) {
            res = run_vectorized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_vectorized_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 7: // 1 Dimensional Vectorized Approach
        if (check_results) {
            res = run_1d_vec(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_1d_vec_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 8: // Basic Warptiling
        if (check_results) {
            res = run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_warptiling_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 9: // 1-Dimensional Warptiling
        if (check_results) {
            res = run_1d_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_1d_warptiling_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 10: // Emergent Sparsity Matrix Multiplication (our kernel)
        if (check_results) {
            res = run_esmm(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 11: // Experimental warpskipping approach to ESMM
        if (check_results) {
            res = run_esmm_warpskipping(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_warpskipping_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 12: // Experimental double buffered approach to ESMM
        if (check_results) {
            res = run_esmm_buffered(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_buffered_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 13: // Experimental offset based A-Sparsity approach to ESMM
        if (check_results) {
            res = run_esmm_offsets(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, pattern);
        } else {
            res = run_esmm_offsets_no_check(rows, cols, inners, d_A, d_B, d_C, runs, pattern);
        }
        break;
    case 14: // cuBlas
        if (check_results) {
            run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
        } else {
            run_cuBlas_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        res = true; // Assume cuBLAS always succeeds
        break;
    default:
        cout << "Invalid kernel choice: " << kernel_choice << endl;
        return false;
    }

    if (verbose) {
        if (check_results) {
            cout << "  Status: " << (res ? "PASSED" : "FAILED") << endl;
        } else {
            cout << "  Status: COMPLETED (no verification)" << endl;
        }
    }

    return res;
}

int main(int argc, char *argv[]) {
    // Define Matrix Dims
    constexpr int rows = 4096;
    constexpr int cols = 4096;
    constexpr int inners = 4096;

    // Default values
    std::vector<int> kernel_choices = {10};
    int runs = 1;
    bool verbose = false;
    bool check_results = true;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--no-check" || arg == "-n") {
            check_results = false;
        } else if (i == 1) {
            if (isdigit(arg[0]) || arg == "all" || arg.find(',') != std::string::npos || arg.find('-') != std::string::npos) {
                kernel_choices = parse_kernel_selection(arg);
                if (kernel_choices.empty()) {
                    cout << "Error: Invalid kernel selection '" << arg << "'" << endl;
                    print_usage(argv[0]);
                    return 1;
                }
            }
        } else if (i == 2 && isdigit(arg[0])) {
            runs = atoi(argv[i]);
            if (runs <= 0) {
                cout << "Error: Number of runs must be positive" << endl;
                return 1;
            }
        }
    }

    if (verbose) {
        cout << "Matrix dimensions: " << rows << "x" << cols << " * " << cols << "x" << inners << endl;
        cout << "Number of runs per kernel: " << runs << endl;
        cout << "Result checking: " << (check_results ? "ENABLED" : "DISABLED") << endl;
        cout << "Kernels to run: ";
        for (size_t i = 0; i < kernel_choices.size(); i++) {
            cout << kernel_choices[i];
            if (i < kernel_choices.size() - 1) cout << ", ";
        }
        cout << endl << endl;
    }

    float *h_A = (float *)malloc(rows * inners * sizeof(float));
    float *h_B = (float *)malloc(inners * cols * sizeof(float));
    float *h_C = (float *)malloc(rows * cols * sizeof(float));
    float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

    constexpr std::string_view sparsity = "00000010";

    randomize_matrix_with_pattern(h_A, rows, inners, sparsity);
    randomize_matrix(h_B, inners, cols);
    memset(h_C, 0, rows * cols * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, rows * inners * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, inners * cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

    cudaCheckError(cudaMemcpy(d_A, h_A, rows * inners * sizeof(float),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, inners * cols * sizeof(float),
                              cudaMemcpyHostToDevice));

    if (check_results) {
        if (verbose) cout << "Generating CPU reference solution..." << endl;
        matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);
    } else {
        if (verbose) cout << "Skipping CPU reference solution (no-check mode)..." << endl;
    }

    int passed = 0;
    int total = kernel_choices.size();

    for (int kernel_choice : kernel_choices) {
        bool result = run_single_kernel(kernel_choice, rows, cols, inners,
                                       d_A, d_B, d_C, h_C, h_C_ref, runs, 
                                       verbose, check_results, sparsity);
        if (result || !check_results) passed++;

        if (!verbose) {
            cout << "Kernel " << kernel_choice << " (" << get_kernel_name(kernel_choice) << "): ";
            if (check_results) {
                cout << (result ? "PASSED" : "FAILED");
            } else {
                cout << "COMPLETED (no verification)";
            }
            cout << endl;
        }

        if (verbose && kernel_choice != kernel_choices.back()) {
            cout << endl;
        }
    }

    cout << endl << "Summary: " << passed << "/" << total << " kernels ";
    if (check_results) {
        cout << "passed" << endl;
    } else {
        cout << "completed (no verification)" << endl;
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (passed == total) ? 0 : 1;
}

