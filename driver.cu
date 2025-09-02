#include "runners.cuh"
#include "utils.cuh"
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

using std::cout;
using std::endl;
using std::cin;



std::vector<int> parse_kernel_selection(const std::string& input) {
    std::vector<int> kernels;
    if (input == "all") {
        for (int i = 1; i <= 12; i++) {
            kernels.push_back(i);
        }
        return kernels;
    }
    // Check if it's a range (e.g., "1-5")
    size_t dash_pos = input.find('-');
    if (dash_pos != std::string::npos) {
        int start = std::stoi(input.substr(0, dash_pos));
        int end = std::stoi(input.substr(dash_pos + 1));
        for (int i = start; i <= end && i <= 12; i++) {
            kernels.push_back(i);
        }
        return kernels;
    }
    // Check if it's comma-separated (e.g., "1,3,5")
    std::stringstream ss(input);
    std::string kernel_str;
    while (std::getline(ss, kernel_str, ',')) {
        int kernel = std::stoi(kernel_str);
        if (kernel >= 1 && kernel <= 12) {
            kernels.push_back(kernel);
        }
    }
    return kernels;
}

const char* get_kernel_name(int kernel_choice) {
    switch (kernel_choice) {
        case 1: return "Naive Implementation";
        case 2: return "Global Memory Coalescing";
        case 3: return "Shared Memory Blocks";
        case 4: return "One Dimensional Blocktiling";
        case 5: return "Two Dimensional Blocktiling";
        case 6: return "Vectorized Memory Accessing";
        case 7: return "1D Vectorized Approach";
        case 8: return "Basic Warptiling";
        case 9: return "1D Warptiling";
        case 10: return "Emergent Sparsity Matrix Multiplication (ESMM)";
        case 11: return "ESMM Warpskipping";
        case 12: return "cuBLAS";
        default: return "Unknown Kernel";
    }
}

bool run_single_kernel(int kernel_choice, int rows, int cols, int inners, 
                      float* d_A, float* d_B, float* d_C, 
                      float* h_C, float* h_C_ref, int runs, bool verbose) {
    bool res = false;

    // Reset d_C to zeros before each kernel
    cudaCheckError(cudaMemset(d_C, 0, rows * cols * sizeof(float)));

    if (verbose) {
        cout << "Running kernel " << kernel_choice << ": " << get_kernel_name(kernel_choice) << endl;
    }

    switch (kernel_choice) {
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
        res = run_vectorized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 7: // 1 Dimensional Vectorized Approach
        res = run_1d_vec(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 8: // Basic Warptiling
        res = run_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 9: // 1-Dimensional Warptiling
        res = run_1d_warptiling(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 10: // Emergent Sparsity Matrix Multiplication (our kernel)
        res = run_esmm(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 11: // Experimental warpskipping approach to ESMM
        //res = run_esmm_warpskipping(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        break;
    case 12: // cuBlas
        run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
        res = true; // Assume cuBLAS always succeeds
        break;
    default:
        cout << "Invalid kernel choice: " << kernel_choice << endl;
        return false;
    }

    if (verbose) {
        cout << "  Status: " << (res ? "PASSED" : "FAILED") << endl;
    }

    return res;
}

int main(int argc, char *argv[]) {
    // Define Matrix Dims
    constexpr int rows = 1024;
    constexpr int cols = 1024;
    constexpr int inners = 1024;

    // Default values
    std::vector<int> kernel_choices = {6}; // Default to kernel 6
    int runs = 1;
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (i == 1) {
            // First non-flag argument is kernel choice
            if (isdigit(arg[0]) || arg == "all" || arg.find(',') != std::string::npos || arg.find('-') != std::string::npos) {
                kernel_choices = parse_kernel_selection(arg);
                if (kernel_choices.empty()) {
                    cout << "Error: Invalid kernel selection '" << arg << "'" << endl;
                    print_usage(argv[0]);
                    return 1;
                }
            }
        } else if (i == 2) {
            // Second non-flag argument is runs
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
        cout << "Kernels to run: ";
        for (size_t i = 0; i < kernel_choices.size(); i++) {
            cout << kernel_choices[i];
            if (i < kernel_choices.size() - 1) cout << ", ";
        }
        cout << endl << endl;
    }

    // Allocate host matrices
    float *h_A = (float *)malloc(rows * inners * sizeof(float));
    float *h_B = (float *)malloc(inners * cols * sizeof(float));
    float *h_C = (float *)malloc(rows * cols * sizeof(float));
    float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

    // Generate random data w/ given sparsity:
    std::vector<int> sparsity = stringToVector("10101010");

    // Generate A matrix
    randomize_matrix_with_pattern(h_A, rows, inners, sparsity);
    // Generate B matrix
    randomize_matrix(h_B, inners, cols);
    // Set h_C to zeros
    memset(h_C, 0, rows * cols * sizeof(float));

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, rows * inners * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, inners * cols * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_C, rows * cols * sizeof(float)));

    // Copy random data to device matrices
    cudaCheckError(cudaMemcpy(d_A, h_A, rows * inners * sizeof(float),
                              cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, inners * cols * sizeof(float),
                              cudaMemcpyHostToDevice));

    // Generate reference solution on CPU
    if (verbose) cout << "Generating CPU reference solution..." << endl;
    matrixMultiplyCPU(h_A, h_B, h_C_ref, rows, cols, inners);

    // Run selected kernels
    int passed = 0;
    int total = kernel_choices.size();

    for (int kernel_choice : kernel_choices) {
        bool result = run_single_kernel(kernel_choice, rows, cols, inners,
                                       d_A, d_B, d_C, h_C, h_C_ref, runs, verbose);
        if (result) passed++;
        if (!verbose) {
            cout << "Kernel " << kernel_choice << " (" << get_kernel_name(kernel_choice) 
                 << "): " << (result ? "PASSED" : "FAILED") << endl;
        }
        if (verbose && kernel_choice != kernel_choices.back()) {
            cout << endl;
        }
    }

    cout << endl << "Summary: " << passed << "/" << total << " kernels passed" << endl;
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (passed == total) ? 0 : 1;
}
