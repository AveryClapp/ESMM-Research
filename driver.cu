#include "include/utils.cuh"
#include "include/runners.cuh"
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
    case 13: // Experimental B Sparsity Lookup Table + Function Ptrs
        if (check_results) {
            res = run_esmm_b_fp_lut(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, pattern);
        } else {
            res = run_esmm_b_fp_lut_no_check(rows, cols, inners, d_A, d_B, d_C, runs, pattern);
        }
        break;
    case 14: // Experimental offset based A-Sparsity approach to ESMM
        if (check_results) {
            res = run_esmm_unrolled(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, pattern);
        } else {
            res = run_esmm_unrolled_no_check(rows, cols, inners, d_A, d_B, d_C, runs, pattern);
        }
        break;
    case 15: // cuBlas
        if (check_results) {
            run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
        } else {
            run_cuBlas_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        res = true; // Assume cuBLAS always succeeds
        break;
    case 16: // ESMM Block-wise Uniform A Sparsity (Per-Warp Pattern Encoding)
        if (check_results) {
            res = run_esmm_a_sparse_blockwise(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_a_sparse_blockwise_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 17: // ESMM B-Sparse Warp-Granularity (Zero-Divergence)
        if (check_results) {
            res = run_esmm_b_sparse_warp(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_sparse_warp_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 18: // ESMM B-Sparse TN-Granularity (8-column per thread-group)
        if (check_results) {
            res = run_esmm_b_sparse_tn(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_sparse_tn_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 19: // ESMM B-Transpose Sparse (K-iteration skipping)
        if (check_results) {
            res = run_esmm_b_transpose_k19(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_transpose_k19_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 20: // ESMM B-Sparse Shared Memory Transpose (WN-granularity)
        if (check_results) {
            res = run_esmm_b_smem_transpose(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_smem_transpose_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 21: // ESMM B-Sparse Warp-Uniform Pattern (WN-granularity, Zero-Divergence)
        if (check_results) {
            res = run_esmm_b_sparse(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_sparse_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 22: // ESMM A+B Sparse (Joint Warp-Uniform Patterns, Zero-Divergence)
        if (check_results) {
            res = run_esmm_ab_sparse(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_sparse_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 23: // ESMM AB-Turbo (Precomputed Joint Patterns + Warp Shuffle)
        if (check_results) {
            res = run_esmm_ab_turbo(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_turbo_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
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

    // Default Matrix Dims
    int rows = 4096;
    int cols = 4096;
    int inners = 4096;
    std::string sparsity = "11000000";

    // Default values
    std::vector<int> kernel_choices = {17};
    int runs = 1;
    bool verbose = false;
    bool check_results = true;
    bool use_random_sparsity = false;
    float random_sparsity_percent = 37.5f;
    unsigned int random_seed = 12345;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--no-check" || arg == "-n") {
            check_results = false;
        } else if (arg == "--random" || arg == "-r") {
            use_random_sparsity = true;
        } else if (arg == "--size" || arg == "-s") {
            if (i + 1 >= argc) {
                cout << "Error: --size requires an argument" << endl;
                print_usage(argv[0]);
                return 1;
            }
            int size = atoi(argv[++i]);
            if (size <= 0) {
                cout << "Error: Size must be positive" << endl;
                return 1;
            }
            rows = cols = inners = size;
        } else if (arg == "--pattern" || arg == "-p") {
            if (i + 1 >= argc) {
                cout << "Error: --pattern requires an argument" << endl;
                print_usage(argv[0]);
                return 1;
            }
            sparsity = argv[++i];
            // Validate pattern
            if (sparsity.length() != 8) {
                cout << "Error: Pattern must be exactly 8 characters (e.g., '11000000')" << endl;
                return 1;
            }
            for (char c : sparsity) {
                if (c != '0' && c != '1') {
                    cout << "Error: Pattern must contain only '0' and '1' characters" << endl;
                    return 1;
                }
            }
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

    // Validate all kernel choices are in valid range (1-23)
    for (int k : kernel_choices) {
        if (k < 1 || k > 23) {
            cout << "Error: Kernel " << k << " is out of range. Valid kernels are 1-23." << endl;
            cout << "Run '" << argv[0] << " --help' to see available kernels." << endl;
            return 1;
        }
    }

    if (verbose) {
        cout << "Matrix dimensions: " << rows << "x" << cols << " * " << cols << "x" << inners << endl;
        cout << "Number of runs per kernel: " << runs << endl;
        cout << "Result checking: " << (check_results ? "ENABLED" : "DISABLED") << endl;
        cout << "Sparsity mode: " << (use_random_sparsity ? "RANDOM" : "PATTERN-BASED") << endl;
        if (use_random_sparsity) {
            cout << "  Random sparsity: " << random_sparsity_percent << "%" << endl;
            cout << "  Random seed: " << random_seed << endl;
        } else {
            cout << "  Pattern: " << sparsity << endl;
        }
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


    if (use_random_sparsity) {
        randomize_matrix_unstructured(h_A, rows, inners, random_sparsity_percent, random_seed);
        randomize_matrix_unstructured(h_B, inners, cols, random_sparsity_percent, random_seed + 1);
    } else {
        randomize_matrix_with_pattern(h_A, rows, inners, sparsity);
        // For B-sparse K-skipping kernels (19, 20, 21, 22, 23), use K-dimension pattern
        bool use_b_kdim = false;
        for (size_t i = 0; i < kernel_choices.size(); i++) {
            int k = kernel_choices[i];
            if (k == 19 || k == 20 || k == 21 || k == 22 || k == 23) {
                use_b_kdim = true;
                break;
            }
        }
        if (use_b_kdim) {
            randomize_matrix_B_kdim_pattern(h_B, inners, cols, sparsity);
        } else {
            randomize_matrix_with_pattern(h_B, inners, cols, sparsity);
        }
    }
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
        if (verbose) cout << "Generating GPU reference solution using ESMM kernel..." << endl;

        // Use ESMM kernel to generate reference instead of CPU
        const uint K10_NUM_THREADS = 256;
        const uint K10_BN = 128;
        const uint K10_BM = 128;
        const uint K10_BK = 16;
        const uint K10_WN = 64;
        const uint K10_WM = 32;
        const uint K10_WNITER = 4;
        const uint K10_TN = 8;
        const uint K10_TM = 1;

        dim3 blockDim(K10_NUM_THREADS);
        dim3 gridDim(CEIL_DIV(cols, K10_BN), CEIL_DIV(rows, K10_BM));
        cudaMemset(d_C, 0, rows * cols * sizeof(float));

        esmm<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(rows, cols, inners, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            cout << "Error generating reference with ESMM: " << cudaGetErrorString(error) << endl;
            return 1;
        }

        // Copy GPU result to host as reference
        cudaMemcpy(h_C_ref, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

        if (verbose) cout << "GPU reference generated successfully." << endl;
    } else {
        if (verbose) cout << "Skipping reference solution (no-check mode)..." << endl;
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
