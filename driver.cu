#include "include/utils.cuh"
#include "include/runners.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <string_view>

using std::cout;
using std::endl;
using std::cin;

static const std::set<int> VALID_KERNELS = {14, 15, 16, 17, 20, 21, 25, 26, 27, 28, 29};

bool run_single_kernel(int kernel_choice, int rows, int cols, int inners,
                      float* d_A, float* d_B, float* d_C,
                      float* h_C, float* h_C_ref, int runs,
                      bool verbose, bool check_results, std::string_view pattern,
                      bool skip_stats = false) {
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
    case 14: // A-Only Sparse (Offset List, Templated Dispatch)
        if (check_results) {
            res = run_esmm_unrolled(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, pattern);
        } else {
            res = run_esmm_unrolled_no_check(rows, cols, inners, d_A, d_B, d_C, runs, pattern);
        }
        break;
    case 15: // cuBLAS baseline
        if (check_results) {
            run_cuBlas(rows, cols, inners, d_A, d_B, d_C, h_C, runs);
        } else {
            run_cuBlas_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        res = true; // cuBLAS always succeeds
        break;
    case 16: // A-Sparse Block-wise (Warp-Granularity Patterns)
        if (check_results) {
            res = run_esmm_a_sparse_blockwise(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_a_sparse_blockwise_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 17: // B-Sparse Warp-Granularity (32-col, Zero-Divergence)
        if (check_results) {
            res = run_esmm_b_sparse_warp(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_b_sparse_warp_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 20: // A+B Sparse OPTIMIZED (Zero-Overhead Inner Loop)
        if (check_results) {
            res = run_esmm_ab_sparse_optimized(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_sparse_optimized_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 21: // A+B Sparse - 8x32 Granularity
        if (check_results) {
            res = run_esmm_ab_8x32(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_8x32_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 25: // A+B Simple Fused (Main Contribution)
        if (check_results) {
            res = run_esmm_ab_simple_fused(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_simple_fused_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 26: // A+B Optimized V2 (K20 + 32-row + Block Skip + Float4)
        if (check_results) {
            res = run_esmm_ab_optimized_v2(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_optimized_v2_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 27: // Ablation: 32-row only, no block skip, no float4
        if (check_results) {
            res = run_esmm_ab_optimized_v2_baseline(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs);
        } else {
            res = run_esmm_ab_optimized_v2_baseline_no_check(rows, cols, inners, d_A, d_B, d_C, runs);
        }
        break;
    case 28: // K25 with 32-row granularity (gmem patterns, block+warp skip, float4)
        if (check_results) {
            res = run_esmm_ab_gmem_32(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true, skip_stats);
        } else {
            res = run_esmm_ab_gmem_32_no_check(rows, cols, inners, d_A, d_B, d_C, runs, skip_stats);
        }
        break;
    case 29: // K26 + templated MAX_K_BLOCKS + float2 A-loads
        if (check_results) {
            res = run_esmm_ab_optimized_v3(rows, cols, inners, d_A, d_B, d_C, h_C, h_C_ref, runs, true, skip_stats);
        } else {
            res = run_esmm_ab_optimized_v3_no_check(rows, cols, inners, d_A, d_B, d_C, runs, skip_stats);
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
    std::string sparsity_a = "";  // Empty means use 'sparsity'
    std::string sparsity_b = "";  // Empty means use 'sparsity'

    // Default values
    std::vector<int> kernel_choices = {17};
    int runs = 1;
    bool verbose = false;
    bool check_results = true;
    bool use_random_sparsity = false;
    bool use_blockwise_sparsity = false;
    float random_sparsity_percent = 37.5f;
    unsigned int random_seed = 12345;

    bool skip_stats = false;

    // Real weight loading
    std::string load_matrix_a = "";
    std::string load_matrix_b = "";
    int load_M = 0, load_K = 0, load_N = 0;

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
        } else if (arg == "--blockwise" || arg == "-b") {
            use_blockwise_sparsity = true;
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
        } else if (arg == "--pattern-a" || arg == "-pa") {
            if (i + 1 >= argc) {
                cout << "Error: --pattern-a requires an argument" << endl;
                print_usage(argv[0]);
                return 1;
            }
            sparsity_a = argv[++i];
            if (sparsity_a.length() != 8) {
                cout << "Error: Pattern must be exactly 8 characters (e.g., '11000000')" << endl;
                return 1;
            }
            for (char c : sparsity_a) {
                if (c != '0' && c != '1') {
                    cout << "Error: Pattern must contain only '0' and '1' characters" << endl;
                    return 1;
                }
            }
        } else if (arg == "--pattern-b" || arg == "-pb") {
            if (i + 1 >= argc) {
                cout << "Error: --pattern-b requires an argument" << endl;
                print_usage(argv[0]);
                return 1;
            }
            sparsity_b = argv[++i];
            if (sparsity_b.length() != 8) {
                cout << "Error: Pattern must be exactly 8 characters (e.g., '11000000')" << endl;
                return 1;
            }
            for (char c : sparsity_b) {
                if (c != '0' && c != '1') {
                    cout << "Error: Pattern must contain only '0' and '1' characters" << endl;
                    return 1;
                }
            }
        } else if (arg == "--load-a") {
            if (i + 1 >= argc) { cout << "Error: --load-a requires a path" << endl; return 1; }
            load_matrix_a = argv[++i];
        } else if (arg == "--load-b") {
            if (i + 1 >= argc) { cout << "Error: --load-b requires a path" << endl; return 1; }
            load_matrix_b = argv[++i];
        } else if (arg == "--skip-stats") {
            skip_stats = true;
        } else if (arg == "--dims") {
            if (i + 3 >= argc) { cout << "Error: --dims requires M K N" << endl; return 1; }
            load_M = atoi(argv[++i]);
            load_K = atoi(argv[++i]);
            load_N = atoi(argv[++i]);
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

    // Validate all kernel choices are in valid set
    for (int k : kernel_choices) {
        if (VALID_KERNELS.find(k) == VALID_KERNELS.end()) {
            cout << "Error: Kernel " << k << " is not a valid kernel. Valid kernels are: 14, 15, 16, 17, 20, 21, 25." << endl;
            cout << "Run '" << argv[0] << " --help' to see available kernels." << endl;
            return 1;
        }
    }

    // If separate A/B patterns not specified, use the unified pattern
    if (sparsity_a.empty()) sparsity_a = sparsity;
    if (sparsity_b.empty()) sparsity_b = sparsity;

    if (verbose) {
        cout << "Matrix dimensions: " << rows << "x" << cols << " * " << cols << "x" << inners << endl;
        cout << "Number of runs per kernel: " << runs << endl;
        cout << "Result checking: " << (check_results ? "ENABLED" : "DISABLED") << endl;

        if (use_blockwise_sparsity) {
            cout << "Sparsity mode: BLOCK-LEVEL (warp-uniform patterns)" << endl;
            if (sparsity_a == sparsity_b) {
                cout << "  Block sparsity: " << sparsity_a << " (applied at BK=8 granularity)" << endl;
            } else {
                cout << "  A-matrix pattern: " << sparsity_a << " (applied at BK=8 granularity)" << endl;
                cout << "  B-matrix pattern: " << sparsity_b << " (applied at BK=8 granularity)" << endl;
            }
        } else if (use_random_sparsity) {
            cout << "Sparsity mode: RANDOM (unstructured)" << endl;
            cout << "  Random sparsity: " << random_sparsity_percent << "%" << endl;
            cout << "  Random seed: " << random_seed << endl;
        } else {
            cout << "Sparsity mode: PATTERN-BASED (column-wise)" << endl;
            if (sparsity_a == sparsity_b) {
                cout << "  Pattern: " << sparsity_a << endl;
            } else {
                cout << "  A-matrix pattern: " << sparsity_a << endl;
                cout << "  B-matrix pattern: " << sparsity_b << endl;
            }
        }

        cout << "Kernels to run: ";
        for (size_t i = 0; i < kernel_choices.size(); i++) {
            cout << kernel_choices[i];
            if (i < kernel_choices.size() - 1) cout << ", ";
        }
        cout << endl << endl;
    }

    // Apply loaded-matrix dimensions before allocating
    bool load_mode = !load_matrix_a.empty() || !load_matrix_b.empty();
    if (load_mode) {
        if (load_M <= 0 || load_K <= 0 || load_N <= 0) {
            cout << "Error: --dims M K N required with --load-a / --load-b" << endl;
            return 1;
        }
        rows = load_M; inners = load_K; cols = load_N;
    }

    float *h_A = (float *)malloc(rows * inners * sizeof(float));
    float *h_B = (float *)malloc(inners * cols * sizeof(float));
    float *h_C = (float *)malloc(rows * cols * sizeof(float));
    float *h_C_ref = (float *)malloc(rows * cols * sizeof(float));

    // Generate sparsity patterns based on mode
    if (load_mode) {
        auto load_bin = [](const std::string& path, float* buf, size_t n) -> bool {
            if (path.empty()) return true;
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) { printf("Error: cannot open %s\n", path.c_str()); return false; }
            size_t nread = fread(buf, sizeof(float), n, f);
            fclose(f);
            if (nread != n) {
                printf("Error: expected %zu floats, got %zu from %s\n", n, nread, path.c_str());
                return false;
            }
            return true;
        };
        if (!load_bin(load_matrix_a, h_A, (size_t)rows * inners)) return 1;
        if (!load_bin(load_matrix_b, h_B, (size_t)inners * cols)) return 1;
        // If A not loaded, fill with random dense values (not zero — zero A skips everything)
        if (load_matrix_a.empty()) {
            srand(random_seed);
            for (size_t i = 0; i < (size_t)rows * inners; i++)
                h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        if (verbose) {
            if (!load_matrix_a.empty()) cout << "Loaded A from: " << load_matrix_a << endl;
            else cout << "A: random dense (seed=" << random_seed << ")" << endl;
            if (!load_matrix_b.empty()) cout << "Loaded B from: " << load_matrix_b << endl;
        }
    } else if (use_blockwise_sparsity) {
        if (verbose) cout << "Generating block-level sparsity patterns..." << endl;

        int ones_count_a = 0;
        for (char c : sparsity_a) { if (c == '1') ones_count_a++; }
        float density_percent_a = (ones_count_a / 8.0f) * 100.0f;
        float block_sparsity_percent_a = 100.0f - density_percent_a;

        int ones_count_b = 0;
        for (char c : sparsity_b) { if (c == '1') ones_count_b++; }
        float density_percent_b = (ones_count_b / 8.0f) * 100.0f;
        float block_sparsity_percent_b = 100.0f - density_percent_b;

        for (int k : kernel_choices) {
            if (k == 21) {
                // 8-row A granularity for K21 (8x32 fine-grained)
                randomize_matrix_A<8, 8>(h_A, rows, inners, block_sparsity_percent_a, random_seed);
            } else {
                // 64-row A granularity for K14, K16, K17, K20, K25
                randomize_matrix_A_blocklevel<8, 32>(h_A, rows, inners, block_sparsity_percent_a, random_seed);
            }
        }

        randomize_matrix_B_blocklevel_fixed<8, 32>(h_B, inners, cols, block_sparsity_percent_b, random_seed + 1);

        if (verbose) {
            if (block_sparsity_percent_a == block_sparsity_percent_b) {
                cout << "Block-level sparsity generated (target: " << block_sparsity_percent_a << "%)" << endl;
            } else {
                cout << "Block-level sparsity generated:" << endl;
                cout << "  A-matrix: " << block_sparsity_percent_a << "% sparsity (" << density_percent_a << "% density)" << endl;
                cout << "  B-matrix: " << block_sparsity_percent_b << "% sparsity (" << density_percent_b << "% density)" << endl;
            }
        }
    } else if (use_random_sparsity) {
        randomize_matrix_unstructured(h_A, rows, inners, random_sparsity_percent, random_seed);
        randomize_matrix_unstructured(h_B, inners, cols, random_sparsity_percent, random_seed + 1);
    } else {
        randomize_matrix_with_pattern(h_A, rows, inners, sparsity_a);
        randomize_matrix_with_pattern(h_B, inners, cols, sparsity_b);
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
        if (verbose) cout << "Generating GPU reference solution using cuBLAS..." << endl;

        cublasHandle_t ref_handle;
        cublasCreate(&ref_handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaMemset(d_C, 0, rows * cols * sizeof(float));

        cublasStatus_t status = cublasSgemm(ref_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            cols, rows, inners,
                                            &alpha, d_B, cols, d_A, inners,
                                            &beta, d_C, cols);
        cudaDeviceSynchronize();
        cublasDestroy(ref_handle);

        if (status != CUBLAS_STATUS_SUCCESS) {
            cout << "Error generating reference with cuBLAS: status=" << status << endl;
            return 1;
        }

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
                                       verbose, check_results, sparsity, skip_stats);
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
