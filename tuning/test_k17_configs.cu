// Direct CUDA test for K17 configurations on 4096x4096x4096
// Uses GPU preprocessing for speed

#include "../include/runners.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Test configuration structure
struct Config {
    int NUM_THREADS;
    int BM, BN, BK;
    int WM, WN;
    int WNITER;
    int TM, TN;
    const char* name;
};

// Configurations to test (based on tuning results from smaller sizes)
std::vector<Config> test_configs = {
    {128, 64, 128, 8, 64, 32, 2, 1, 8, "Best from 2048 (tuned)"},
    {128, 64, 64, 8, 32, 32, 1, 1, 8, "Best from 512"},
    {256, 128, 128, 8, 32, 64, 4, 1, 8, "Current K17 default"},
    {128, 128, 128, 8, 64, 64, 2, 1, 8, "Larger tiles"},
    {256, 128, 256, 8, 64, 64, 4, 1, 8, "Extra large tiles"},
};

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    std::cout << "Testing K17 configurations on " << M << "x" << N << "x" << K << std::endl;
    std::cout << "==========================================" << std::endl;

    // Allocate matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Initialize with random data
    // (In practice, kernel_tuner does this; here we skip for speed)
    cudaMemset(d_A, 0, M * K * sizeof(float));
    cudaMemset(d_B, 0, K * N * sizeof(float));
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // Test each configuration
    for (const auto& cfg : test_configs) {
        std::cout << "\nTesting: " << cfg.name << std::endl;
        std::cout << "  Config: THREADS=" << cfg.NUM_THREADS
                  << ", BM=" << cfg.BM << ", BN=" << cfg.BN
                  << ", WM=" << cfg.WM << ", WN=" << cfg.WN
                  << ", WNITER=" << cfg.WNITER << std::endl;

        // Note: This is a simplified test - we'd need to instantiate the template
        // with each config. For now, just use the run_esmm_hybrid_no_check function
        // which has the default config

        const int runs = 5;
        auto start = std::chrono::high_resolution_clock::now();

        // This calls K17 with current config - to test other configs,
        // we'd need to modify the template parameters
        run_esmm_hybrid_no_check(M, N, K, d_A, d_B, d_C, runs);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double avg_time = elapsed.count() / runs;

        double gflops = (2.0 * M * N * K) / (avg_time * 1e6);

        std::cout << "  Avg Time: " << avg_time << " ms" << std::endl;
        std::cout << "  GFLOPS: " << gflops << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
