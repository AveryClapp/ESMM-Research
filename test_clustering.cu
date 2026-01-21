/**
 * Test driver for Row Clustering Preprocessor
 * 
 * Build:
 *   nvcc -O3 -o test_clustering test_clustering.cu -arch=sm_80
 * 
 * Usage:
 *   ./test_clustering [M] [K] [element_sparsity]
 *   ./test_clustering 4096 4096 0.9    # 90% element sparsity
 */

#include "src/preprocessors/row_clustering_preprocessor.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>

// Generate matrix with random element-wise sparsity
__global__ void generate_sparse_matrix_kernel(
    float* A, int M, int K, 
    float* rand_vals,
    float sparsity_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * K) {
        A[idx] = (rand_vals[idx] < sparsity_threshold) ? 0.0f : (rand_vals[idx] - 0.5f) * 2.0f;
    }
}

void generate_sparse_matrix(float* d_A, int M, int K, float element_sparsity) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 12345);
    
    float* d_rand;
    cudaMalloc(&d_rand, M * K * sizeof(float));
    curandGenerateUniform(gen, d_rand, M * K);
    
    int threads = 256;
    int blocks = (M * K + threads - 1) / threads;
    generate_sparse_matrix_kernel<<<blocks, threads>>>(
        d_A, M, K, d_rand, element_sparsity);
    
    cudaFree(d_rand);
    curandDestroyGenerator(gen);
}

// Measure actual element sparsity
float measure_element_sparsity(const float* d_A, int M, int K) {
    std::vector<float> h_A(M * K);
    cudaMemcpy(h_A.data(), d_A, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    int zeros = 0;
    for (int i = 0; i < M * K; i++) {
        if (h_A[i] == 0.0f) zeros++;
    }
    return static_cast<float>(zeros) / (M * K);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int K = (argc > 2) ? atoi(argv[2]) : 4096;
    float target_sparsity = (argc > 3) ? atof(argv[3]) : 0.9f;
    
    printf("==============================================\n");
    printf("Row Clustering Preprocessor Test\n");
    printf("==============================================\n");
    printf("Matrix size: %d × %d\n", M, K);
    printf("Target element sparsity: %.1f%%\n", target_sparsity * 100);
    printf("\n");
    
    // Allocate and generate sparse matrix
    float* d_A;
    cudaMalloc(&d_A, M * K * sizeof(float));
    generate_sparse_matrix(d_A, M, K, target_sparsity);
    
    float actual_sparsity = measure_element_sparsity(d_A, M, K);
    printf("Actual element sparsity: %.1f%%\n", actual_sparsity * 100);
    printf("\n");
    
    // Test all three clustering strategies
    using namespace row_clustering;
    
    std::vector<std::pair<ClusterConfig::Strategy, const char*>> strategies = {
        {ClusterConfig::Strategy::SORT_BY_DENSITY, "SORT_BY_DENSITY"},
        {ClusterConfig::Strategy::SORT_BY_PATTERN, "SORT_BY_PATTERN"},
        {ClusterConfig::Strategy::CLUSTER_SIMILAR, "CLUSTER_SIMILAR"},
    };
    
    for (auto& [strategy, name] : strategies) {
        printf("----------------------------------------------\n");
        printf("Strategy: %s\n", name);
        printf("----------------------------------------------\n");
        
        ClusterConfig config;
        config.strategy = strategy;
        config.block_m = 8;
        config.block_k = 32;
        
        RowClusteringPreprocessor clusterer(config);
        ClusteringResult result = clusterer.analyze_and_reorder(d_A, M, K, true);
        
        printf("\n");
        result.free();
    }
    
    // Benchmark overhead
    printf("==============================================\n");
    printf("Overhead Benchmark (10 iterations)\n");
    printf("==============================================\n");
    
    ClusterConfig config;
    config.strategy = ClusterConfig::Strategy::SORT_BY_PATTERN;
    RowClusteringPreprocessor clusterer(config);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    auto result = clusterer.analyze_and_reorder(d_A, M, K, false);
    result.free();
    
    // Benchmark
    const int ITERS = 10;
    float total_analysis = 0, total_reorder = 0;
    
    for (int i = 0; i < ITERS; i++) {
        result = clusterer.analyze_and_reorder(d_A, M, K, false);
        total_analysis += result.analysis_time_ms;
        total_reorder += result.reorder_time_ms;
        result.free();
    }
    
    printf("Average analysis time: %.3f ms\n", total_analysis / ITERS);
    printf("Average reorder time: %.3f ms\n", total_reorder / ITERS);
    printf("Total preprocessing: %.3f ms\n", (total_analysis + total_reorder) / ITERS);
    
    // Compare against your K25 preprocessing time (~400µs)
    printf("\nFor reference:\n");
    printf("  K25 pattern preprocessing: ~0.4 ms\n");
    printf("  Clustering adds: ~%.1f ms overhead\n", 
           (total_analysis + total_reorder) / ITERS - 0.4);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    
    return 0;
}

