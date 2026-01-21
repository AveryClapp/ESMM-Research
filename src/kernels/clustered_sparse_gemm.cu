/**
 * Integration Example: Row Clustering + K25 Sparse GEMM
 * 
 * Shows how to combine the clustering preprocessor with your existing
 * K25 joint sparsity kernel for maximum block sparsity capture.
 * 
 * This file demonstrates the full pipeline:
 *   1. Analyze A and compute row clustering
 *   2. Check if clustering provides sufficient benefit
 *   3. Preprocess B patterns (unchanged)
 *   4. Run K25 on clustered A
 *   5. Unpermute output C
 */

#pragma once

#include "row_clustering_preprocessor.cuh"
// Include your existing headers:
// #include "src/preprocessors/ab_preprocessor.cu"
// #include "src/kernels/esmm_ab_8x32.cu"
// #include "adaptive_gemm_dispatcher.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace clustered_gemm {

// Forward declarations for your existing code
struct ABPatternMetadata;

// Placeholder for your existing B preprocessor
template <const int BK, const int WN>
ABPatternMetadata preprocess_b_patterns(const float* d_B, int K, int N);

// Placeholder for your existing K25 kernel
void run_k25_kernel(
    int M, int N, int K,
    const float* A, const float* B, float* C,
    const uint8_t* a_patterns, const uint8_t* b_patterns,
    int numKBlocks
);

// ============================================================================
// Configuration
// ============================================================================

struct ClusteredGemmConfig {
    // Clustering thresholds
    float min_element_sparsity = 0.5f;     // Don't cluster if <50% sparse
    float min_improvement_ratio = 1.3f;     // Need 30% more block sparsity
    float max_clustering_overhead_ms = 2.0f; // Max acceptable overhead
    
    // Block sizes (must match K25)
    int block_m = 8;
    int block_k = 32;
    int bk = 8;
    
    // Strategy
    row_clustering::ClusterConfig::Strategy strategy = 
        row_clustering::ClusterConfig::Strategy::SORT_BY_PATTERN;
};

// ============================================================================
// Main Pipeline
// ============================================================================

class ClusteredSparseGemmPipeline {
public:
    ClusteredSparseGemmPipeline(ClusteredGemmConfig config = {}) 
        : config_(config), cublas_handle_(nullptr) {
        cublasCreate(&cublas_handle_);
        
        row_clustering::ClusterConfig cluster_config;
        cluster_config.block_m = config.block_m;
        cluster_config.block_k = config.block_k;
        cluster_config.bk = config.bk;
        cluster_config.strategy = config.strategy;
        clusterer_ = std::make_unique<row_clustering::RowClusteringPreprocessor>(cluster_config);
    }
    
    ~ClusteredSparseGemmPipeline() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (cached_result_.d_A_clustered) cached_result_.free();
    }
    
    /**
     * Full GEMM with automatic clustering decision
     * 
     * @param d_A       Input matrix A [M × K]
     * @param d_B       Input matrix B [K × N]
     * @param d_C       Output matrix C [M × N]
     * @param M, N, K   Dimensions
     * @param reuse_clustering  If true, reuse previous clustering (for same A)
     * @param verbose   Print debug info
     * 
     * @return true if sparse path was used, false if cuBLAS
     */
    bool gemm(
        const float* d_A,
        const float* d_B,
        float* d_C,
        int M, int N, int K,
        bool reuse_clustering = false,
        bool verbose = false
    ) {
        // Step 1: Analyze and cluster A (or reuse cached)
        if (!reuse_clustering || !has_cached_clustering_) {
            analyze_and_cluster(d_A, M, K, verbose);
        }
        
        // Step 2: Decide path based on clustering results
        bool use_sparse = should_use_sparse_path(verbose);
        
        if (!use_sparse) {
            // Fall back to cuBLAS
            run_cublas(d_A, d_B, d_C, M, N, K);
            return false;
        }
        
        // Step 3: Run sparse path
        run_sparse_path(d_B, d_C, M, N, K, verbose);
        return true;
    }
    
    /**
     * Get clustering statistics from last run
     */
    const row_clustering::ClusteringResult& get_clustering_result() const {
        return cached_result_;
    }

private:
    ClusteredGemmConfig config_;
    cublasHandle_t cublas_handle_;
    std::unique_ptr<row_clustering::RowClusteringPreprocessor> clusterer_;
    
    row_clustering::ClusteringResult cached_result_ = {};
    bool has_cached_clustering_ = false;
    
    void analyze_and_cluster(const float* d_A, int M, int K, bool verbose) {
        if (has_cached_clustering_) {
            cached_result_.free();
        }
        
        cached_result_ = clusterer_->analyze_and_reorder(d_A, M, K, verbose);
        has_cached_clustering_ = true;
    }
    
    bool should_use_sparse_path(bool verbose) const {
        if (!has_cached_clustering_) return false;
        
        // Check improvement ratio
        bool good_improvement = cached_result_.improvement_ratio >= config_.min_improvement_ratio;
        
        // Check absolute block sparsity (need at least some to beat cuBLAS)
        bool enough_sparsity = cached_result_.clustered_block_sparsity >= 0.40f;
        
        // Check overhead isn't too high
        float total_overhead = cached_result_.analysis_time_ms + cached_result_.reorder_time_ms;
        bool acceptable_overhead = total_overhead <= config_.max_clustering_overhead_ms;
        
        bool use_sparse = good_improvement && enough_sparsity && acceptable_overhead;
        
        if (verbose) {
            printf("Sparse path decision:\n");
            printf("  Clustered block sparsity: %.1f%% (need ≥40%%)\n", 
                   cached_result_.clustered_block_sparsity * 100);
            printf("  Improvement ratio: %.2fx (need ≥%.2fx)\n",
                   cached_result_.improvement_ratio, config_.min_improvement_ratio);
            printf("  Clustering overhead: %.2f ms (max %.2f ms)\n",
                   total_overhead, config_.max_clustering_overhead_ms);
            printf("  Decision: %s\n", use_sparse ? "SPARSE" : "cuBLAS");
        }
        
        return use_sparse;
    }
    
    void run_cublas(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // cuBLAS uses column-major, so for row-major A×B:
        cublasSgemm(cublas_handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    
    void run_sparse_path(const float* d_B, float* d_C, int M, int N, int K, bool verbose) {
        // Allocate temporary for permuted output
        float* d_C_clustered;
        cudaMalloc(&d_C_clustered, M * N * sizeof(float));
        
        // Preprocess B (your existing preprocessor)
        // auto b_meta = preprocess_b_patterns<8, 32>(d_B, K, N);
        
        // For now, placeholder - you'd integrate your actual B preprocessor
        uint8_t* d_b_patterns = nullptr;  // = b_meta.d_b_patterns;
        int num_k_blocks = K / config_.bk;
        
        // Run K25 on clustered A
        // run_k25_kernel(M, N, K,
        //                cached_result_.d_A_clustered,
        //                d_B,
        //                d_C_clustered,
        //                cached_result_.d_a_patterns,
        //                d_b_patterns,
        //                num_k_blocks);
        
        // Placeholder: just copy for testing
        cudaMemcpy(d_C_clustered, d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Unpermute output
        clusterer_->unpermute_output(d_C_clustered, d_C, cached_result_, N);
        
        cudaFree(d_C_clustered);
        // cudaFree(d_b_patterns);
    }
};

// ============================================================================
// Simplified API
// ============================================================================

/**
 * One-shot clustered sparse GEMM
 * Use ClusteredSparseGemmPipeline for repeated operations with same A.
 */
inline bool clustered_sparse_gemm(
    const float* d_A,
    const float* d_B, 
    float* d_C,
    int M, int N, int K,
    bool verbose = false
) {
    ClusteredSparseGemmPipeline pipeline;
    return pipeline.gemm(d_A, d_B, d_C, M, N, K, false, verbose);
}

} // namespace clustered_gemm


// ============================================================================
// Example Usage
// ============================================================================

/*

#include "clustered_sparse_gemm.cuh"

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // ... initialize A, B with your data ...
    
    // Option 1: One-shot
    bool used_sparse = clustered_gemm::clustered_sparse_gemm(
        d_A, d_B, d_C, M, N, K, true);
    
    // Option 2: Pipeline for repeated operations (same A, different B)
    clustered_gemm::ClusteredSparseGemmPipeline pipeline;
    
    // First call analyzes A
    pipeline.gemm(d_A, d_B1, d_C1, M, N, K, false, true);
    
    // Subsequent calls reuse clustering
    pipeline.gemm(d_A, d_B2, d_C2, M, N, K, true, false);  // reuse_clustering=true
    pipeline.gemm(d_A, d_B3, d_C3, M, N, K, true, false);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

*/


