#pragma once

/**
 * Row Clustering Preprocessor for Block Sparsity Optimization
 * 
 * Reorders rows of matrix A to maximize 8×32 block sparsity capture.
 * Integrates with existing K25 sparse GEMM pipeline.
 * 
 * Pipeline:
 *   1. Analyze: Compute per-row sparsity signatures (GPU)
 *   2. Cluster: Sort/group rows by pattern similarity (CPU)
 *   3. Permute: Gather A rows into clustered order (GPU)
 *   4. Encode: Run standard pattern preprocessing on clustered A
 *   5. GEMM: Run K25 on clustered data
 *   6. Unpermute: Scatter C rows back to original order (GPU)
 * 
 * Usage:
 *   RowClusteringPreprocessor clusterer;
 *   auto result = clusterer.analyze_and_reorder(d_A, M, K);
 *   // result.d_A_clustered is ready for K25
 *   // After GEMM, call clusterer.unpermute_output(d_C_clustered, d_C, M, N)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>

namespace row_clustering {

// ============================================================================
// Configuration
// ============================================================================

struct ClusterConfig {
    int block_m = 8;      // Row granularity for blocks
    int block_k = 32;     // Column granularity for blocks (matches K25's WN)
    int bk = 8;           // K-tile size for pattern encoding
    
    // Clustering strategy
    enum class Strategy {
        SORT_BY_DENSITY,      // Simple: sort rows by total zero count
        SORT_BY_PATTERN,      // Better: sort by which K-blocks are zero
        CLUSTER_SIMILAR,      // Best: group rows with similar patterns
    };
    Strategy strategy = Strategy::SORT_BY_PATTERN;
    
    // Threshold for "zero" (handles numerical noise)
    float zero_threshold = 0.0f;
};

// ============================================================================
// Result structure
// ============================================================================

struct ClusteringResult {
    // Device pointers
    float* d_A_clustered;       // Reordered A matrix [M × K]
    int* d_row_perm;            // Permutation: new_row -> old_row [M]
    int* d_row_inv_perm;        // Inverse: old_row -> new_row [M]
    uint8_t* d_a_patterns;      // Sparsity patterns for clustered A
    
    // Dimensions
    int M, K;
    int num_row_tiles;          // M / block_m
    int num_k_blocks;           // K / bk
    
    // Statistics
    float original_block_sparsity;   // Before clustering
    float clustered_block_sparsity;  // After clustering
    float improvement_ratio;
    float analysis_time_ms;
    float reorder_time_ms;
    
    void free() {
        if (d_A_clustered) cudaFree(d_A_clustered);
        if (d_row_perm) cudaFree(d_row_perm);
        if (d_row_inv_perm) cudaFree(d_row_inv_perm);
        if (d_a_patterns) cudaFree(d_a_patterns);
        d_A_clustered = nullptr;
        d_row_perm = nullptr;
        d_row_inv_perm = nullptr;
        d_a_patterns = nullptr;
    }
};

// ============================================================================
// GPU Kernels: Row Analysis
// ============================================================================

/**
 * Compute per-row density (number of non-zeros)
 * Launch: <<<M, 256>>>
 */
__global__ void compute_row_density_kernel(
    const float* __restrict__ A,
    int M, int K,
    int* __restrict__ row_nnz,      // Output: non-zeros per row
    float zero_threshold
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    int nnz = 0;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        float val = A[row * K + j];
        if (fabsf(val) > zero_threshold) nnz++;
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        nnz += __shfl_down_sync(0xFFFFFFFF, nnz, offset);
    }
    
    // Block reduction (if blockDim > 32)
    __shared__ int warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) warp_sums[warp_id] = nnz;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) {
            total += warp_sums[w];
        }
        row_nnz[row] = total;
    }
}

/**
 * Compute per-row block pattern signature
 * Each row gets a 64-bit signature indicating which 32-col blocks are non-zero
 * Launch: <<<M, 256>>>
 */
__global__ void compute_row_block_signature_kernel(
    const float* __restrict__ A,
    int M, int K,
    int block_k,                    // Usually 32
    uint64_t* __restrict__ signatures,
    float zero_threshold
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const int num_blocks = (K + block_k - 1) / block_k;
    const int max_sig_blocks = 64;  // Signature can encode up to 64 blocks
    
    // Each thread checks one or more K-blocks
    uint64_t local_sig = 0;
    
    for (int b = threadIdx.x; b < num_blocks && b < max_sig_blocks; b += blockDim.x) {
        int k_start = b * block_k;
        int k_end = min(k_start + block_k, K);
        
        bool has_nonzero = false;
        for (int k = k_start; k < k_end; k++) {
            if (fabsf(A[row * K + k]) > zero_threshold) {
                has_nonzero = true;
                break;
            }
        }
        
        if (has_nonzero) {
            local_sig |= (1ULL << b);
        }
    }
    
    // OR-reduce across threads
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sig |= __shfl_down_sync(0xFFFFFFFF, local_sig, offset);
    }
    
    __shared__ uint64_t warp_sigs[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) warp_sigs[warp_id] = local_sig;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        uint64_t final_sig = 0;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) {
            final_sig |= warp_sigs[w];
        }
        signatures[row] = final_sig;
    }
}

/**
 * Count zero 8×32 blocks (for statistics)
 * Launch: <<<num_row_tiles, 256>>>
 */
__global__ void count_zero_blocks_kernel(
    const float* __restrict__ A,
    int M, int K,
    int block_m, int block_k,
    int* __restrict__ zero_block_count,
    float zero_threshold
) {
    int tile_row = blockIdx.x;
    int row_start = tile_row * block_m;
    if (row_start >= M) return;
    
    int num_k_blocks = (K + block_k - 1) / block_k;
    int local_zero_blocks = 0;
    
    // Each thread checks one K-block across all block_m rows
    for (int kb = threadIdx.x; kb < num_k_blocks; kb += blockDim.x) {
        int k_start = kb * block_k;
        int k_end = min(k_start + block_k, K);
        
        bool block_is_zero = true;
        
        for (int r = 0; r < block_m && (row_start + r) < M; r++) {
            for (int k = k_start; k < k_end; k++) {
                if (fabsf(A[(row_start + r) * K + k]) > zero_threshold) {
                    block_is_zero = false;
                    break;
                }
            }
            if (!block_is_zero) break;
        }
        
        if (block_is_zero) local_zero_blocks++;
    }
    
    // Reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        local_zero_blocks += __shfl_down_sync(0xFFFFFFFF, local_zero_blocks, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(zero_block_count, local_zero_blocks);
    }
}

// ============================================================================
// GPU Kernels: Permutation
// ============================================================================

/**
 * Apply row permutation (gather): A_out[i, :] = A_in[perm[i], :]
 * Launch: <<<M, 256>>>
 */
__global__ void gather_rows_kernel(
    const float* __restrict__ A_in,
    float* __restrict__ A_out,
    const int* __restrict__ perm,
    int M, int K
) {
    int new_row = blockIdx.x;
    if (new_row >= M) return;
    
    int old_row = perm[new_row];
    
    // Vectorized copy (float4 = 16 bytes)
    int k = threadIdx.x * 4;
    while (k + 3 < K) {
        float4 val = *reinterpret_cast<const float4*>(&A_in[old_row * K + k]);
        *reinterpret_cast<float4*>(&A_out[new_row * K + k]) = val;
        k += blockDim.x * 4;
    }
    
    // Handle remainder
    k = threadIdx.x + (K / 4) * 4;
    while (k < K) {
        A_out[new_row * K + k] = A_in[old_row * K + k];
        k += blockDim.x;
    }
}

/**
 * Apply inverse row permutation (scatter): C_out[perm[i], :] = C_in[i, :]
 * Launch: <<<M, 256>>>
 */
__global__ void scatter_rows_kernel(
    const float* __restrict__ C_in,
    float* __restrict__ C_out,
    const int* __restrict__ perm,  // perm[new_idx] = old_idx
    int M, int N
) {
    int new_row = blockIdx.x;
    if (new_row >= M) return;
    
    int old_row = perm[new_row];
    
    // Vectorized copy
    int n = threadIdx.x * 4;
    while (n + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&C_in[new_row * N + n]);
        *reinterpret_cast<float4*>(&C_out[old_row * N + n]) = val;
        n += blockDim.x * 4;
    }
    
    // Handle remainder
    n = threadIdx.x + (N / 4) * 4;
    while (n < N) {
        C_out[old_row * N + n] = C_in[new_row * N + n];
        n += blockDim.x;
    }
}

// ============================================================================
// GPU Kernel: Pattern Encoding (for clustered A)
// ============================================================================

/**
 * Encode 8-bit patterns for each 8-row tile × K-block
 * Same as your existing preprocessor, but operates on clustered A
 * Launch: <<<num_row_tiles, 256>>>
 */
__global__ void encode_clustered_patterns_kernel(
    const float* __restrict__ A_clustered,
    uint8_t* __restrict__ patterns,
    int M, int K,
    int block_m,    // 8
    int bk,         // 8
    float zero_threshold
) {
    int tile_row = blockIdx.x;
    int row_start = tile_row * block_m;
    if (row_start >= M) return;
    
    int num_k_blocks = K / bk;
    
    // Each thread handles one or more K-blocks
    for (int kb = threadIdx.x; kb < num_k_blocks; kb += blockDim.x) {
        uint8_t pattern = 0;
        
        // Check each of the 8 BK-sized columns within this K-block
        for (int bit = 0; bit < bk; bit++) {
            int k = kb * bk + bit;
            bool col_has_nonzero = false;
            
            for (int r = 0; r < block_m && (row_start + r) < M; r++) {
                if (fabsf(A_clustered[(row_start + r) * K + k]) > zero_threshold) {
                    col_has_nonzero = true;
                    break;
                }
            }
            
            if (col_has_nonzero) {
                pattern |= (1 << bit);
            }
        }
        
        patterns[tile_row * num_k_blocks + kb] = pattern;
    }
}

// ============================================================================
// CPU-side Clustering Algorithms
// ============================================================================

namespace clustering_cpu {

/**
 * Simple sort by density (zeros descending = sparse rows first)
 */
inline std::vector<int> sort_by_density(
    const std::vector<int>& row_nnz,
    int M
) {
    std::vector<int> perm(M);
    std::iota(perm.begin(), perm.end(), 0);
    
    // Sort by ascending nnz (most sparse rows first)
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return row_nnz[a] < row_nnz[b];
    });
    
    return perm;
}

/**
 * Sort by block pattern signature
 * Groups rows with identical sparsity patterns together
 */
inline std::vector<int> sort_by_pattern(
    const std::vector<uint64_t>& signatures,
    const std::vector<int>& row_nnz,
    int M
) {
    std::vector<int> perm(M);
    std::iota(perm.begin(), perm.end(), 0);
    
    // Sort by: (1) signature, (2) density within same signature
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (signatures[a] != signatures[b]) {
            // Fewer set bits = sparser pattern = comes first
            return __builtin_popcountll(signatures[a]) < __builtin_popcountll(signatures[b]);
        }
        return row_nnz[a] < row_nnz[b];
    });
    
    return perm;
}

/**
 * Cluster similar patterns into groups of block_m
 * Greedy algorithm: for each group, pick rows with most similar patterns
 */
inline std::vector<int> cluster_similar(
    const std::vector<uint64_t>& signatures,
    const std::vector<int>& row_nnz,
    int M,
    int block_m
) {
    std::vector<int> perm;
    perm.reserve(M);
    
    std::vector<bool> used(M, false);
    
    // For each group of block_m rows
    while (perm.size() < static_cast<size_t>(M)) {
        // Find the most sparse unused row as anchor
        int anchor = -1;
        int min_nnz = INT_MAX;
        for (int i = 0; i < M; i++) {
            if (!used[i] && row_nnz[i] < min_nnz) {
                min_nnz = row_nnz[i];
                anchor = i;
            }
        }
        
        if (anchor < 0) break;
        
        used[anchor] = true;
        perm.push_back(anchor);
        
        // Find block_m - 1 most similar rows
        uint64_t anchor_sig = signatures[anchor];
        
        std::vector<std::pair<int, int>> candidates;  // (hamming_distance, row_idx)
        for (int i = 0; i < M; i++) {
            if (!used[i]) {
                int dist = __builtin_popcountll(signatures[i] ^ anchor_sig);
                candidates.push_back({dist, i});
            }
        }
        
        // Sort by similarity to anchor
        std::sort(candidates.begin(), candidates.end());
        
        // Take top block_m - 1
        int take = std::min(static_cast<int>(candidates.size()), block_m - 1);
        for (int i = 0; i < take; i++) {
            int row = candidates[i].second;
            used[row] = true;
            perm.push_back(row);
        }
    }
    
    return perm;
}

} // namespace clustering_cpu

// ============================================================================
// Main Preprocessor Class
// ============================================================================

class RowClusteringPreprocessor {
public:
    RowClusteringPreprocessor(ClusterConfig config = {}) : config_(config) {}
    
    /**
     * Analyze matrix A and compute optimal row clustering
     * Returns reordered A and permutation info
     */
    ClusteringResult analyze_and_reorder(
        const float* d_A,
        int M, int K,
        bool verbose = false
    ) {
        ClusteringResult result = {};
        result.M = M;
        result.K = K;
        result.num_row_tiles = M / config_.block_m;
        result.num_k_blocks = K / config_.bk;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // ====== Phase 1: Analyze row patterns (GPU) ======
        cudaEventRecord(start);
        
        int* d_row_nnz;
        uint64_t* d_signatures;
        cudaMalloc(&d_row_nnz, M * sizeof(int));
        cudaMalloc(&d_signatures, M * sizeof(uint64_t));
        
        const int THREADS = 256;
        
        compute_row_density_kernel<<<M, THREADS>>>(
            d_A, M, K, d_row_nnz, config_.zero_threshold);
        
        compute_row_block_signature_kernel<<<M, THREADS>>>(
            d_A, M, K, config_.block_k, d_signatures, config_.zero_threshold);
        
        // Measure original block sparsity
        int* d_zero_blocks;
        cudaMalloc(&d_zero_blocks, sizeof(int));
        cudaMemset(d_zero_blocks, 0, sizeof(int));
        
        int num_row_tiles = (M + config_.block_m - 1) / config_.block_m;
        count_zero_blocks_kernel<<<num_row_tiles, THREADS>>>(
            d_A, M, K, config_.block_m, config_.block_k, 
            d_zero_blocks, config_.zero_threshold);
        
        int h_original_zero_blocks;
        cudaMemcpy(&h_original_zero_blocks, d_zero_blocks, sizeof(int), cudaMemcpyDeviceToHost);
        
        int total_blocks = num_row_tiles * (K / config_.block_k);
        result.original_block_sparsity = static_cast<float>(h_original_zero_blocks) / total_blocks;
        
        // Copy analysis to host
        std::vector<int> h_row_nnz(M);
        std::vector<uint64_t> h_signatures(M);
        cudaMemcpy(h_row_nnz.data(), d_row_nnz, M * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_signatures.data(), d_signatures, M * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.analysis_time_ms, start, stop);
        
        // ====== Phase 2: Compute permutation (CPU) ======
        std::vector<int> h_perm;
        
        switch (config_.strategy) {
            case ClusterConfig::Strategy::SORT_BY_DENSITY:
                h_perm = clustering_cpu::sort_by_density(h_row_nnz, M);
                break;
            case ClusterConfig::Strategy::SORT_BY_PATTERN:
                h_perm = clustering_cpu::sort_by_pattern(h_signatures, h_row_nnz, M);
                break;
            case ClusterConfig::Strategy::CLUSTER_SIMILAR:
                h_perm = clustering_cpu::cluster_similar(h_signatures, h_row_nnz, M, config_.block_m);
                break;
        }
        
        // Compute inverse permutation
        std::vector<int> h_inv_perm(M);
        for (int i = 0; i < M; i++) {
            h_inv_perm[h_perm[i]] = i;
        }
        
        // ====== Phase 3: Apply permutation (GPU) ======
        cudaEventRecord(start);
        
        // Allocate device memory
        cudaMalloc(&result.d_row_perm, M * sizeof(int));
        cudaMalloc(&result.d_row_inv_perm, M * sizeof(int));
        cudaMalloc(&result.d_A_clustered, M * K * sizeof(float));
        
        cudaMemcpy(result.d_row_perm, h_perm.data(), M * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(result.d_row_inv_perm, h_inv_perm.data(), M * sizeof(int), cudaMemcpyHostToDevice);
        
        // Gather rows into new order
        gather_rows_kernel<<<M, THREADS>>>(
            d_A, result.d_A_clustered, result.d_row_perm, M, K);
        
        // Measure new block sparsity
        cudaMemset(d_zero_blocks, 0, sizeof(int));
        count_zero_blocks_kernel<<<num_row_tiles, THREADS>>>(
            result.d_A_clustered, M, K, config_.block_m, config_.block_k,
            d_zero_blocks, config_.zero_threshold);
        
        int h_clustered_zero_blocks;
        cudaMemcpy(&h_clustered_zero_blocks, d_zero_blocks, sizeof(int), cudaMemcpyDeviceToHost);
        result.clustered_block_sparsity = static_cast<float>(h_clustered_zero_blocks) / total_blocks;
        
        // ====== Phase 4: Encode patterns for clustered A ======
        int num_patterns = result.num_row_tiles * result.num_k_blocks;
        cudaMalloc(&result.d_a_patterns, num_patterns * sizeof(uint8_t));
        
        encode_clustered_patterns_kernel<<<result.num_row_tiles, THREADS>>>(
            result.d_A_clustered, result.d_a_patterns,
            M, K, config_.block_m, config_.bk, config_.zero_threshold);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.reorder_time_ms, start, stop);
        
        // Compute improvement
        result.improvement_ratio = result.clustered_block_sparsity / 
                                   std::max(result.original_block_sparsity, 0.001f);
        
        if (verbose) {
            printf("Row Clustering Results:\n");
            printf("  Matrix: %d × %d\n", M, K);
            printf("  Block size: %d × %d\n", config_.block_m, config_.block_k);
            printf("  Original block sparsity: %.1f%%\n", result.original_block_sparsity * 100);
            printf("  Clustered block sparsity: %.1f%%\n", result.clustered_block_sparsity * 100);
            printf("  Improvement: %.2fx\n", result.improvement_ratio);
            printf("  Analysis time: %.3f ms\n", result.analysis_time_ms);
            printf("  Reorder time: %.3f ms\n", result.reorder_time_ms);
        }
        
        // Cleanup temporaries
        cudaFree(d_row_nnz);
        cudaFree(d_signatures);
        cudaFree(d_zero_blocks);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return result;
    }
    
    /**
     * Unpermute output matrix C after GEMM
     * C_out[original_row, :] = C_clustered[new_row, :]
     */
    void unpermute_output(
        const float* d_C_clustered,
        float* d_C_out,
        const int* d_row_perm,
        int M, int N
    ) {
        const int THREADS = 256;
        scatter_rows_kernel<<<M, THREADS>>>(
            d_C_clustered, d_C_out, d_row_perm, M, N);
    }
    
    /**
     * Convenience overload using ClusteringResult
     */
    void unpermute_output(
        const float* d_C_clustered,
        float* d_C_out,
        const ClusteringResult& result,
        int N
    ) {
        unpermute_output(d_C_clustered, d_C_out, result.d_row_perm, result.M, N);
    }

private:
    ClusterConfig config_;
};

// ============================================================================
// Integrated Pipeline: Clustered Sparse GEMM
// ============================================================================

/**
 * Full pipeline: cluster + sparse GEMM + unpermute
 * 
 * For repeated GEMMs with same A, cache the ClusteringResult and reuse.
 */
struct ClusteredSparseGEMM {
    RowClusteringPreprocessor clusterer;
    ClusteringResult cluster_result;
    bool has_cached_clustering = false;
    
    ClusteredSparseGEMM(ClusterConfig config = {}) : clusterer(config) {}
    
    /**
     * Prepare clustering for matrix A (call once, reuse for multiple B)
     */
    void prepare(const float* d_A, int M, int K, bool verbose = false) {
        if (has_cached_clustering) {
            cluster_result.free();
        }
        cluster_result = clusterer.analyze_and_reorder(d_A, M, K, verbose);
        has_cached_clustering = true;
    }
    
    /**
     * Check if clustering provides sufficient benefit
     */
    bool clustering_worthwhile(float min_improvement = 1.3f) const {
        return has_cached_clustering && 
               cluster_result.improvement_ratio >= min_improvement;
    }
    
    /**
     * Get the clustered A matrix (for use with your K25 kernel)
     */
    const float* get_clustered_A() const {
        return cluster_result.d_A_clustered;
    }
    
    /**
     * Get the A patterns (for use with your K25 kernel)
     */
    const uint8_t* get_a_patterns() const {
        return cluster_result.d_a_patterns;
    }
    
    /**
     * Unpermute the output
     */
    void unpermute(const float* d_C_clustered, float* d_C, int N) {
        clusterer.unpermute_output(d_C_clustered, d_C, cluster_result, N);
    }
    
    ~ClusteredSparseGEMM() {
        if (has_cached_clustering) {
            cluster_result.free();
        }
    }
};

} // namespace row_clustering

