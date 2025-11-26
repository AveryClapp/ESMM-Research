#pragma once

/*
 * ============================================================================
 * Kernel: B-Sparse GEMM with Warp-Uniform Pattern Checking
 * ============================================================================
 *
 * KEY INSIGHT: All 32 threads in a warp access the SAME K-row of B.
 * Pattern check is warp-uniform → zero divergence.
 *
 * WHY THIS WINS:
 *   1. NO TRANSPOSE (vs K19, K20)
 *   2. NO COMPLEX DISPATCH (vs K21, K22) - just one AND + branch
 *   3. ZERO WARP DIVERGENCE
 */

#include <cuda_runtime.h>
#include "../preprocessors/b_preprocessor_wn_uniform.cu"

#ifndef WARPSIZE
#define WARPSIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

template <const int BM, const int BN, const int BK, 
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_b_sparse(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const uint8_t* __restrict__ b_patterns,
    int numKBlocks
) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);
    
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;
    
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    
    __shared__ float As[BK * (BM + 1)];
    __shared__ float Bs[BK * BN];
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;
    
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);
    
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];
    
    // WARP-UNIFORM: globalWarpCol is the SAME for all 32 threads in a warp
    const uint globalWarpCol = cCol * (BN / WN) + warpCol;
    
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const int kBlock = bkIdx / BK;
        
        // All 32 threads read the SAME pattern byte - ZERO DIVERGENCE
        const uint8_t b_pattern = b_patterns[globalWarpCol * numKBlocks + kBlock];
        
        // Load A tile (column-major for coalescing)
        #pragma unroll
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * (BM + 1) + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + 1) + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + 1) + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + 1) + innerRowA + offset] = tmp.w;
        }
        
        // Load B tile (row-major)
        #pragma unroll
        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(
                &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(
                    &B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        
        __syncthreads();
        
        // INNER LOOP WITH B-PATTERN SKIPPING
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            
            // ALL 32 threads evaluate the SAME condition
            if (!(b_pattern & (1 << dotIdx))) {
                continue;
            }
            
            // Load A values
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                regM[wSubRowIdx * TM + 0] = As[(dotIdx * (BM + 1)) + warpRow * WM +
                    wSubRowIdx * WSUBM + threadRowInWarp * TM + 0];
            }
            
            // Load B values
            #pragma unroll
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                const uint bBase = (dotIdx * BN) + warpCol * WN + 
                    wSubColIdx * WSUBN + threadColInWarp * TN;
                
                regN[wSubColIdx * TN + 0] = Bs[bBase + 0];
                regN[wSubColIdx * TN + 1] = Bs[bBase + 1];
                regN[wSubColIdx * TN + 2] = Bs[bBase + 2];
                regN[wSubColIdx * TN + 3] = Bs[bBase + 3];
                regN[wSubColIdx * TN + 4] = Bs[bBase + 4];
                regN[wSubColIdx * TN + 5] = Bs[bBase + 5];
                regN[wSubColIdx * TN + 6] = Bs[bBase + 6];
                regN[wSubColIdx * TN + 7] = Bs[bBase + 7];
            }
            
            // Outer product
            #pragma unroll
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                #pragma unroll
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    const float aVal = regM[wSubRowIdx * TM + 0];
                    const int resBase = (wSubRowIdx * WNITER + wSubColIdx) * TN;
                    
                    threadResults[resBase + 0] += aVal * regN[wSubColIdx * TN + 0];
                    threadResults[resBase + 1] += aVal * regN[wSubColIdx * TN + 1];
                    threadResults[resBase + 2] += aVal * regN[wSubColIdx * TN + 2];
                    threadResults[resBase + 3] += aVal * regN[wSubColIdx * TN + 3];
                    threadResults[resBase + 4] += aVal * regN[wSubColIdx * TN + 4];
                    threadResults[resBase + 5] += aVal * regN[wSubColIdx * TN + 5];
                    threadResults[resBase + 6] += aVal * regN[wSubColIdx * TN + 6];
                    threadResults[resBase + 7] += aVal * regN[wSubColIdx * TN + 7];
                }
            }
        }
        
        A += BK;
        B += BK * N;
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        #pragma unroll
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            
            #pragma unroll
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                const int resBase = (wSubRowIdx * WNITER + wSubColIdx) * TN;
                
                float4 tmp;
                tmp.x = threadResults[resBase + 0];
                tmp.y = threadResults[resBase + 1];
                tmp.z = threadResults[resBase + 2];
                tmp.w = threadResults[resBase + 3];
                reinterpret_cast<float4*>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * N + 
                    threadColInWarp * TN + 0])[0] = tmp;
                
                tmp.x = threadResults[resBase + 4];
                tmp.y = threadResults[resBase + 5];
                tmp.z = threadResults[resBase + 6];
                tmp.w = threadResults[resBase + 7];
                reinterpret_cast<float4*>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * N + 
                    threadColInWarp * TN + 4])[0] = tmp;
            }
        }
    }
}

// ============================================================================
// Runner Function
// ============================================================================

// CRITICAL CONFIG: NUM_THREADS/32 = (BM/WM) × (BN/WN)
// With 128 threads (4 warps): 4 = (64/64) × (128/32) = 1 × 4 ✓

template <int BM = 64, int BN = 128, int BK = 8, 
          int WM = 64, int WN = 32, int WNITER = 2,
          int TM = 1, int TN = 8, int NUM_THREADS = 128>
float run_b_sparse_gemm(
    int M, int N, int K,
    float* d_A, float* d_B, float* d_C,
    int warmup_runs = 3,
    int timed_runs = 10
) {
    BPatternMetadata meta = preprocess_b_patterns<BK, WN>(d_B, K, N);
    
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    for (int i = 0; i < warmup_runs; i++) {
        cudaMemset(d_C, 0, M * N * sizeof(float));
        esmm_b_sparse<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<grid, block>>>(M, N, K, d_A, d_B, d_C, meta.d_patterns, meta.numKBlocks);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < timed_runs; i++) {
        esmm_b_sparse<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<grid, block>>>(M, N, K, d_A, d_B, d_C, meta.d_patterns, meta.numKBlocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / timed_runs;
    
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    
    printf("\nB-Sparse GEMM Results:\n");
    printf("  Matrix: %d x %d x %d\n", M, N, K);
    printf("  Time: %.3f ms (avg of %d runs)\n", avg_ms, timed_runs);
    printf("  Performance: %.1f GFLOPS\n", gflops);
    printf("  Effective K-skip: %.1f%%\n", meta.sparsityPercent);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_b_pattern_metadata(meta);
    
    return avg_ms;
}

