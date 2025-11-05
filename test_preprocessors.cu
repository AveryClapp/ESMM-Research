#include "include/utils.cuh"
#include "include/metadata.cuh"
#include "src/preprocessors/a_preprocessor_hybrid.cu"
#include "src/preprocessors/b_preprocessor_hybrid.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const int M = 4096;
    const int K = 4096;
    const int N = 4096;
    const int WM = 32;
    const int BK = 8;
    const int TN = 8;

    printf("Matrix dimensions: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Parameters: WM=%d, BK=%d, TN=%d\n\n", WM, BK, TN);

    // Allocate matrices
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));

    // Initialize with 50% sparsity
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (rand() % 100 < 50) ? 0.0f : (float)(rand() % 10 + 1);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (rand() % 100 < 50) ? 0.0f : (float)(rand() % 10 + 1);
    }

    // Copy to device
    float *d_A, *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Test A preprocessor
    printf("Testing A Preprocessor:\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, M, K, WM, BK);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float a_time = 0;
    cudaEventElapsedTime(&a_time, start, stop);
    printf("A-matrix GPU preprocessing: %d blocks (%.1f KB metadata) in %.3f ms\n\n",
           A_meta.numWarpRows * A_meta.numKBlocks,
           (A_meta.numWarpRows * A_meta.numKBlocks) / 1024.0f,
           a_time);

    // Test B preprocessor
    printf("Testing B Preprocessor:\n");
    BMatrixPatternMetadata B_meta = analyze_b_sparsity_pattern_gpu(d_B, K, N, BK, TN);
    printf("\n");

    // Compare
    printf("=== Performance Comparison ===\n");
    printf("A preprocessor: %.3f ms\n", a_time);
    printf("B preprocessor: Check output above\n");

    // Cleanup
    free_block_pattern_metadata(A_meta);
    free_b_pattern_metadata(B_meta);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
