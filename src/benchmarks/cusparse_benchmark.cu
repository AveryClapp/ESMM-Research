// src/benchmarks/cusparse_benchmark.cu
// Standalone cuSPARSE SpMM benchmark for comparison against K25.
// Outputs one CSV row: size,density,spmm_us,total_us,spmm_gflops,total_gflops

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(x) do { \
    cusparseStatus_t err = (x); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

// Fill host matrix with blockwise sparsity: 64-row x 8-col tiles
// Matching the K25 A-matrix convention (randomize_matrix_A_blocklevel).
void fill_blockwise_sparse(float* A, int M, int K, float density) {
    const int TILE_M = 64, TILE_K = 8;
    int mblocks = (M + TILE_M - 1) / TILE_M;
    int kblocks = (K + TILE_K - 1) / TILE_K;
    for (int bm = 0; bm < mblocks; bm++) {
        for (int bk = 0; bk < kblocks; bk++) {
            float r = (float)rand() / (float)RAND_MAX;
            int active = (r < density) ? 1 : 0;
            int m_start = bm * TILE_M, m_end = m_start + TILE_M < M ? m_start + TILE_M : M;
            int k_start = bk * TILE_K, k_end = k_start + TILE_K < K ? k_start + TILE_K : K;
            for (int m = m_start; m < m_end; m++) {
                for (int k = k_start; k < k_end; k++) {
                    A[m * K + k] = active ? ((float)rand() / (float)RAND_MAX - 0.5f) : 0.0f;
                }
            }
        }
    }
}

void fill_random(float* X, long long n) {
    for (long long i = 0; i < n; i++) X[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

int main(int argc, char* argv[]) {
    int   N           = 4096;
    float density     = 0.5f;
    int   runs        = 10;
    int   print_header = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size")   == 0 && i + 1 < argc) N       = atoi(argv[++i]);
        if (strcmp(argv[i], "--density")== 0 && i + 1 < argc) density = (float)atof(argv[++i]);
        if (strcmp(argv[i], "--runs")   == 0 && i + 1 < argc) runs    = atoi(argv[++i]);
        if (strcmp(argv[i], "--header") == 0)                  print_header = 1;
    }

    if (print_header) {
        printf("size,density,spmm_us,total_us,spmm_gflops,total_gflops\n");
        return 0;
    }

    int M = N, K = N;
    long long flops = 2LL * N * N * N;

    srand(42);

    // ---- Host allocation and fill ----
    long long A_elems = (long long)M * K;
    long long B_elems = (long long)K * N;
    long long C_elems = (long long)M * N;

    float* hA = (float*)malloc(A_elems * sizeof(float));
    float* hB = (float*)malloc(B_elems * sizeof(float));
    if (!hA || !hB) { fprintf(stderr, "Host malloc failed\n"); exit(1); }

    fill_blockwise_sparse(hA, M, K, density);
    fill_random(hB, B_elems);

    // ---- Device allocation ----
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, A_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, B_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, C_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, A_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, C_elems * sizeof(float)));

    // ---- cuSPARSE setup ----
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Dense descriptor for A (for conversion)
    cusparseDnMatDescr_t dnA;
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnA, M, K, K, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Sparse CSR descriptor - start with nnz=0, fill after analysis
    cusparseSpMatDescr_t spA;
    int64_t* d_csr_offsets;
    int64_t* d_csr_cols;
    float*   d_csr_vals;
    CHECK_CUDA(cudaMalloc(&d_csr_offsets, (M + 1) * sizeof(int64_t)));

    CHECK_CUSPARSE(cusparseCreateCsr(&spA, M, K, 0,
                                     d_csr_offsets, NULL, NULL,
                                     CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Analysis pass: determine nnz and fill row offsets
    size_t conv_buf_size = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, dnA, spA,
                                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                    &conv_buf_size));
    void* conv_buf;
    CHECK_CUDA(cudaMalloc(&conv_buf, conv_buf_size > 0 ? conv_buf_size : 1));
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, dnA, spA,
                                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                   conv_buf));

    // Get nnz from the sparse descriptor
    int64_t rows64, cols64, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(spA, &rows64, &cols64, &nnz));

    // Allocate CSR column-index and values arrays now that we know nnz
    CHECK_CUDA(cudaMalloc(&d_csr_cols, nnz * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_csr_vals, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(spA, d_csr_offsets, d_csr_cols, d_csr_vals));

    // Initial conversion (outside timing bands) for SpMM-only baseline
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                  conv_buf));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Dense descriptors for B (K×N) and C (M×N)
    cusparseDnMatDescr_t dnB, dnC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnB, K, N, N, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnC, M, N, N, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // SpMM workspace buffer
    size_t spmm_buf_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, spA, dnB, &beta, dnC,
                                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                            &spmm_buf_size));
    void* spmm_buf;
    CHECK_CUDA(cudaMalloc(&spmm_buf, spmm_buf_size > 0 ? spmm_buf_size : 1));

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // ---- Band 1: SpMM only (A already in CSR, conversion not timed) ----
    for (int i = 0; i < 3; i++) {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    float spmm_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&spmm_ms, t0, t1));
    double spmm_us = (double)(spmm_ms / runs) * 1000.0;

    // ---- Band 2: total = dense->CSR conversion + SpMM ----
    for (int i = 0; i < 2; i++) {
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                      CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                      conv_buf));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) {
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                      CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                      conv_buf));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, t0, t1));
    double total_us = (double)(total_ms / runs) * 1000.0;

    double spmm_gflops  = (double)flops / (spmm_us  * 1e-6) / 1e9;
    double total_gflops = (double)flops / (total_us * 1e-6) / 1e9;

    printf("%d,%.4f,%.3f,%.3f,%.2f,%.2f\n",
           N, density, spmm_us, total_us, spmm_gflops, total_gflops);

    // Cleanup
    cusparseDestroyDnMat(dnA);
    cusparseDestroyDnMat(dnB);
    cusparseDestroyDnMat(dnC);
    cusparseDestroySpMat(spA);
    cusparseDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(d_csr_offsets); cudaFree(d_csr_cols); cudaFree(d_csr_vals);
    cudaFree(conv_buf); cudaFree(spmm_buf);
    free(hA); free(hB);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
