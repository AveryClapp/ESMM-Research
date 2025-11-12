/*
 * Tunable A-matrix Preprocessor for kernel_tuner
 *
 * Generates 8-bit sparsity patterns for each row/K-block combination
 * This version exposes tunable parameters for optimization
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Tunable parameters (will be replaced by kernel_tuner)
#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#ifndef ROWS_PER_THREAD
#define ROWS_PER_THREAD 1
#endif

#ifndef USE_SHARED_MEM
#define USE_SHARED_MEM 0
#endif

#ifndef VECTORIZE
#define VECTORIZE 1  // 1=float, 2=float2, 4=float4
#endif

#ifndef K_BATCH_SIZE
#define K_BATCH_SIZE 4  // Process this many K-blocks per loop iteration
#endif

const int BK = 8;  // Fixed: 8-element K-blocks

/*
 * Tunable Preprocessor Kernel
 *
 * Each thread processes ROWS_PER_THREAD rows
 * Generates patterns for all K-blocks in those rows
 */
extern "C" __global__ void preprocess_a_patterns(
    int M, int K,
    const float* A,
    uint8_t* rowPatterns
) {
    const int numKBlocks = K / BK;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_start = tid * ROWS_PER_THREAD;

    // Bounds check
    if (row_start >= M) return;

    // Process ROWS_PER_THREAD rows
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        const int row = row_start + r;
        if (row >= M) break;

        const float* row_ptr = &A[row * K];

        // Process K-blocks in batches
        for (int kBlockBatch = 0; kBlockBatch < numKBlocks; kBlockBatch += K_BATCH_SIZE) {
            #pragma unroll
            for (int kb = 0; kb < K_BATCH_SIZE && (kBlockBatch + kb) < numKBlocks; kb++) {
                const int kBlock = kBlockBatch + kb;
                uint8_t pattern = 0;

                // Check BK elements for non-zeros
                #if VECTORIZE == 4
                // Vectorized float4 loads (BK=8, so 2 float4 loads)
                #pragma unroll
                for (int k = 0; k < BK; k += 4) {
                    const int col = kBlock * BK + k;
                    float4 vals = *((const float4*)(&row_ptr[col]));
                    if (vals.x != 0.0f) pattern |= (1 << (k+0));
                    if (vals.y != 0.0f) pattern |= (1 << (k+1));
                    if (vals.z != 0.0f) pattern |= (1 << (k+2));
                    if (vals.w != 0.0f) pattern |= (1 << (k+3));
                }
                #elif VECTORIZE == 2
                // Vectorized float2 loads (BK=8, so 4 float2 loads)
                #pragma unroll
                for (int k = 0; k < BK; k += 2) {
                    const int col = kBlock * BK + k;
                    float2 vals = *((const float2*)(&row_ptr[col]));
                    if (vals.x != 0.0f) pattern |= (1 << (k+0));
                    if (vals.y != 0.0f) pattern |= (1 << (k+1));
                }
                #else
                // Scalar loads
                #pragma unroll
                for (int k = 0; k < BK; k++) {
                    const int col = kBlock * BK + k;
                    if (row_ptr[col] != 0.0f) {
                        pattern |= (1 << k);
                    }
                }
                #endif

                rowPatterns[row * numKBlocks + kBlock] = pattern;
            }
        }
    }
}
