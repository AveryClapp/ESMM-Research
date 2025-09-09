#pragma once
#include <cuda_runtime.h>

/* C-Compatible version of double buffered ESMM kernel using Kernel Tuner defines */

#define WARPSIZE 32

// Calculate derived constants from Kernel Tuner defines
#define WMITER ((WM * WN) / (WARPSIZE * TM * TN * WNITER))
#define WSUBM (WM / WMITER)
#define WSUBN (WN / WNITER)
#define rowStrideA (((NUM_THREADS / 2) * 4) / BK)
#define rowStrideB ((NUM_THREADS / 2) / (BN / 4))

__forceinline__ __device__ void multiply_dense(int wSubRowIdx, int wSubColIdx,
                                int WNITER_val, float regM_val, float* regN,
                                        float* threadResults) {
    const int regNBase = wSubColIdx * TN;
    const int threadResBase = wSubRowIdx * (WNITER_val * TN) + (wSubColIdx * TN);
    
    for (int i = 0; i < TN; ++i) {
        threadResults[threadResBase + i] += regM_val * regN[regNBase + i];
    }
}

__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
    
    for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        // transpose A while storing it
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

__device__ void processFromSmem(float *regM, float *regN, float *threadResults, 
                                const float *As, const float *Bs, const int warpRow, 
                                const int warpCol, const int threadRowInWarp, 
                                const int threadColInWarp) {
    
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // Load regM for all WMITER sub-warps
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            regM[wSubRowIdx] =
                As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                threadRowInWarp];
        }
        
        // Load regN for all WNITER sub-warps
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (int i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                    threadColInWarp * TN + i];
            }
        }

        // Calculate per-thread results
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                multiply_dense(wSubRowIdx, wSubColIdx, WNITER, regM[wSubRowIdx], regN, threadResults);
            }
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
esmm_buffered(const int M, const int N, const int K, float *A, float *B, float *C) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // Allocate space for the current blocktile in SMEM
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    // Divide threads into two groups: loaders and computers
    bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    /**
     * Calculate the indices that this thread will load into SMEM.
     * This is half of what we used for other kernels since we are dividing into two groups
     */
    const int innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
    const int innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
    const int innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
    const int innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    if (doubleBufferIdx == 0) {
        // Load block 0
        loadFromGmem(N, K, A, B, As[0], Bs[0], innerRowA, innerColA, innerRowB, innerColB);
    }

    __syncthreads();

    // outer-most loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
        if (doubleBufferIdx == 0) {
            // Process block 0
            processFromSmem(regM, regN, threadResults, As[0], Bs[0], warpRow,
                            warpCol, threadRowInWarp, threadColInWarp);

            // Process block 1 (loaded by other side)
            if (bkIdx + BK < K) {
                processFromSmem(regM, regN, threadResults, As[1], Bs[1], warpRow, warpCol,
                                threadRowInWarp, threadColInWarp);
            }

            // Load block 2 into the first half of As & Bs (this will be block 0 next iteration)
            if (bkIdx + 2 * BK < K) {
                loadFromGmem(N, K, A + 2 * BK, B + 2 * BK * N, As[0], Bs[0], innerRowA, innerColA,
                            innerRowB, innerColB);
            }
        } else {
            // Load block 1 into the second half of As & Bs
            if (bkIdx + BK < K) {
                loadFromGmem(N, K, A + BK, B + BK * N, As[1], Bs[1], innerRowA,
                            innerColA, innerRowB, innerColB);
            }

            // Process the rest of block 0
            processFromSmem(regM, regN, threadResults, As[0], Bs[0], warpRow,
                            warpCol, threadRowInWarp, threadColInWarp);

            // Process block 1
            if (bkIdx + BK < K) {
                processFromSmem(regM, regN, threadResults, As[1], Bs[1], warpRow, warpCol,
                                threadRowInWarp, threadColInWarp);
            }

            // Load block 3 into the second half of As & Bs (this will be block 1 next iteration)
            if (bkIdx + 3 * BK < K) {
                loadFromGmem(N, K, A + 3 * BK, B + 3 * BK * N, As[1], Bs[1], innerRowA,
                            innerColA, innerRowB, innerColB);
            }
        }

        A += 2 * BK;
        B += 2 * BK * N;
        __syncthreads();
    }

    // Write out the results
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *threadResult = &threadResults[(wSubRowIdx * WNITER + wSubColIdx) * TM * TN];
            
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int globalRow = cRow * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp + resIdxM;
                    int globalCol = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                    
                    if (globalRow < M && globalCol < N) {
                        C[globalRow * N + globalCol] = threadResult[resIdxM * TN + resIdxN];
                    }
                }
            }
        }
    }
}
