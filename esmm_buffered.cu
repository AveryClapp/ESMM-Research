#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.cuh"

/* Experimental improvemnt to esmm.cu, where we leverage double buffering for a more efficient and pipelined kernel */
/**
 * The motivation behind this kernel is that in our previous version, all threads are either loading from GMEM or computing.
 * This doesn't seem like such a bad thing, except for the fact that memory units are idle during the compute phase and the 
 * compute units are idle during the load phase. So, we can simulate a sort of ping-pong effect where half the threads either
 * load a block or compute on that block and the other half are doing the inverse. Of course, we need to be careful to ensure
 * that all blocks attempting to be computed on are already loaded. 
 */



/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */

namespace db {
  template <const int BM, const int BN, const int BK, const int rowStrideA,
    const int rowStrideB>
  __device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                               float *As, float *Bs, const int innerRowA,
                               const int innerColA, const int innerRowB,
                               const int innerColB) {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(
          &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
  }

  template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
    const int TM, const int TN>
  __device__ void processFromSmem(float *regM, float *regN, float *threadResults, 
                                  const float *As, const float *Bs, const uint warpRow, 
                                  const uint warpCol, const uint threadRowInWarp, 
                                  const uint threadColInWarp) {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
          regM[wSubRowIdx + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
            threadRowInWarp + i];
        }
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
          regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
            threadColInWarp * TN + i];
        }
      }

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          //calculate per-thread results
          //multiply_dense(wSubRowIdx, wSubColIdx, WNITER, regM[wSubRowIdx], regN, threadResults);
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  regM[wSubRowIdx * TM + resIdxM] *
                  regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }
  }
} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
esmm_buffered(const int M, const int N, const int K, float *A, float *B, float *C) {
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

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // Divide threads into two groups: loaders and computers
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  /**
   * Calculate the indices that this thread will load into SMEM.
   * This is half of what we used for other kernels since we are dividing into
   * loaders and computers
   */
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // Load block 0
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // Process block 0
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
        TN>(regM, regN, threadResults, As, Bs, warpRow,
            warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // Process block 1 (loaded by other side)
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
          TM, TN>(regM, regN, threadResults, As + (BM * BK),
                  Bs + (BK * BN), warpRow, warpCol,
                  threadRowInWarp, threadColInWarp);
      }
      __syncthreads();

      // Load block 2 into the first half of As & Bs (this will be block 0 next iteration)
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
          N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
          innerRowB, innerColB);
      }
    } else {
      // Load block 1 into the second half of As & Bs
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
          N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
          innerColA, innerRowB, innerColB);
      }
      __syncthreads();

      // Process the rest of block 0
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
        TN>(regM, regN, threadResults, As, Bs, warpRow,
            warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // Process the rest of block 1
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
          TM, TN>(regM, regN, threadResults, As + (BM * BK),
                  Bs + (BK * BN), warpRow, warpCol,
                  threadRowInWarp, threadColInWarp);
      }
    }

    A += 2 * BK;
    B += 2 * BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp;
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
            wSubColIdx * TN + resIdxN;
          tmp.x = threadResults[i + 0];
          tmp.y = threadResults[i + 1];
          tmp.z = threadResults[i + 2];
          tmp.w = threadResults[i + 3];
          // write back
          reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
            threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}
