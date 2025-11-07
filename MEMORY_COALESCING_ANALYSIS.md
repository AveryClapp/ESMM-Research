# Memory Coalescing Analysis: B-Transpose Architecture

## Current Architecture (Kernel 17) - Baseline

### Warp Tile: 32 rows × 64 columns (WM=32, WN=64, TM=1, TN=8)

**A Matrix Loading (Global → Shared)**
```cuda
// From driver.cu:128-130, esmm_hybrid.cu:154-161
innerRowA = threadIdx.x / (BK/4) = threadIdx.x / 2
innerColA = threadIdx.x % (BK/4) = threadIdx.x % 2
rowStrideA = (NUM_THREADS * 4) / BK = (256 * 4) / 8 = 128

// Thread mapping (first 256 threads in block, BM=128, BK=8):
Thread 0:   innerRowA=0,   innerColA=0  → loads A[0, 0:3]   (4 floats)
Thread 1:   innerRowA=0,   innerColA=1  → loads A[0, 4:7]   (4 floats)
Thread 2:   innerRowA=1,   innerColA=0  → loads A[1, 0:3]   (4 floats)
Thread 3:   innerRowA=1,   innerColA=1  → loads A[1, 4:7]   (4 floats)
...
Thread 254: innerRowA=127, innerColA=0  → loads A[127, 0:3] (4 floats)
Thread 255: innerRowA=127, innerColA=1  → loads A[127, 4:7] (4 floats)

// Global memory layout (row-major): A[row][col]
// Address: A + row * K + col
Thread 0:   A + 0*K + 0   (offset 0)
Thread 1:   A + 0*K + 4   (offset 4*sizeof(float) = 16 bytes)
Thread 2:   A + 1*K + 0   (offset K*sizeof(float))
...

// Coalescing: POOR - consecutive threads access different rows (stride K)
// BUT uses float4 vectorized loads → mitigates partially
```

**B Matrix Loading (Global → Shared)**
```cuda
// esmm_hybrid.cu:163-168
innerRowB = threadIdx.x / (BN/4) = threadIdx.x / 32
innerColB = threadIdx.x % (BN/4) = threadIdx.x % 32
rowStrideB = NUM_THREADS / (BN/4) = 256 / 32 = 8

// Thread mapping (BN=128, BK=8):
Thread 0:  innerRowB=0, innerColB=0  → loads B[0, 0:3]   (4 floats)
Thread 1:  innerRowB=0, innerColB=1  → loads B[0, 4:7]   (4 floats)
...
Thread 31: innerRowB=0, innerColB=31 → loads B[0, 124:127] (4 floats)
Thread 32: innerRowB=1, innerColB=0  → loads B[1, 0:3]   (4 floats)
...

// Global memory layout (row-major): B[row][col]
// Address: B + row * N + col
Thread 0:  B + 0*N + 0     (offset 0)
Thread 1:  B + 0*N + 4     (offset 16 bytes)
Thread 2:  B + 0*N + 8     (offset 32 bytes)
...
Thread 31: B + 0*N + 124   (offset 496 bytes)

// Coalescing: EXCELLENT - consecutive threads access consecutive columns in same row
// Within each group of 32 threads (same row), fully coalesced float4 loads
```

**Compute (Shared Memory Access)**
```cuda
// esmm_hybrid.cu:58-59, 65-72
// Each thread loads from shared memory:
As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM]
Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + offset]

// As layout: column-major [K][M] → As[k*BM + m]
// Bs layout: row-major [K][N] → Bs[k*BN + n]

// As access:
// threadRowInWarp varies 0..15 across warp → accessing different rows
// Bank conflicts possible but mitigated by broadcast

// Bs access:
// Each thread accesses TN=8 consecutive elements
// Different threads access different columns → potential bank conflicts
```

**C Matrix Writing (Shared → Global)**
```cuda
// esmm_hybrid.cu:220-239
// Each thread writes TM * TN = 1 * 8 = 8 outputs
// Uses float4 vectorized writes (2 writes per thread for 8 elements)

C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN]

// threadRowInWarp varies 0..15, threadColInWarp varies 0..1
// Thread 0:  writes C[warpRow*32 + 0, warpCol*64 + 0:7]
// Thread 1:  writes C[warpRow*32 + 0, warpCol*64 + 8:15]
// Thread 2:  writes C[warpRow*32 + 1, warpCol*64 + 0:7]
...

// Coalescing: POOR - threads 0,2,4,... write to different rows
// BUT float4 writes help
```

---

## Proposed B-Transpose Architecture

### Warp Tile: 64 rows × 32 columns (WM=64, WN=32, TM=8, TN=1)

**Key Parameters:**
```cpp
WM = 64, WN = 32, BK = 8
TM = 8, TN = 1
WNITER = 1, WMITER = 4
WSUBM = WM / WMITER = 16
WSUBN = WN / WNITER = 32

threadColInWarp = threadIdxInWarp % (WSUBN / TN) = threadIdxInWarp % 32 = threadIdxInWarp
threadRowInWarp = threadIdxInWarp / (WSUBN / TN) = threadIdxInWarp / 32 = 0 (for all threads!)

// Wait, this doesn't work! All threads would have threadRowInWarp=0
// Need to rethink the mapping...
```

**Correction: Proper Thread Mapping**
```cpp
// For a 64×32 output tile with 32 threads:
// Each thread should handle 64 outputs (64*32/32 = 64)
// With TM=8, TN=1: each thread computes 8*1=8 outputs in inner loop
// Need WMITER=8 or WNITER=8 to get 64 outputs per thread

// Option A: WMITER=8, WNITER=1
WMITER = 8, WNITER = 1
WSUBM = WM / WMITER = 64 / 8 = 8
WSUBN = WN / WNITER = 32 / 1 = 32

// Thread mapping: all 32 threads spread across 32 columns
threadColInWarp = threadIdxInWarp % (WSUBN / TN) = threadIdxInWarp % 32
threadRowInWarp = threadIdxInWarp / (WSUBN / TN) = threadIdxInWarp / 32 = 0

// Then iterate WMITER=8 times over row sub-blocks
for (wSubRowIdx = 0; wSubRowIdx < 8; wSubRowIdx++) {
    // Thread processes row: wSubRowIdx * WSUBM + threadRowInWarp * TM
    //                     = wSubRowIdx * 8 + 0 * 8
    //                     = wSubRowIdx * 8 to (wSubRowIdx * 8 + 7)
    // Column: threadColInWarp * TN = threadColInWarp * 1 = threadColInWarp
}

// This gives us the desired mapping:
Thread 0:  8 rows × 1 col (rows 0-7 in iteration, column 0)
Thread 1:  8 rows × 1 col (rows 0-7 in iteration, column 1)
...
Thread 31: 8 rows × 1 col (rows 0-7 in iteration, column 31)
```

---

### Memory Access Analysis

**A Matrix Loading (Global → Shared)**
```cuda
// Block tile: BM=64 × BK=8
// Thread mapping (keeping similar to original):
innerRowA = threadIdx.x / (BK/4) = threadIdx.x / 2
innerColA = threadIdx.x % (BK/4) = threadIdx.x % 2
rowStrideA = (NUM_THREADS * 4) / BK = (256 * 4) / 8 = 128

// Thread mapping (same as before):
Thread 0:   loads A[0, 0:3]   (4 floats)
Thread 1:   loads A[0, 4:7]   (4 floats)
Thread 2:   loads A[1, 0:3]   (4 floats)
...

// Coalescing: SAME AS BEFORE - POOR but mitigated by float4
// This doesn't change with B transpose!
```

**B^T Matrix Loading (Global → Shared)**
```cuda
// B^T dimensions: N×K (transposed from K×N)
// Block tile: BN=32 × BK=8

// Thread mapping for loading B^T:
innerRowBT = threadIdx.x / (BK/4) = threadIdx.x / 2
innerColBT = threadIdx.x % (BK/4) = threadIdx.x % 2
rowStrideBT = (NUM_THREADS * 4) / BK = 128

// Thread mapping:
Thread 0:   loads B^T[0, 0:3]   (4 floats)
Thread 1:   loads B^T[0, 4:7]   (4 floats)
Thread 2:   loads B^T[1, 0:3]   (4 floats)
...
Thread 63:  loads B^T[31, 4:7]  (4 floats)

// Global memory layout (B^T is N×K, row-major)
// Address: B^T + row * K + col
Thread 0:  B^T + 0*K + 0   (offset 0)
Thread 1:  B^T + 0*K + 4   (offset 16 bytes)
Thread 2:  B^T + 1*K + 0   (offset K*sizeof(float))
...

// Coalescing: POOR - consecutive threads access different rows (stride K)
// Same issue as A loading, mitigated by float4
```

**Compute (Shared Memory Access) - The Critical Part**
```cuda
// B^T access pattern:
// All threads in warp need SAME row of B^T for their column
// Each warp computes 64 rows × 32 columns

for (int col = 0; col < 32; col++) {
    // For column 'col', ALL threads need B^T[globalCol + col, k]
    uint8_t B_pattern = B_patterns[kBlock * numNBlocks + (globalCol + col) / 8];

    // All threads read same pattern → warp-uniform! ✅

    for (int sparse_idx = 0; sparse_idx < count; sparse_idx++) {
        uint8_t k = offsets[sparse_idx];

        // All threads load SAME B^T element:
        float b_val = Bs_transposed[col * BK + k];  // BROADCAST ✅

        // Each thread loads its own A elements (different rows):
        for (int rowOffset = 0; rowOffset < TM; rowOffset++) {
            int row = threadRowInWarp + wSubRowIdx * WSUBM + rowOffset;
            float a_val = As[k * BM + row];  // Different per thread

            threadResults[...] += a_val * b_val;
        }
    }
}

// Shared memory banking:
// - B^T access: all threads read same address → BROADCAST (single bank, but broadcast is efficient)
// - A access: different rows → different banks (likely good distribution)
```

**C Matrix Writing (Shared → Global)**
```cuda
// Each thread writes TM * TN * WMITER = 8 * 1 * 8 = 64 outputs
// Organized as 8 iterations (WMITER) × 8 rows (TM) × 1 column (TN)

for (wSubRowIdx = 0; wSubRowIdx < 8; wSubRowIdx++) {
    for (resIdxM = 0; resIdxM < 8; resIdxM++) {
        int row = warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
        int col = warpCol * WN + threadColInWarp * TN;

        C[row * N + col] = threadResults[...];
    }
}

// Thread 0:  writes C[row0, col0], C[row1, col0], ..., C[row63, col0]
// Thread 1:  writes C[row0, col1], C[row1, col1], ..., C[row63, col1]
// ...
// Thread 31: writes C[row0, col31], C[row1, col31], ..., C[row63, col31]

// Coalescing analysis:
// Consecutive threads write to consecutive columns in SAME row → EXCELLENT! ✅
// Much better than current architecture where consecutive threads write different rows
```

---

## Summary Comparison

| Aspect | Current (32×64) | B-Transpose (64×32) |
|--------|-----------------|---------------------|
| **A Loading** | Poor (stride K) | Poor (stride K) - SAME |
| **B Loading** | Excellent (consecutive) | Poor (stride K) - WORSE |
| **B^T Sparsity Check** | Divergent ❌ | Warp-uniform ✅ |
| **Shared Mem B Access** | Different addresses | Broadcast ✅ |
| **Shared Mem A Access** | Similar | Similar |
| **C Writing** | Poor (stride N) | Excellent (consecutive) ✅ |

**Net Assessment:**
- **LOSS**: B^T loading is worse (but can be mitigated with preprocessing/caching)
- **WIN**: B-sparsity checks are warp-uniform (eliminates divergence!)
- **WIN**: C writing is better coalesced
- **WIN**: Shared memory B access uses broadcast (efficient)

**Overall: PROMISING** ✅
The key win (warp-uniform B-sparsity) likely outweighs the B^T loading downside, especially if:
1. B is reused multiple times (transpose cost amortized)
2. B has significant sparsity (skip loading entire zero rows)
3. Improved C writing helps overall memory efficiency

---

## Register Pressure Analysis

**Current (Kernel 17):**
```cpp
WMITER = 2, TM = 1, WNITER = 4, TN = 8

float regM[WMITER * TM] = float[2 * 1] = float[2]
float regN[WNITER * TN] = float[4 * 8] = float[32]
float threadResults[WMITER * TM * WNITER * TN] = float[2*1*4*8] = float[64]

Total: 2 + 32 + 64 = 98 floats = 392 bytes per thread
```

**Proposed (B-Transpose):**
```cpp
WMITER = 8, TM = 8, WNITER = 1, TN = 1

float regM[WMITER * TM] = float[8 * 8] = float[64]
float regN[WNITER * TN] = float[1 * 1] = float[1]
float threadResults[WMITER * TM * WNITER * TN] = float[8*8*1*1] = float[64]

Total: 64 + 1 + 64 = 129 floats = 516 bytes per thread
```

**Analysis:**
- **Increase: +31% register usage** (129 vs 98 floats)
- **Impact**: Might reduce occupancy slightly
- **Mitigation**: Can tune WMITER/WNITER to reduce registers if needed

**Occupancy estimate:**
- Assume 65536 registers per SM
- Current: 98 floats → ~40 registers/thread → 1638 threads/SM (limited by other factors)
- Proposed: 129 floats → ~52 registers/thread → 1260 threads/SM
- **Conclusion**: Should still maintain good occupancy (typical target is 1024-2048 threads/SM)

---

## Conclusion

The B-transpose approach is **architecturally sound** and addresses the fundamental divergence problem. Key next steps:

1. ✅ Implement B-transpose preprocessing kernel
2. ✅ Implement prototype ESMM B-transpose kernel (K19)
3. ✅ Benchmark and profile
4. ✅ Iterate on warp tile dimensions if needed (maybe 64×64 for balance?)
