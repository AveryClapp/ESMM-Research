# ESMM Kernel Architecture Visualization

## Overview
This document visualizes the Emergent Sparsity Matrix Multiplication (ESMM) kernel and its preprocessing stage.

---

## Part 1: A-Matrix Preprocessing Kernel

### Purpose
Scan matrix A to identify which columns (K-dimension) have non-zero values for each warp tile, enabling the main kernel to skip zero computations.

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Matrix A (M × K)                              │
│                                                                   │
│  ┌──────────────────────────┐                                   │
│  │   Thread Block (BM=128)  │  ← Each block processes 128 rows  │
│  │  ┌─────────────────────┐ │                                   │
│  │  │ Warp Row 0 (WM=32)  │ │  ← 4 warp rows per block         │
│  │  ├─────────────────────┤ │     (BM/WM = 128/32 = 4)         │
│  │  │ Warp Row 1 (WM=32)  │ │                                   │
│  │  ├─────────────────────┤ │                                   │
│  │  │ Warp Row 2 (WM=32)  │ │                                   │
│  │  ├─────────────────────┤ │                                   │
│  │  │ Warp Row 3 (WM=32)  │ │                                   │
│  │  └─────────────────────┘ │                                   │
│  └──────────────────────────┘                                   │
│         │                                                         │
│         │ Scan in K-dimension →                                  │
│         ▼                                                         │
│  ┌──────────────────────────┐                                   │
│  │   K-Block 0  │  K-Block 1  │ ... (BK=8 each)                 │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Warp Processing Detail

```
For each K-block (8 columns):
┌────────────────────────────────────────────────────────────┐
│  Thread in Warp 0 (32 threads scan 32 rows)                │
│                                                              │
│  dotIdx=0  dotIdx=1  dotIdx=2  ...  dotIdx=7 (BK=8)       │
│     ↓         ↓         ↓              ↓                    │
│  [  A  ] → [  A  ] → [  A  ] → ... → [  A  ]              │
│  [  .  ]   [  .  ]   [  .  ]         [  .  ]              │
│  [  .  ]   [  .  ]   [  .  ]         [  .  ]              │
│  [  .  ]   [  .  ]   [  .  ]         [  .  ]              │
│  [ row ]   [ row ]   [ row ]         [ row ]              │
│  [ 0-31]   [ 0-31]   [ 0-31]         [ 0-31]              │
│                                                              │
│  Each column: __ballot_sync() checks if ANY thread has !=0  │
│  If active != 0: Record column index (dotIdx)               │
└────────────────────────────────────────────────────────────┘
```

### Output Format (Shared Memory → Global Memory)

```
For each K-block, each warp row stores:
┌────────────────────────────────────────────────────┐
│ [count] [off0] [off1] [off2] [off3] | Pattern     │
├────────────────────────────────────────────────────┤
│   3       0      2      5      0    | "10100100" │
│   ^       ^      ^      ^      ^                   │
│   │       └──────┴──────┴───── Offset indices     │
│   └─ Number of non-zero columns found             │
│                                                     │
│ If count == -1: Sparsity > 50%, fallback to dense │
│ If count == 0:  All zeros, skip this K-block       │
│ If 0 < count ≤ 4: Use offsets for sparse compute  │
└────────────────────────────────────────────────────┘

Total size per block: (K/BK) × NUM_WARP_ROWS × WMITER × 5 ints
Example: (1024/8) × 4 × 1 × 5 = 2,560 integers per block
```

### Data Flow

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Global Mem  │      │  Shared Mem  │      │  Global Mem  │
│  Matrix A    │ ───> │  denseList   │ ───> │   A_LIST     │
│              │ Load │  (per block) │ Copy │  (all blocks)│
└──────────────┘      └──────────────┘      └──────────────┘
     Scan in             Accumulate            Vectorized
   K-dimension          with atomics            copy (int4)
```

---

## Part 2: ESMM Main Kernel

### Thread Block Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                Thread Block (BM=128, BN=128)                     │
│                                                                   │
│  8 Warps arranged in 2×4 grid:                                  │
│                                                                   │
│         BN = 128 (columns) →                                     │
│    ┌────────────────────────────────────┐                       │
│    │  Warp(0,0)  │  Warp(0,1)           │                       │
│    │   WM×WN     │   WM×WN              │  ← Warp Row 0        │
│  B ├─────────────┼──────────────────────┤                       │
│  M │  Warp(1,0)  │  Warp(1,1)           │  ← Warp Row 1        │
│  = ├─────────────┼──────────────────────┤                       │
│  1 │  Warp(2,0)  │  Warp(2,1)           │  ← Warp Row 2        │
│  2 ├─────────────┼──────────────────────┤                       │
│  8 │  Warp(3,0)  │  Warp(3,1)           │  ← Warp Row 3        │
│    └─────────────┴──────────────────────┘                       │
│         WN=64         WN=64                                      │
│                                                                   │
│  Each warp: 32 threads computing 32×64 output tile              │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Warp Computation (32×64 tile)

```
Warp computes WM×WN = 32×64 output elements

Per Thread: TM×(WNITER×TN) = 1×(4×8) = 1×32 elements

┌──────────────────────────────────────────────────────┐
│  32 Threads × 32 outputs each = 1024 outputs         │
│                                                        │
│  Thread layout in warp:                               │
│                                                        │
│       WSUBN/TN = 8 threads →                          │
│    ┌────────────────────────┐                         │
│  W │ T0  T1  T2 ... T7      │  Each thread computes  │
│  S ├────────────────────────┤  1 row × 32 cols       │
│  U │ T8  T9  T10... T15     │  (TM=1, WNITER×TN=32) │
│  B ├────────────────────────┤                         │
│  M │ T16 T17 T18... T23     │                         │
│  / ├────────────────────────┤                         │
│  T │ T24 T25 T26... T31     │                         │
│  M └────────────────────────┘                         │
│  =      WNITER = 4 iterations across 64 columns       │
│  4                                                     │
└──────────────────────────────────────────────────────┘
```

### Memory Flow (K-loop iteration)

```
ITERATION k (processing K-block):

┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Load to Shared Memory                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Global A                    Shared As                           │
│  (128×8 tile)  ────────>    (8×128)                             │
│                  float4       Transposed                         │
│                  vectorized   for coalesced                      │
│                  load         access                             │
│                                                                   │
│  Global B                    Shared Bs                           │
│  (8×128 tile)  ────────>    (8×128)                             │
│                  float4       Direct                             │
│                  vectorized   layout                             │
│                                                                   │
│                         __syncthreads()                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Compute (dotIdx loop, BK=8)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  For dotIdx = 0 to 7:                                            │
│                                                                   │
│    Shared As[dotIdx,:] ──> regM[WMITER × TM]                    │
│    (1 col of As)            (1 value per thread)                 │
│                                                                   │
│    Shared Bs[dotIdx,:] ──> regN[WNITER × TN]                    │
│    (1 row of Bs)            (32 values: 4 iters × 8)            │
│                                                                   │
│    Compute:                                                       │
│    threadResults[1×32] += regM[1] × regN[32]                    │
│                           (outer product)                         │
│                                                                   │
│    This is where sparsity masking would skip dotIdx if A=0       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: Writeback to Global C                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  After all K-blocks processed:                                   │
│                                                                   │
│  threadResults[1×32]  ────────>  Global C                       │
│  (registers)             float4   (1×32 tile)                    │
│                          vectorized                              │
│                                                                   │
│  Each thread writes its 1×32 output to corresponding location    │
└─────────────────────────────────────────────────────────────────┘
```

### Register Usage Per Thread

```
┌────────────────────────────────────────────────────────┐
│  Thread Registers (per thread in warp):                │
├────────────────────────────────────────────────────────┤
│                                                          │
│  regM[WMITER × TM = 1×1 = 1 float]                     │
│  ┌───┐                                                  │
│  │ a │  Single value from A                            │
│  └───┘                                                  │
│                                                          │
│  regN[WNITER × TN = 4×8 = 32 floats]                   │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                    │
│  │ b │ b │ b │ b │ b │ b │ b │ b │  8 values (iter 0) │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ b │ b │ b │ b │ b │ b │ b │ b │  8 values (iter 1) │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ b │ b │ b │ b │ b │ b │ b │ b │  8 values (iter 2) │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ b │ b │ b │ b │ b │ b │ b │ b │  8 values (iter 3) │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                    │
│                                                          │
│  threadResults[WMITER×TM × WNITER×TN = 1×32 = 32]      │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                    │
│  │ c │ c │ c │ c │ c │ c │ c │ c │                    │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ c │ c │ c │ c │ c │ c │ c │ c │                    │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ c │ c │ c │ c │ c │ c │ c │ c │                    │
│  ├───┼───┼───┼───┼───┼───┼───┼───┤                    │
│  │ c │ c │ c │ c │ c │ c │ c │ c │  Accumulates      │
│  └───┴───┴───┴───┴───┴───┴───┴───┘  across K          │
│                                                          │
│  Total: 65 float registers per thread                   │
└────────────────────────────────────────────────────────┘
```

---

## Part 3: How Preprocessing Enables ESMM Optimization

### Without Preprocessing (Current ESMM)

```
K-loop (all 128 iterations for K=1024, BK=8):
┌──────────────────────────────────────────┐
│  dotIdx: 0 1 2 3 4 5 6 7                │
│  Load A: ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓  (8 loads)    │
│  Load B: ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓  (8 loads)    │
│  Compute:✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓  (8 FMAs)     │
└──────────────────────────────────────────┘
All 8 dot products computed even if A has zeros
```

### With Preprocessing (Optimized)

```
Preprocessed offsets: [count=3, off0=0, off1=3, off2=7]
                       (only columns 0, 3, 7 have non-zero)

K-loop iteration:
┌──────────────────────────────────────────┐
│  dotIdx: 0 1 2 3 4 5 6 7                │
│  Load A: ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✓  (3 loads)    │
│  Load B: ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✓  (3 loads)    │
│  Compute:✓ ✗ ✗ ✓ ✗ ✗ ✗ ✓  (3 FMAs)     │
└──────────────────────────────────────────┘
Only process columns with non-zero values → 62.5% savings!
```

### Sparse Kernel Pseudocode

```c
// Read preprocessed data for this warp+kblock
int count = A_LIST[offset];
if (count == 0) {
    // Skip entire K-block
    continue;
} else if (count == -1) {
    // Dense fallback (>50% sparse)
    for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
        // Standard computation
    }
} else {
    // Sparse path: only process non-zero columns
    for (int i = 0; i < count; i++) {
        int dotIdx = A_LIST[offset + 1 + i];  // Get offset
        // Load only this column and compute
        regM = As[dotIdx * BM + ...];
        regN = Bs[dotIdx * BN + ...];
        threadResults += regM * regN;
    }
}
```

---

## Performance Characteristics

### Memory Access Pattern

```
Dense GEMM:
- A: (M×K) sequential reads in K-blocks
- B: (K×N) sequential reads in K-blocks
- C: (M×N) sequential writes

ESMM with Preprocessing:
- A_LIST: Random access to offset metadata (cached in L1)
- A: Potentially non-contiguous reads based on offsets
- B: Same as dense (always loaded)
- C: Same as dense
```

### Occupancy vs Register Pressure

```
Per Thread Resources:
- Registers: ~65 floats = 260 bytes
- Shared memory per block:
  - As: 128×8×4 = 4KB
  - Bs: 128×8×4 = 4KB
  - Total: 8KB

GPU limits (example: A100):
- 65,536 registers per SM
- 164KB shared memory per SM
- Max 2048 threads per SM

Theoretical occupancy:
- Register limited: 65536/(260×256) ≈ 98% (good!)
- Shared limited: 164KB/8KB = 20 blocks (excellent!)
```

---

## Key Optimizations

1. **Warp-level tiling**: Reduces synchronization overhead
2. **Vectorized memory (float4)**: 4x throughput on loads/stores
3. **Transposed A in shared**: Column-major for broadcast access
4. **Register blocking**: Minimizes shared memory accesses
5. **Preprocessed offsets**: Skip zero computations dynamically

## Sparsity Patterns

```
Pattern: "11111111" (dense)
  Preprocessing: All offsets stored, count=8, uses dense path

Pattern: "10100000" (25% dense)
  Preprocessing: [count=2, off0=0, off1=2], uses sparse path

Pattern: "10101010" (50% dense)
  Preprocessing: [count=4, off0=0, off1=2, off2=4, off3=6]

Pattern: "11111110" (87.5% dense)
  Preprocessing: [count=-1], fallback to dense (overhead not worth it)
```

---

## Summary

The ESMM kernel combines aggressive tiling (block → warp → thread) with
preprocessing-guided sparsity exploitation. The preprocessing kernel scans
matrix A once to build an offset table, then the main kernel uses these
offsets to skip zero-valued computations, achieving speedup proportional
to the sparsity ratio (up to 2x for 50% sparsity).
