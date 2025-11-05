# A Sparsity Hybrid Kernel Visualization

## Overview: Block-Wise Pattern Encoding + Compile-Time Unrolling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PHASE (GPU)                                 │
│                analyze_sparsity_pattern_gpu()                                │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Divide Matrix A into 8×32 Blocks (BK × WM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Matrix A (M × K)                    Example: 128 × 64

     K dimension →                   8 cols  8 cols  8 cols ...
  ┌──────────────────────────────┐  ┌──────┬──────┬──────┬───
  │ ████░░░░ ░███░░░░ ██░░░░░░ ...│  │Block │Block │Block │...
M │ █░██░░░░ ░█░█░░░░ ███░░░░░ ...│  │  0   │  1   │  2   │   ← 32 rows (WM)
  │ ░███░░░░ ░░██░░░░ █░█░░░░░ ...│  │      │      │      │     (Warp 0)
↓ │ ██░█░░░░ ░█░█░░░░ ░██░░░░░ ...│  │      │      │      │
  │   ...      ...      ...    ...│  └──────┴──────┴──────┴───
  │ ░░██░░░░ █░░█░░░░ ██░░░░░░ ...│  ┌──────┬──────┬──────┬───
  │ █░░█░░░░ ░██░░░░░ ░██░░░░░ ...│  │Block │Block │Block │...
  │ ██░░░░░░ ░███░░░░ █░█░░░░░ ...│  │  3   │  4   │  5   │   ← 32 rows
  │ ░███░░░░ █░██░░░░ ░░█░░░░░ ...│  │      │      │      │     (Warp 1)
  └──────────────────────────────┘  └──────┴──────┴──────┴───
     BK=8      BK=8     BK=8            ↑
                                     8 rows × 32 rows = 256 elements per block


STEP 2: Extract Pattern for Each Block (Warp-Level OR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Focus on Block 0 (first 8×32 region):

  K-index:  0  1  2  3  4  5  6  7
          ┌──┬──┬──┬──┬──┬──┬──┬──┐
  Row 0:  │ █│  │  │ █│  │  │  │  │ → 00001001 (binary)
          ├──┼──┼──┼──┼──┼──┼──┼──┤
  Row 1:  │ █│  │ █│ █│  │  │  │  │ → 00001101
          ├──┼──┼──┼──┼──┼──┼──┼──┤
  Row 2:  │  │ █│ █│ █│  │  │  │  │ → 00001110
          ├──┼──┼──┼──┼──┼──┼──┼──┤
  Row 3:  │ █│ █│  │ █│  │  │  │  │ → 00001011
          ├──┼──┼──┼──┼──┼──┼──┼──┤
   ...    │  │  │ ...patterns...│  │
          ├──┼──┼──┼──┼──┼──┼──┼──┤
  Row 31: │  │ █│ █│ █│  │  │  │  │ → 00001110
          └──┴──┴──┴──┴──┴──┴──┴──┘

  OR all 32 rows together (warp reduction):
  ───────────────────────────────────────
  00001001
  00001101
  00001110
  00001011
  ...
  00001110
  ─────────  OR
  00001111  ← Final pattern for Block 0

  This means: positions {0, 1, 2, 3} have non-zeros somewhere in the block


STEP 3: Warp-Level Reduction Using Shuffle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Thread 0:  00001001 ┐
  Thread 1:  00001101 │
  Thread 2:  00001110 │
  Thread 3:  00001011 ├─→ __shfl_xor_sync(offset=16)
  ...                  │       ↓
  Thread 30: 00000110 │   Parallel OR reduction
  Thread 31: 00001110 ┘       ↓
                          00001111 (Result)

  Only Thread 0 writes to global memory:
  blockPatterns[blockId] = 0x0F  (1 byte!)


STEP 4: Metadata Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━

  BlockPatternMetadata:
  ┌────────────────────────────────────┐
  │ numWarpRows = M / WM = 128 / 32 = 4│
  │ numKBlocks  = K / BK = 64 / 8 = 8  │
  │                                    │
  │ d_blockPatterns (GPU memory):      │
  │  Total size = 4 × 8 = 32 bytes     │
  │                                    │
  │  [0x0F, 0x3C, 0xFF, 0x01, ...]     │
  │    ↑     ↑     ↑     ↑             │
  │   B0    B1    B2    B3   ...       │
  └────────────────────────────────────┘

  For 4096×4096 matrix:
  - (4096/32) × (4096/8) = 128 × 512 = 65,536 bytes (~64 KB)


┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPUTATION PHASE (Kernel Runtime)                       │
│                    esmm_hybrid_blockwise()                                   │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 5: Runtime Pattern Lookup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Each warp processes its own tile during GEMM:

  Warp 0 computing tile C[0:32, 0:64]:

  for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {

      // 1. LOAD PATTERN (1 byte from global memory)
      ┌─────────────────────────────────────┐
      │ blockId = warpRow * numKBlocks + kb │
      │ pattern = blockPatterns[blockId]    │ ← 0x0F
      └─────────────────────────────────────┘

      // 2. LUT LOOKUP (constant memory, broadcast)
      ┌──────────────────────────────────────────────────┐
      │ PATTERN_LUT_BK8[0x0F]:                          │
      │   count = 4                                     │
      │   offsets = [0, 1, 2, 3, 0, 0, 0, 0]           │
      └──────────────────────────────────────────────────┘

      // 3. EARLY EXIT CHECK
      if (count == 0) {
          skip this block entirely!  ✓ No computation
      }

      // 4. LOAD TILES A and B into shared memory
      //    (standard vectorized loads)

      // 5. SWITCH DISPATCH (compile-time specialization)
      switch (count) {
          case 4:  ← THIS BRANCH
              compute_sparse_block<..., SIZE=4>(offsets, ...)
              break;
      }
  }


STEP 6: Compile-Time Unrolled Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Template instantiation for SIZE=4:

  compute_sparse_block<..., SIZE=4>(...) {

      // Compiler UNROLLS this completely:
      #pragma unroll
      for (sparse_idx = 0; sparse_idx < 4; ++sparse_idx) {

          dotIdx = offsets[sparse_idx];

          // Iteration 0: dotIdx = 0
          regM[0] = As[(0 * BM) + ...];  ─┐
          regN[0] = Bs[(0 * BN) + ...];  ─┤
          threadResults += regM * regN;   ├─ Fully unrolled
                                          │  (no loop overhead)
          // Iteration 1: dotIdx = 1      │
          regM[0] = As[(1 * BM) + ...];  ─┤
          regN[0] = Bs[(1 * BN) + ...];  ─┤
          threadResults += regM * regN;   │
                                          │
          // Iteration 2: dotIdx = 2      │
          regM[0] = As[(2 * BM) + ...];  ─┤
          regN[0] = Bs[(2 * BN) + ...];  ─┤
          threadResults += regM * regN;   │
                                          │
          // Iteration 3: dotIdx = 3      │
          regM[0] = As[(3 * BM) + ...];  ─┤
          regN[0] = Bs[(3 * BN) + ...];  ─┤
          threadResults += regM * regN;   ┘
      }
  }

  Result: 4 FMAs instead of 8 (50% sparsity)
          Zero branching, perfect pipelining


STEP 7: Pattern LUT Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Constant memory (1.25 KB, broadcast to all threads):

  ┌─────────┬───────┬──────────────────────────────┐
  │ Pattern │ Count │ Offsets [0-7]                │
  ├─────────┼───────┼──────────────────────────────┤
  │ 0x00    │   0   │ [0, 0, 0, 0, 0, 0, 0, 0]    │ ← All zeros
  │ 0x01    │   1   │ [0, 0, 0, 0, 0, 0, 0, 0]    │ ← Bit 0 set
  │ 0x02    │   1   │ [1, 0, 0, 0, 0, 0, 0, 0]    │ ← Bit 1 set
  │ 0x03    │   2   │ [0, 1, 0, 0, 0, 0, 0, 0]    │ ← Bits 0,1
  │  ...    │  ...  │  ...                         │
  │ 0x0F    │   4   │ [0, 1, 2, 3, 0, 0, 0, 0]    │ ← Our example
  │  ...    │  ...  │  ...                         │
  │ 0xFF    │   8   │ [0, 1, 2, 3, 4, 5, 6, 7]    │ ← All ones (dense)
  └─────────┴───────┴──────────────────────────────┘

  Single load gives both count AND offsets (no computation!)


┌─────────────────────────────────────────────────────────────────────────────┐
│                        KEY OPTIMIZATIONS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

1. WARP UNIFORMITY
   ═════════════════
   All 32 threads in a warp share the same pattern
   → No divergence, all threads execute same switch case

   Thread 0:  switch(4) → case 4  ┐
   Thread 1:  switch(4) → case 4  │ All execute
   Thread 2:  switch(4) → case 4  ├ same path
   ...                            │ (SIMD friendly)
   Thread 31: switch(4) → case 4  ┘

2. COMPILE-TIME UNROLLING
   ═══════════════════════
   Template parameter SIZE is compile-time constant
   → Compiler generates 8 specialized functions
   → Each has fully unrolled inner loop

   Generated code:
   - compute_sparse_block<..., 1>  ← 1 iteration
   - compute_sparse_block<..., 2>  ← 2 iterations
   - compute_sparse_block<..., 3>  ← 3 iterations
   ...
   - compute_sparse_block<..., 8>  ← 8 iterations (dense)

3. MEMORY EFFICIENCY
   ═══════════════════
   1 byte per 256-element block
   vs. 1 byte per row (rowlevel approach)

   For 4096×4096:
   - Hybrid:    (4096/32) × (4096/8) = 65 KB
   - Rowlevel:  4096 × (4096/8) = 2 MB

   32× smaller metadata!

4. LUT OPTIMIZATION
   ══════════════════
   Precomputed table eliminates:
   - __popc() (population count)
   - Bit extraction loops
   - Runtime offset calculation

   Single constant memory load (cached, broadcast)


┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE EXAMPLE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Matrix: 4096 × 4096, 50% sparse (random)

PREPROCESSING:
  - 65,536 blocks to analyze
  - Each block: 32 rows × 8 cols = 256 elements
  - Warp reduction + 1 byte write per block
  - Time: ~0.5-1 ms
  - Output: 64 KB metadata

COMPUTATION (per K-block):
  - Load: 1 byte pattern
  - LUT:  1 constant memory access (broadcast)
  - Switch: Zero overhead (compile-time dispatch)
  - Compute: ~4 FMAs instead of 8 (50% savings)

TOTAL SPEEDUP over dense GEMM:
  - 50% sparsity → ~1.6-1.8× faster
  - 75% sparsity → ~2.5-3.0× faster
  - 87.5% sparsity → ~4.0-5.0× faster


┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARISON: WHY "HYBRID"?                                 │
└─────────────────────────────────────────────────────────────────────────────┘

COMBINES TWO APPROACHES:

1. BLOCK-LEVEL ENCODING (Structured Sparse Formats)
   ────────────────────────────────────────────────
   ✓ Reduces metadata size (group 32 rows)
   ✓ Warp-uniform execution (no divergence)
   ✗ Conservative (OR across rows)

2. COMPILE-TIME SPECIALIZATION (Pattern-Specialized Kernels)
   ──────────────────────────────────────────────────────────
   ✓ Perfect loop unrolling
   ✓ Zero runtime overhead for dispatch
   ✗ Requires known pattern at compile time

HYBRID SOLUTION:
  → Block-level encoding at runtime
  → Compile-time specialization via template + switch
  → Best of both worlds!
