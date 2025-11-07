# B-Matrix Transpose Analysis: Solving Warp Divergence

## The Problem with Current B-Sparsity (Recap)

**Current Architecture** (Kernel 17 - A-only sparsity):
- Warp tile: **32 rows × 64 columns** (WM=32, WN=64)
- Thread tile: **1 row × 8 columns** (TM=1, TN=8)
- Each warp has 32 threads processing different output positions

**Thread Mapping within Warp:**
```
threadIdxInWarp = 0..31
threadColInWarp = threadIdxInWarp % 2  (0 or 1)
threadRowInWarp = threadIdxInWarp / 2  (0..15)

Thread 0:  row 0,  cols 0-7   (threadRow=0, threadCol=0, TN=8)
Thread 1:  row 0,  cols 8-15  (threadRow=0, threadCol=1, TN=8)
Thread 2:  row 1,  cols 0-7   (threadRow=1, threadCol=0, TN=8)
...
Thread 31: row 15, cols 8-15  (threadRow=15, threadCol=1, TN=8)
```

With WNITER=4, each thread iterates over 4 sub-blocks to cover 64 columns:
- Iteration 0: columns 0-15
- Iteration 1: columns 16-31
- Iteration 2: columns 32-47
- Iteration 3: columns 48-63

**Why B-Sparsity Fails:**
- Each thread needs different columns of B (different n values)
- B-sparsity patterns are per 8 columns (8×8 blocks in K×N)
- **Every thread sees a DIFFERENT B-pattern → warp divergence**

## The Transpose Solution

### Core Insight

**Matrix multiplication:**
- Original: `C[m,n] = sum_k( A[m,k] × B[k,n] )`
- With B^T: `C[m,n] = sum_k( A[m,k] × B^T[n,k] )`

**Key observation:**
- If we **swap the warp tile orientation** from "wide" (32×64) to "tall" (64×32)
- Make all threads in warp compute outputs for the **SAME column** (same n value)
- Then all threads access the **SAME row of B^T** → same sparsity pattern!

### Proposed New Architecture

**New Warp Tile: 64 rows × 32 columns** (flip dimensions)
- WM = 64, WN = 32
- TM = 8, TN = 1 (flip thread tile too)
- WMITER = 4, WNITER = 1 (or adjust as needed)

**New Thread Mapping:**
```
Within a warp computing a 64×32 tile:
- All 32 threads compute outputs for SAME 32 columns
- Each thread handles 8 different rows

Example (simplified):
Thread 0:  rows 0-7,   col 0
Thread 1:  rows 0-7,   col 1
Thread 2:  rows 0-7,   col 2
...
Thread 31: rows 0-7,   col 31

(with WMITER=4, iterate over row blocks to cover all 64 rows)
```

**Loading B^T for computation:**
- Output column n requires row n of B^T
- All threads computing column n need the same B^T[n, k] values
- **All threads check the same 8-bit sparsity pattern for row n → warp-uniform!**

### Detailed Execution Flow

**Preprocessing:**
1. Transpose B to B^T (N×K)
2. Encode B^T row-wise sparsity: 8-bit pattern per 8×N block
3. Store patterns: `B_patterns[kBlock * numNBlocks + nBlock]`

**Runtime (K-loop iteration):**
```cuda
for (int kBlock = 0; kBlock < numKBlocks; kBlock++) {
    // Load A tile (BM=64 × BK=8) into shared memory
    // Load B^T tile (BN=32 × BK=8) into shared memory

    // For each warp computing a 64×32 output tile:
    for (int colIdx = 0; colIdx < 32; colIdx++) {
        // ALL threads in warp need row colIdx of B^T
        uint8_t B_pattern = B_patterns[kBlock * numNBlocks + globalCol + colIdx];

        // Warp-uniform check! All threads see same pattern
        if (B_pattern == 0) {
            continue; // All threads skip together - no divergence!
        }

        // Load only non-zero elements from B^T[colIdx, :]
        uint8_t count = PATTERN_LUT_BK8[B_pattern].count;
        uint8_t* offsets = PATTERN_LUT_BK8[B_pattern].offsets;

        // Compute with sparse elements (fully unrolled via switch on count)
        for (int i = 0; i < count; i++) {
            int k = offsets[i];
            // Each thread loads its own A values (different rows)
            // But all threads use same B^T value (same row of B^T)
            for (int rowIdx = 0; rowIdx < 8; rowIdx++) {
                C[threadRow*8 + rowIdx][col] += A[threadRow*8 + rowIdx][k] * B^T[col][k];
            }
        }
    }
}
```

### Benefits

✅ **Warp-Uniform B-Sparsity Checks**: All threads see same B-pattern
✅ **Zero Divergence**: Skip/execute decisions are warp-uniform
✅ **A-Sparsity Preserved**: Still encode A row-wise (works with new mapping)
✅ **Compile-Time Unrolling**: Switch on pattern count still works
✅ **Bandwidth Savings**: Skip loading entire B^T rows that are zero

### Trade-offs and Considerations

**Pros:**
1. Solves the fundamental divergence problem
2. Can exploit both A and B sparsity simultaneously
3. Maintains memory coalescing (need to verify)
4. Reuses existing pattern LUT infrastructure

**Cons/Risks:**
1. **Transpose overhead**: One-time cost to transpose B → B^T
   - For training: amortized over many reuses
   - For inference: might be acceptable
2. **Memory layout changes**: Need to verify coalescing for:
   - A loads (now accessing different rows per thread)
   - B^T loads (accessing same row across threads)
   - C writes (different pattern than before)
3. **Warp tile shape change**: 64×32 vs 32×64 might have different performance characteristics
4. **Shared memory access patterns**: Need to re-verify bank conflicts
5. **Register pressure**: TM=8 (vs TM=1) means more registers per thread

### Critical Questions to Answer

1. **Memory Coalescing**:
   - A loads: Each thread loads from different rows but consecutive threads load consecutive rows? Need verification
   - B^T loads: All threads load same row → broadcast? Or conflicts?
   - C writes: Still coalesced?

2. **Shared Memory Banking**:
   - As[BM=64 × BK=8]: Different access pattern with TM=8
   - Bs[BN=32 × BK=8]: Need to check bank conflicts

3. **Register Pressure**:
   - Current: regM[WMITER=2 × TM=1] = 2 floats
   - Proposed: regM[WMITER=4 × TM=8] = 32 floats
   - Will this cause register spilling?

4. **Occupancy**:
   - More registers per thread → lower occupancy?
   - Need to profile

5. **A-Sparsity Compatibility**:
   - A patterns are encoded per 8×32 block (BK × WM_old)
   - With WM=64, need to re-encode as 8×64 blocks?
   - Or adjust pattern granularity?

### Implementation Plan

**Phase 1: Proof of Concept**
1. Create B-transpose preprocessing kernel
2. Implement basic B^T sparsity kernel (B-only, no A sparsity first)
3. Verify correctness
4. Benchmark vs dense baseline

**Phase 2: Optimize**
5. Add A-sparsity back (dual sparsity)
6. Optimize shared memory access patterns
7. Tune warp tile dimensions (try 64×32, 64×64, etc.)
8. Profile memory coalescing and occupancy

**Phase 3: Comparison**
9. Compare vs Kernel 17 (A-only)
10. Test on different sparsity patterns
11. Measure transpose overhead amortization

### Expected Performance

**Theoretical speedup for 50% B-sparsity:**
- Skip 50% of B loads → ~25% bandwidth reduction
- Skip 50% of FMAs → ~33% compute reduction
- If overhead is minimal, could see 1.3-1.5× speedup over A-only

**Combined A+B at 50% each:**
- Skip 75% of work (1 - 0.5*0.5 = 0.75)
- Theoretical 4× speedup over dense
- Realistic: 2-3× if overhead is well-managed

### Files to Modify

1. `src/preprocessors/b_preprocessor_transpose.cu` (new)
   - Transpose B → B^T
   - Encode row-wise patterns

2. `src/kernels/esmm_btranspose.cu` (new)
   - New kernel with flipped warp tile
   - B^T sparsity only (start simple)

3. `src/kernels/esmm_dual_transpose.cu` (new)
   - Combined A + B^T sparsity
   - Full optimization

4. `include/runners.cuh`
   - Add runner functions

5. `driver.cu`
   - Add kernel choices 19, 20

### Next Steps

1. Validate architectural assumptions (memory patterns)
2. Implement prototype B-transpose kernel
3. Run initial benchmarks
4. Iterate based on profiling data
