# Why B-Sparsity Is So Hard (And How to Fix It)

## The Core Problem with B-Sparsity

### Issue 1: Irregular Memory Access
```cpp
// A-sparsity (WORKS):
for (int sparse_idx = 0; sparse_idx < count; ++sparse_idx) {
    int k = offsets[sparse_idx];
    float a = A[i][k];        // Irregular read - OK
    float b = B[k][j];        // Regular access by column j
    C[i][j] += a * b;
}

// B-sparsity (BREAKS):
for (int sparse_idx = 0; sparse_idx < count; ++sparse_idx) {
    int j = offsets[sparse_idx];
    float a = A[i][k];        // Regular access
    float b = B[k][j];        // Irregular column access - DISASTER!
    C[i][j] += a * b;         // Scattered writes - DISASTER!
}
```

**Problem:** Scattered column reads + scattered output writes = horrible memory performance

### Issue 2: Output Write Complexity
```cpp
// Dense output: Each thread writes to contiguous C locations
C[threadRow][threadCol : threadCol+8] = results[0:8];  // Vectorized!

// Sparse output: Each thread writes to DIFFERENT C locations
for (int idx = 0; idx < sparse_count; idx++) {
    int col = sparse_offsets[idx];
    C[threadRow][col] = results[idx];  // Scattered! No coalescing!
}
```

### Issue 3: Warp Divergence
```cpp
// Different threads in same warp have different sparse patterns
Thread 0: columns [2, 5, 7]       -> 3 FMAs
Thread 1: columns [1, 3, 4, 6]    -> 4 FMAs
Thread 2: columns [0, 2]          -> 2 FMAs
// MASSIVE divergence! Execution serializes!
```

---

## Why Previous B-Sparsity Attempts Failed

### Common Attempt #1: Per-Thread Column Skipping
```cpp
// Try to skip zero columns per thread
for (int col = 0; col < N; col++) {
    if (B[k][col] != 0) {
        // Process this column
    }
}
```

**Why it fails:**
- ❌ Each thread checks DIFFERENT columns (divergence)
- ❌ Scattered memory access to B
- ❌ Scattered output writes to C
- ❌ Branch overhead exceeds benefit

### Common Attempt #2: Column Bitmaps
```cpp
// Pre-compute which columns are non-zero
uint64_t col_mask = get_nonzero_columns(B, k);
for (int col = 0; col < 64; col++) {
    if (col_mask & (1ULL << col)) {
        // Process
    }
}
```

**Why it fails:**
- ❌ Still divergent (different masks per thread)
- ❌ Still scattered memory access
- ❌ Overhead of mask checking + branch misprediction

### Common Attempt #3: Block-Level Skipping
```cpp
// Skip entire column blocks if fully zero
for (int colBlock = 0; colBlock < N/64; colBlock++) {
    if (B_block_patterns[colBlock] == 0) continue;
    // Process all 64 columns
}
```

**Why it fails (with old config):**
- ⚠️ WN=64: Column blocks too large (rarely fully zero)
- ⚠️ If block has 1 non-zero, process all 64 columns
- ⚠️ At 50% sparsity: P(block fully zero) = 0.5^(8×64) ≈ 0

---

## The NEW Approach: Warp-Uniform Column Skipping with WN=32

### Key Insight: Smaller Blocks + Warp-Uniform Decisions

**Old config:** WN=64 → blocks too large
**New config:** WN=32 → blocks 2× smaller → 2^256 times more likely to be fully zero!

### Probability Analysis

At 50% sparsity:
```
P(8×64 block is fully zero) = 0.5^512 ≈ 0      (old WN=64)
P(8×32 block is fully zero) = 0.5^256 ≈ 10^-77 (still ~0)
```

Wait... this still doesn't work! Let me recalculate:

```
P(single element is zero) = 0.5
P(8×32 block ALL zero) = 0.5^256 ≈ 0

BUT: We need to think about it differently!
```

### The Real Insight: Warp-Level Decision, Not Element-Level

**Don't ask:** "Is this 8×32 block all zero?"
**Instead ask:** "Is there ANY work for this warp to do?"

```cpp
// Warp-level decision (warp-uniform!)
__shared__ bool warp_has_work[NUM_WARPS];

// One thread per warp checks
if (laneId == 0) {
    uint8_t b_col_pattern = B_COL_PATTERNS[kBlock * (BN/WN) + warpCol];
    uint8_t a_row_pattern = A_ROW_PATTERNS[blockRow * (BM/WM) + warpRow];
    
    // If BOTH are zero, no work for this warp
    warp_has_work[warpIdx] = (b_col_pattern != 0) && (a_row_pattern != 0);
}
__syncthreads();

if (!warp_has_work[warpIdx]) {
    return;  // Entire warp skips - no divergence!
}
```

**This works because:**
- ✅ Warp-uniform decision (all threads agree)
- ✅ No divergence within warp
- ✅ Simple check (2 pattern loads + AND)
- ✅ With WN=32, patterns are smaller → more skipping opportunities

---

## Concrete Implementation Strategy

### Strategy 1: Warp-Level Early Exit (Easiest)

**Goal:** Skip warps where A-block OR B-block is fully zero

```cpp
// In kernel, before loading from shared memory
const uint8_t a_pattern = A_BLOCK_PATTERNS[globalWarpRow * numKBlocks + kBlock];
const uint8_t b_pattern = B_COL_PATTERNS[kBlock * (BN/WN) + warpCol];

// Early exit for entire warp (warp-uniform!)
if (a_pattern == 0 || b_pattern == 0) {
    continue;  // All threads in warp skip together
}
```

**Benefits:**
- Zero divergence
- Multiplicative savings: P(skip) = P(A=0) + P(B=0) - P(A=0)×P(B=0)
- At 50% sparsity each: P(skip) = 0.5 + 0.5 - 0.25 = 0.75 (75% skipping!)

**Cost:**
- 1 additional uint8_t load per warp per K-block (~5 cycles)
- 1 branch (warp-uniform, well-predicted)

**ROI:** 
- Cost: 5 cycles
- Benefit: Skip 75% of 8×32×64 FMAs = skip 12,288 FMAs ≈ 384 cycles
- **Ratio: 77:1** ✅✅✅

### Strategy 2: Column-Group Skipping (Medium difficulty)

**Goal:** Within a warp, skip entire column groups (8 columns at a time)

```cpp
// B pattern: which 8-column groups are non-zero
// For WN=32, we have 4 groups of 8
uint8_t b_col_groups = B_COL_GROUP_PATTERNS[...];  // 4 bits

for (int group = 0; group < 4; group++) {
    if (!(b_col_groups & (1 << group))) {
        continue;  // Skip this 8-column group
    }
    
    // Load and compute for this group
    for (int t = 0; t < 8; t++) {
        regN[group * 8 + t] = Bs[...];
    }
}
```

**Benefits:**
- More fine-grained skipping
- Still warp-uniform (all threads see same pattern)
- Saves B loads + FMAs for zero column groups

**Cost:**
- Slightly more complex pattern encoding
- Loop overhead for groups

### Strategy 3: Hybrid A+B with Adaptive Output (Advanced)

**Problem:** Output writes are still dense

**Solution:** Accumulate in shared memory, then compact writes

```cpp
__shared__ float C_sparse[BM * MAX_SPARSE_COLS];
__shared__ uint16_t C_col_indices[BM * MAX_SPARSE_COLS];
__shared__ uint16_t C_col_counts[BM];

// During computation
int sparse_output_idx = atomicAdd(&C_col_counts[threadRow], 1);
C_sparse[threadRow * MAX_SPARSE_COLS + sparse_output_idx] = result;
C_col_indices[threadRow * MAX_SPARSE_COLS + sparse_output_idx] = col;

// At the end, write sparsely
for (int i = 0; i < C_col_counts[threadRow]; i++) {
    int col = C_col_indices[threadRow * MAX_SPARSE_COLS + i];
    C[blockRow * BM + threadRow][col] = C_sparse[threadRow * MAX_SPARSE_COLS + i];
}
```

**Benefits:**
- Handles sparse output correctly
- Coalesces non-zero results before writing

**Cost:**
- Shared memory overhead
- Atomic operations for counting

---

## Recommended Implementation Path

### Phase 1: Warp-Level Joint A+B Skipping (DO THIS FIRST!)

**Effort:** 2-3 hours
**Expected speedup:** 1.10× → 1.4-1.6× on 50% doubly-sparse

```cpp
// Minimal change to existing K17
// Just add one check:

if (a_count == 0 || b_pattern == 0) {
    A += BK;
    B += BK * N;
    continue;
}
```

Where `b_pattern` comes from preprocessing B like you do for A.

### Phase 2: Verify It Works

Test scenarios:
1. A sparse (50%), B dense → should match current performance
2. A dense, B sparse (50%) → should be ~1.3-1.4× faster
3. A sparse (50%), B sparse (50%) → should be ~1.6× faster

### Phase 3: Profile and Iterate

```bash
ncu --set full ./driver
# Look for:
# - Are warps actually skipping?
# - Memory throughput improvement
# - Compute utilization
```

---

## Why It Might Work NOW (But Didn't Before)

| Factor | Old Config | New Config |
|--------|-----------|------------|
| WN size | 64 | 32 (2× smaller!) |
| Pattern granularity | 8×64 = 512 | 8×32 = 256 ✅ |
| Warp work amount | Larger chunks | Smaller chunks ✅ |
| Register pressure | High (256 threads) | Lower (128 threads) ✅ |
| WNITER overhead | 4 iterations | 2 iterations ✅ |

**The combination of:**
- Smaller WN (more skipping opportunities)
- Fewer threads (more registers for patterns)
- Fewer iterations (less loop overhead)

**Makes B-sparsity tractable NOW when it wasn't before!**

---

## Test Case to Try

```cpp
// Generate test case
// A: 50% sparse, block-structured
// B: 50% sparse, block-structured
// Both aligned to 8×32 blocks

// Expected results:
// Dense×Dense:    7,260 μs (cuBLAS)
// Sparse×Dense:   6,600 μs (current K17)
// Dense×Sparse:   ~5,500 μs (with B-skipping)
// Sparse×Sparse:  ~4,500 μs (with A+B skipping) = 1.6× speedup!
```

---

## The One Key Change to Try First

In `esmm_hybrid.cu`, around line 142:

```cpp
// BEFORE:
const uint8_t pattern = blockPatterns[blockId];
const uint8_t count = PATTERN_LUT_BK8[pattern].count;
if (count == 0) {
    A += BK;
    B += BK * N;
    continue;
}

// AFTER:
const uint8_t a_pattern = blockPatterns[blockId];
const uint8_t b_pattern = b_blockPatterns[kBlock * (BN/WN) + warpCol];
const uint8_t a_count = PATTERN_LUT_BK8[a_pattern].count;

if (a_count == 0 || b_pattern == 0) {
    A += BK;
    B += BK * N;
    continue;
}
```

That's it! Just add B-pattern checking alongside A-pattern checking.

**Preprocessing:** Run the same GPU preprocessor on B^T (transposed).

Would you like me to implement this minimal B-sparsity addition?
