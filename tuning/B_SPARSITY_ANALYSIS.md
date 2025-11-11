# B-Sparsity Analysis for Tuned K17 Configuration

## Current K17 Architecture (A-sparsity only)

**Memory Layout:**
- A tiles: BK×BM = 8×64 (new config)
- B tiles: BK×BN = 8×128  
- Block patterns: per 8×32 A-block (WM=32 rows)

**Computation Granularity:**
- Each warp processes: BK×WM subtile of A (8×64) × BK×WN subtile of B (8×32)
- Results in: WM×WN = 64×32 output block per warp

## Impact of New Configuration on B-Sparsity

### Key Parameter Changes
| Parameter | Old | New | Impact on B-sparsity |
|-----------|-----|-----|----------------------|
| WN | 64 | **32** | ✅ Smaller column groups → finer granularity |
| WNITER | 4 | **2** | ✅ Fewer iterations → less loop overhead |
| NUM_THREADS | 256 | **128** | ⚠️ Less parallelism, but better registers |
| WM | 32 | **64** | ➖ Affects A dimension, not B |

### Novel B-Sparsity Opportunities

#### 1. **Warp-Level Column Skipping (WN=32 granularity)**

**Concept:** Check if entire 8×32 B-subtiles are zero before computation

```cpp
// Pseudocode for warp-level B-sparsity
const uint8_t b_pattern = B_COL_PATTERNS[kBlock * (BN/WN) + warpCol];
if (b_pattern == 0) {
    // Skip this warp's 8×32 B-subtile entirely
    // No computation, no output writes
    continue;
}
```

**Benefits:**
- **Granularity**: 8×32 = 256 elements per pattern (1 byte metadata)
- **Memory**: (K/8) × (N/32) bytes = K×N/256 overhead
- **Divergence**: Warp-uniform (no divergence within warp)
- **Savings**: Skip FMAs + skip output writes for zero columns

**Why WN=32 helps:**
- Smaller column groups → more likely to be fully sparse
- With WN=64, need 8×64=512 elements all zero (less likely)
- With WN=32, only need 8×32=256 elements zero (2× more likely)

#### 2. **Thread-Level Column Patterns (TN=8 granularity)**

**Concept:** Each thread handles 8 consecutive B columns (TN=8). Check per-thread patterns.

```cpp
// Current: threads load 8 B elements regardless of sparsity
regN[wSubColIdx * TN + 0] = Bs[...];  // Always loads
regN[wSubColIdx * TN + 1] = Bs[...];
// ... all 8 loads

// B-sparse version: check which of 8 columns are non-zero
const uint8_t thread_b_pattern = ...; // 1 bit per column
#pragma unroll
for (int i = 0; i < 8; i++) {
    if (thread_b_pattern & (1 << i)) {
        regN[wSubColIdx * TN + i] = Bs[...];
    }
}
```

**Benefits:**
- **Finest granularity**: 8×8 = 64 element blocks
- **Memory**: K/8 × N/8 = K×N/64 bytes
- **Trade-off**: More overhead, but detects sparser patterns

**Why new config helps:**
- WNITER=2 (was 4) → fewer threads loading redundantly
- Better register availability (128 threads vs 256) for pattern bookkeeping

#### 3. **Hybrid A+B Sparsity**

**Concept:** Combine A-block patterns (current) with B-column patterns (new)

```cpp
// A pattern: which of 8 K-positions are non-zero
const uint8_t a_pattern = A_BLOCK_PATTERNS[...];  // Current K17
const uint8_t a_count = popcount(a_pattern);

// B pattern: which of 32 columns are non-zero  
const uint8_t b_pattern = B_COL_PATTERNS[...];    // NEW

// Joint sparsity check
if (a_count == 0 || b_pattern == 0) {
    continue;  // Skip this BK×WM×WN block entirely
}

// Sparse×Sparse: only compute non-zero combinations
for (int a_idx = 0; a_idx < a_count; a_idx++) {
    const uint8_t k = a_offsets[a_idx];
    // Load A values
    
    // Only load B values for non-zero columns
    for (int col = 0; col < 32; col++) {
        if (b_pattern & (1 << (col / 8))) {  // Coarse check
            // Load and compute
        }
    }
}
```

**Benefits:**
- **Multiplicative savings**: If A is 50% sparse and B is 50% sparse → 75% computation saved
- **Memory**: A patterns + B patterns (2× overhead but still small)

## Cost-Benefit Analysis

### WN=32 Column-Level B-Sparsity

**Costs:**
- 1 byte per (8×32) B-block
- Memory: K/8 × N/32 = K×N/256 bytes
- For 4096×4096: 4096²/256 = 64 KB ✅ (cheap!)
- 1 additional load + branch per warp per K-block

**Benefits (at 50% B-sparsity):**
- Skip 50% of warp computations
- Skip 50% of output writes
- Reduced memory traffic for C writes

**Break-even:**
- Cost: 1 pattern load + 1 branch = ~10 cycles
- Benefit: Skip 8×32×64 = 16,384 FMAs = ~512 cycles
- **Ratio: 50:1 benefit** ✅✅✅

### Thread-Level (TN=8) B-Sparsity

**Costs:**
- 1 byte per (8×8) B-block  
- Memory: K/8 × N/8 = K×N/64 bytes
- For 4096×4096: 256 KB (still reasonable)
- Higher branch overhead per thread

**Benefits (at 50% B-sparsity):**
- Finer-grained skipping
- Better for structured sparsity (e.g., sparse attention)

**Break-even:**
- More marginal due to higher overhead
- Best for >70% sparsity or structured patterns

## Recommended Implementation Priority

### Phase 1: Warp-Level B-Sparsity (WN=32 blocks)
- **Easiest to implement** (similar to current A-block patterns)
- **Best ROI** (50:1 benefit-cost)
- **Warp-uniform** (no divergence)
- Natural fit with WN=32 config

### Phase 2: Joint A+B Hybrid Sparsity
- Combine existing A-patterns with new B-patterns
- **Multiplicative savings** for doubly-sparse matrices
- Requires careful pattern intersection logic

### Phase 3: Fine-Grained (TN=8) B-Sparsity
- For extreme sparsity (>80%)
- Or structured patterns (e.g., block-sparse attention)

## Conclusion

**Yes!** The new tuned configuration (WN=32, WNITER=2) creates excellent opportunities for B-sparsity:

1. ✅ **Smaller warp tiles (WN=32)** provide natural granularity for B-column patterns
2. ✅ **Fewer iterations (WNITER=2)** reduce loop overhead for pattern checking
3. ✅ **Lower thread count (128)** provides more registers for pattern metadata
4. ✅ **Strong cost-benefit** (50:1 ratio for warp-level B-sparsity)

The WN=32 setting is particularly well-suited for adding warp-uniform B-column skipping with minimal divergence and excellent performance gains.
