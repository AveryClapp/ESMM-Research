# Blockwise Sparsity Generation

## Overview

Blockwise generation creates sparse matrices where sparsity occurs at **tile granularity** rather than element-wise or column-wise. This matches real-world structured sparsity patterns (e.g., from pruning, quantization) better than random or repeating patterns.

## Key Concept

**Block granularity = BK rows/cols are treated as a unit**

- Each BK×WM (for A) or BK×WN (for B) tile is either:
  - **Fully dense** (all values non-zero)
  - **Fully zero** (entire tile is zeros)
- Within each tile, ALL threads in a warp see the SAME sparsity pattern
- Zero branch divergence within warps

## A-Matrix Blockwise Generation

### Function: `randomize_matrix_A_blocklevel<BK=8, WM=64>`

```
Matrix A (M × K):
┌─────────────────────────────────────┐
│ Block(0,0) │ Block(0,1) │ Block(0,2)│  ← Each block is WM=64 rows
│ 64 rows    │ 64 rows    │ 64 rows   │    × BK=8 cols
│            │            │           │
├────────────┼────────────┼───────────┤
│ Block(1,0) │ Block(1,1) │ Block(1,2)│
│ 64 rows    │ 64 rows    │ 64 rows   │
│            │            │           │
└─────────────────────────────────────┘
    8 cols      8 cols      8 cols
   (BK=8)      (BK=8)      (BK=8)
```

### Algorithm (utils.cuh:442-479)

```cpp
for each M-block (64 rows):
    for each K-block (8 cols):
        // Step 1: Generate random 8-bit pattern for this tile
        tile_pattern = random_8bit_pattern(sparsity_percent)

        // Step 2: Apply SAME pattern to ALL 64 rows in this tile
        for row in [0, 64):
            for k in [0, 8):
                if (tile_pattern & (1 << k)):  // Bit k is set
                    A[row, k] = random_nonzero_value()
                else:
                    A[row, k] = 0.0
```

**Key Property:** All 64 rows in a WM-block share the same K-pattern.

### Example (50% sparsity)

```
Block(0,0) pattern: 11110000 (bits 0-3 dense, 4-7 zero)
Applied to all 64 rows:

Row 0:  [x x x x 0 0 0 0]
Row 1:  [x x x x 0 0 0 0]
Row 2:  [x x x x 0 0 0 0]
...
Row 63: [x x x x 0 0 0 0]

Block(1,0) pattern: 10101010 (alternating, different from Block(0,0))
Row 64:  [x 0 x 0 x 0 x 0]
Row 65:  [x 0 x 0 x 0 x 0]
...
```

## B-Matrix Blockwise Generation (CRITICAL DIFFERENCE)

### Function: `randomize_matrix_B_blocklevel_fixed<BK=8, WN=32>`

```
Matrix B (K × N):
┌─────────────────────────────────────┐
│ Block(0,0)              │ Block(0,1)│
│ 8 rows × 32 cols        │ 8×32      │
├─────────────────────────┼───────────┤
│ Block(1,0)              │ Block(1,1)│
│ 8 rows × 32 cols        │ 8×32      │
└─────────────────────────────────────┘
```

### Algorithm (utils.cuh:482-523)

**CRITICAL:** K-patterns must be **consistent across all N-blocks** for joint A+B sparsity to work correctly!

```cpp
// Step 1: Pre-generate ONE pattern per K-block (line 492-500)
for each K-block:
    k_patterns[kBlock] = random_8bit_pattern(sparsity_percent)

// Step 2: Apply SAME K-pattern to ALL N-blocks (line 502-523)
for each N-block (32 cols):
    for each K-block (8 rows):
        tile_pattern = k_patterns[kBlock]  // ← SAME for all N-blocks!

        for col in [0, 32):
            for k in [0, 8):
                if (tile_pattern & (1 << k)):
                    B[k, col] = random_nonzero_value()
                else:
                    B[k, col] = 0.0
```

### Why K-Consistency is Critical

The B-matrix preprocessor (used by K17, K22-K28) computes:

```cuda
// From src/preprocessors/ab_preprocessor.cu
for each K-block:
    for k in [0, 8):
        bool has_any_nonzero = false
        for col in [0, N):  // Check ALL columns
            if B[k, col] != 0:
                has_any_nonzero = true
        pattern[k] = has_any_nonzero ? 1 : 0
```

**If K-patterns vary across N-blocks:**
```
K-block 0, N-block 0: pattern = 11110000
K-block 0, N-block 1: pattern = 10101010

Preprocessor sees: rows 0,1,2,3,5,7 have non-zeros
Result: pattern = 10111111 (WRONG! Doesn't match either block)
```

**With K-consistent patterns:**
```
K-block 0, ALL N-blocks: pattern = 11110000

Preprocessor sees: rows 0,1,2,3 have non-zeros (uniform across all N)
Result: pattern = 11110000 (CORRECT!)
```

## Visual Comparison: Pattern vs Blockwise

### Pattern-Based (Column-wise)
```
Pattern: 11110000 (repeats every 8 columns)

Column: 0 1 2 3 4 5 6 7 | 8 9 ...
Row 0:  x x x x 0 0 0 0 | x x ...
Row 1:  x x x x 0 0 0 0 | x x ...
Row 2:  x x x x 0 0 0 0 | x x ...
...
```
- Same pattern repeated for EVERY column
- Predictable but unrealistic
- Good for initial testing

### Blockwise (Tile-wise)
```
Block(0,0): 11110000    Block(0,1): 10101010
Block(1,0): 11000000    Block(1,1): 11111111

Column: 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
Row 0:  x x x x 0 0 0 0 | x 0 x  0  x  0  x  0   ← Block(0,0) | Block(0,1)
...
Row 63: x x x x 0 0 0 0 | x 0 x  0  x  0  x  0
Row 64: x x 0 0 0 0 0 0 | x x x  x  x  x  x  x   ← Block(1,0) | Block(1,1)
...
```
- Each tile has independent random pattern
- More realistic (models pruned networks)
- Better for evaluating real-world performance

## Sparsity Percentage Calculation

**Input pattern:** 8-bit binary string (e.g., "11110000")

```python
density_percent = (count_of_1s / 8.0) * 100.0
sparsity_percent = 100.0 - density_percent

Examples:
  11111111 → 100% density, 0% sparsity
  11110000 → 50% density, 50% sparsity
  11000000 → 25% density, 75% sparsity
  10000000 → 12.5% density, 87.5% sparsity
```

**Random generation (lines 454-458, 494-498):**
```cpp
for (int bit = 0; bit < 8; bit++) {
    float rand_val = (float)rand() / RAND_MAX;  // [0.0, 1.0]
    if (rand_val >= sparsity_threshold) {
        tile_pattern |= (1 << bit);  // Set bit to 1 (dense)
    }
    // else: bit remains 0 (sparse)
}
```

If `sparsity_percent = 50%`, then `sparsity_threshold = 0.5`:
- ~50% of rand values will be ≥ 0.5 → bits set to 1 (dense)
- ~50% of rand values will be < 0.5 → bits remain 0 (sparse)

## Usage Examples

### Example 1: Unified Sparsity
```bash
# 50% sparsity for both A and B, blockwise generation
./exec_dev 24 1 --size 4096 --blockwise --pattern 11110000 -v

Output:
  A-matrix: 50% sparsity (50% density)
  B-matrix: 50% sparsity (50% density)
```

### Example 2: Different A/B Sparsity
```bash
# 25% A sparsity, 75% B sparsity, blockwise generation
./exec_dev 24 1 --size 4096 -b --pattern-a 11100000 --pattern-b 11000000 -v

Output:
  A-matrix: 37.5% sparsity (62.5% density) ← 5/8 bits set
  B-matrix: 75% sparsity (25% density)      ← 2/8 bits set
```

### Example 3: Benchmark All Combinations
```bash
# Run all 8×8 = 64 combinations of A/B sparsity
./scripts/benchmark_all_ab_combinations.sh 25 4096 --cold-start
```

## Granularity Variants

The codebase has multiple blockwise generators for different granularities:

| Function | Granularity | Use Case | Kernels |
|----------|-------------|----------|---------|
| `randomize_matrix_A_blocklevel<8,64>` | 64 rows × 8 cols | Coarse A-sparsity | K22-K24 |
| `randomize_matrix_A<8,8>` | 8 rows × 8 cols | Fine A-sparsity | K21, K28 |
| `randomize_matrix_B_blocklevel_fixed<8,32>` | 8 rows × 32 cols | B-sparsity (K-consistent) | K17, K22-K28 |

**Choosing granularity:**
- Finer granularity (8×8) → More flexibility, harder to exploit
- Coarser granularity (64×8) → Less flexibility, easier to skip entire warps

## Performance Implications

### Memory Access Patterns
**Blockwise advantages:**
1. **Coalescing:** Entire tiles are dense → better coalescing within blocks
2. **Caching:** Dense tiles have better L1/L2 hit rates
3. **Prefetching:** Hardware can predict access patterns

**Blockwise challenges:**
1. **Load balancing:** Some warps may have all-zero tiles (wasted work)
2. **Pattern overhead:** Need to store/load tile patterns

### Branch Divergence
**Within a warp (32 threads):**
- If all threads in warp access the SAME tile → zero divergence
- Pattern bits uniform across warp → branches are free

**Across warps:**
- Different warps can have different tile patterns → no divergence cost
- Branches are only expensive when threads WITHIN a warp diverge

## Verification Considerations

**Why blockwise verification can fail:**

1. **K-inconsistency bug (fixed):** If B-matrix K-patterns vary across N-blocks, preprocessor generates incorrect patterns → computation uses wrong sparsity

2. **Granularity mismatch:** Using WM=64 generator with WM=32 kernel means kernel expects finer patterns than matrix provides

3. **Floating-point accumulation:** Random tile patterns change computation order → slight FP differences (acceptable with tolerance=1e-3)

**Correct usage:**
```bash
# Match generator granularity to kernel granularity
./exec_dev 28 1 -b -p 11110000  # K28 uses 8-row granularity
./exec_dev 24 1 -b -p 11110000  # K24 uses 64-row granularity
```

## Implementation Files

- **Generator functions:** `include/utils.cuh:439-652`
- **Preprocessors:** `src/preprocessors/ab_preprocessor.cu`
- **Driver integration:** `driver.cu:380-422`
- **Kernel implementations:** `src/kernels/esmm_ab_*.cu`

## Summary

**Blockwise generation provides:**
- ✅ Realistic structured sparsity (models pruned networks)
- ✅ Zero branch divergence within warps (all threads see same pattern)
- ✅ Better memory access patterns (tile-level coalescing)
- ✅ Flexible sparsity levels (via random pattern generation)
- ⚠️ Requires K-consistent patterns for B-matrix (critical for correctness)
- ⚠️ Must match generator granularity to kernel granularity
