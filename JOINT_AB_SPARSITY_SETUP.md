# Joint A+B Sparsity Kernel - Complete Setup

## Overview
Successfully implemented and tested joint A+B sparsity optimization for ESMM matrix multiplication using B-transpose + intersection approach.

## Files Modified/Created

### 1. **Kernel Implementation**
**File:** `src/kernels/esmm_btranspose.cu`

**Key Changes:**
- Updated kernel signature to accept **both** A and B pattern arrays
- Implemented joint sparse computation with K-index intersection
- Added 2D dispatch system (8×8 templates for A_COUNT × B_COUNT)
- Early exit optimization: skips K-blocks where **either** A or B is fully sparse

**Architecture:**
```cuda
template <... const int A_COUNT, const int B_COUNT>
__device__ void compute_joint_sparse_block(
    const uint8_t* a_offsets,
    const uint8_t* b_offsets,
    ...) {

    // Iterate over A offsets
    for (int a_idx = 0; a_idx < A_COUNT; ++a_idx) {
        const uint8_t dotIdx = a_offsets[a_idx];

        // Check if B also has this K-index (INTERSECTION!)
        bool b_has_offset = false;
        for (int b_idx = 0; b_idx < B_COUNT; ++b_idx) {
            if (b_offsets[b_idx] == dotIdx) {
                b_has_offset = true;
                break;
            }
        }

        if (!b_has_offset) continue;  // Skip if only in A, not B

        // Only compute FMAs for joint non-zero K-indices
        load_A_and_B();
        compute_FMAs();
    }
}
```

### 2. **Runner Functions**
**File:** `include/runners.cuh`

**Added Two New Runners:**

#### `run_esmm_btranspose_joint()` - Kernel 19 with verification
```cuda
bool run_esmm_btranspose_joint(int rows, int cols, int inners,
                                float *d_A, float *d_B, float *d_C,
                                float *h_C, float *h_C_ref, int runs);
```

#### `run_esmm_btranspose_joint_no_check()` - Kernel 19 performance-only
```cuda
bool run_esmm_btranspose_joint_no_check(int rows, int cols, int inners,
                                         float *d_A, float *d_B, float *d_C,
                                         int runs);
```

**Updated Existing Runners:**
- `run_esmm_btranspose()` (Kernel 18) - Now uses joint A+B sparsity
- `run_esmm_btranspose_no_check()` (Kernel 18) - Updated signature

### 3. **Driver Updates**
**File:** `driver.cu`

**Added Case 19:**
```cpp
case 19: // ESMM Joint A+B Sparsity (B-Transpose + Intersection)
    if (check_results) {
        res = run_esmm_btranspose_joint(...);
    } else {
        res = run_esmm_btranspose_joint_no_check(...);
    }
    break;
```

### 4. **Kernel Name Registry**
**File:** `include/utils.cuh`

**Updated:**
```cpp
case 19: return "ESMM Joint A+B Sparsity (B-Transpose + Intersection)";
```

## Kernel Configuration

**Optimized Parameters (matching K17):**
```cpp
const uint NUM_THREADS = 128;
const uint BN = 128;  // Block tile: N dimension
const uint BM = 64;   // Block tile: M dimension
const uint BK = 8;    // Block tile: K dimension
const uint WN = 32;   // Warp tile: N dimension
const uint WM = 64;   // Warp tile: M dimension
const uint WNITER = 2; // Warp iterations in N
const uint TN = 8;    // Thread tile: N dimension
const uint TM = 1;    // Thread tile: M dimension
```

## Preprocessing Pipeline

**A-Matrix Preprocessing:**
```cpp
BlockPatternMetadata A_meta = analyze_sparsity_pattern_gpu(d_A, rows, inners, WM, BK);
// Creates patterns for WM×BK blocks (64×8)
// Index: globalWarpRow * numKBlocks + kBlock
```

**B-Matrix Preprocessing:**
```cpp
BTPatternMetadata B_meta = preprocess_b_transpose(d_B, inners, cols, WN, BK);
// Creates patterns for WN×BK blocks (32×8)
// Column-wise analysis (no actual transpose needed!)
// Index: globalColBlock * numKBlocks + kBlock
```

## Testing & Verification

### ✅ Compilation Status
```bash
make clean && make
# Build complete: ./exec_dev
```

### ✅ Correctness Verification
```bash
./exec_dev 18 1 --check-results
# Kernel 18 (ESMM Combined A+B Sparsity): PASSED

./exec_dev 19 1 --check-results
# Kernel 19 (ESMM Joint A+B Sparsity (B-Transpose + Intersection)): PASSED
```

### ⚙️ Performance Testing (4096×4096×4096)

**At 50% Sparsity:**
```
Kernel 16 (A-only):  5.657 ms | 24295 GFLOPS
Kernel 18 (Joint):   6.759 ms | 20333 GFLOPS
Kernel 19 (Joint):   6.641 ms | 20696 GFLOPS
```

**At 87.5% Sparsity:**
```
Kernel 16 (A-only):  5.693 ms | 24143 GFLOPS
Kernel 18 (Joint):   6.789 ms | 20244 GFLOPS
Kernel 19 (Joint):   6.677 ms | 20584 GFLOPS
```

## Usage Examples

### Run with Verification
```bash
# Single kernel
./exec_dev 19 10 --check-results

# Compare multiple kernels
./exec_dev 16,18,19 10 --check-results
```

### Performance-Only Mode
```bash
# No verification, just timing
./exec_dev 19 100 --no-check

# Custom sparsity level
./exec_dev 19 100 --no-check --sparsity 87.5
```

### Verbose Output
```bash
./exec_dev 19 10 --verbose --check-results
```

## Expected Performance (Theoretical)

**At 50% A-sparsity × 50% B-sparsity:**
```
Joint density: 0.5 × 0.5 = 25%
A-only time:   5.7 ms (50% density)
Expected joint: 5.7 × 0.25/0.5 = 2.85 ms
Theoretical speedup: 2× over A-only
```

**At 87.5% sparsity:**
```
Joint density: 0.125 × 0.125 = 1.56%
A-only time:   5.7 ms (12.5% density)
Expected joint: 5.7 × 0.0156/0.125 = 0.71 ms
Theoretical speedup: 8× over A-only
```

## Next Steps for Optimization

### Current Performance Gap
The joint A+B kernels are currently **slower** than A-only (~20 GFLOPS vs 24 GFLOPS). This suggests optimization opportunities:

1. **Verify B-Sparsity Generation**
   - Check if B matrices are actually being generated with K-dimension sparsity
   - Confirm pattern detection is working correctly
   - Add debugging to count actual sparse blocks

2. **Optimize Intersection Logic**
   - Current nested loop for intersection check might have overhead
   - Consider precomputing joint patterns offline (bitwise AND)
   - Profile to measure intersection cost

3. **Analyze Memory Access Patterns**
   - Check if loading both A and B patterns causes cache pressure
   - Verify coalescing is maintained for both matrices

4. **Tune Thread Configuration**
   - Current config (TM=1, TN=8) is from K17
   - May need different optimal config for joint sparsity

5. **Add Pattern Statistics**
   - Print A-pattern and B-pattern density distributions
   - Measure actual joint density vs theoretical
   - Count early exits vs intersection skips

## Debug Commands

### Check Pattern Generation
```bash
# Run with verbose to see preprocessing stats
./exec_dev 19 1 --verbose --no-check

# Check if patterns are being detected
# Look for: "Patterns: X ms (Y blocks, Z KB metadata)"
```

### Profile with NCU
```bash
make profile KERNEL=19
# Analyzes memory throughput, occupancy, warp stalls
```

### Compare Against Baseline
```bash
# K16: A-only sparsity (baseline)
# K17: A-only optimized
# K18: Joint A+B (btranspose base)
# K19: Joint A+B (your implementation)
./exec_dev 16,17,18,19 100 --no-check --sparsity 50
```

## Key Architectural Decisions

### Why B-Transpose?
- Makes B-sparsity checks warp-uniform (all threads need same B^T rows)
- Eliminates branch divergence within warps
- Better than column-wise access for B

### Why 2D Dispatch?
- Fully unrolls loops at compile time (zero branch overhead)
- 64 specialized kernels (8 A-counts × 8 B-counts)
- Better than runtime loops for small counts (1-8)

### Why Intersection (Not Union)?
- Union would require computing A×0 or 0×B (wasted FMAs)
- Intersection skips K-indices where either matrix is zero
- Multiplicative sparsity benefit: 50%×50% = 25% density

## Files Summary

**Modified:**
- `src/kernels/esmm_btranspose.cu` - Joint sparsity kernel
- `include/runners.cuh` - Added/updated runners
- `driver.cu` - Added case 19
- `include/utils.cuh` - Updated kernel name

**Included Preprocessors:**
- `src/preprocessors/a_preprocessor_hybrid.cu` - A-pattern analysis
- `src/preprocessors/b_transpose_preprocessor.cu` - B-pattern analysis

**Build System:**
- `Makefile` - No changes needed (auto-detects)

## Success Criteria

✅ **Compilation:** PASS
✅ **Correctness (K18):** PASS
✅ **Correctness (K19):** PASS
⚠️ **Performance:** Needs optimization (currently slower than A-only)

## Conclusion

The complete skeleton is set up and verified for correctness. The joint A+B sparsity kernel successfully:
- Compiles without errors
- Passes correctness verification
- Integrates with existing driver infrastructure
- Supports all command-line options

**Next priority:** Debug why joint sparsity isn't providing expected speedup (see "Next Steps for Optimization" above).
