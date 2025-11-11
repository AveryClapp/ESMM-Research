# Preprocessor Tuning & Compilation Speed Optimization

## Issue 1: Preprocessor Parameter Mismatch

### Current Problem
**Critical:** Preprocessor uses hardcoded WM=32, but tuned K17 kernel uses WM=64!

```cpp
// src/preprocessors/a_preprocessor_hybrid.cu:118
preprocess_blockwise_patterns<8, 32, 256>  // WM=32 hardcoded!

// include/runners.cuh:896-901 (NEW TUNED CONFIG)
const uint WM = 64;  // Kernel expects WM=64!
```

**Impact:**
- Preprocessor generates patterns for 32-row blocks
- Kernel expects patterns for 64-row blocks
- **Mismatch causes incorrect results or crashes!**

### Fix: Update Preprocessor Call

```cpp
// In analyze_sparsity_pattern_gpu() - line 118
preprocess_blockwise_patterns<8, 64, 256>  // Match WM=64
    <<<gridDim, blockDim>>>(M, K, d_A, meta.d_blockPatterns);
```

### Preprocessor Tuning Opportunities

| Parameter | Current | Tuning Options | Impact |
|-----------|---------|----------------|---------|
| WM | 32 | **64** (to match kernel) | ✅ Critical fix |
| NUM_THREADS | 256 | 128, 256, 512 | Memory coalescing |
| Batch size | 4 K-blocks | 2, 4, 8 | Shared memory utilization |

**Tuning NUM_THREADS:**
```cpp
// Test different thread counts
preprocess_blockwise_patterns<8, 64, 128>  // Fewer threads
preprocess_blockwise_patterns<8, 64, 256>  // Baseline
preprocess_blockwise_patterns<8, 64, 512>  // More threads
```

**Trade-offs:**
- **128 threads**: Less parallelism, but better occupancy for large WM=64
- **256 threads**: Balanced (current)
- **512 threads**: More parallelism, but may reduce occupancy

---

## Issue 2: Compilation Speed

### Current Compilation Time Issues

1. **Template Explosion**:
   - K17 instantiates 9 templates with different SIZE parameters (switch cases 1-8)
   - Each case instantiates `compute_sparse_block` with full parameter list
   - Total: 9 template instantiations × complex inner loops

2. **Optimization Level**:
   - `-O3 --use_fast_math` is aggressive
   - Loop unrolling and inlining generate massive code

3. **No Explicit Template Instantiation**:
   - Templates instantiated on-demand in header files
   - Recompiled in every translation unit

### Optimization Strategies

#### 1. **Explicit Template Instantiation** (Fastest compilation win)

Create `src/kernels/esmm_hybrid_instances.cu`:

```cpp
#include "../../src/kernels/esmm_hybrid.cu"

// Explicitly instantiate only the configurations we use
template __global__ void esmm_hybrid_blockwise<
    64,   // BM
    128,  // BN
    8,    // BK
    64,   // WM
    32,   // WN
    2,    // WNITER
    1,    // TM
    8,    // TN
    128   // NUM_THREADS
>(int, int, int, float*, float*, float*, const uint8_t*, const int);

// Old config (for testing)
template __global__ void esmm_hybrid_blockwise<
    128, 128, 8, 32, 64, 4, 1, 8, 256
>(int, int, int, float*, float*, float*, const uint8_t*, const int);
```

In `runners.cuh`:
```cpp
// Declare extern template to prevent re-instantiation
extern template __global__ void esmm_hybrid_blockwise<64, 128, 8, 64, 32, 2, 1, 8, 128>
    (int, int, int, float*, float*, float*, const uint8_t*, const int);
```

**Expected speedup: 3-5×** (compile only once, not per TU)

#### 2. **Reduce Template Parameters**

Current: 9 parameters + 8 SIZE specializations = 72 potential combinations

Optimization: Use `#define` for fixed parameters:
```cpp
#define K17_BK 8
#define K17_TM 1
#define K17_TN 8

// Now only 6 parameters (reduce compilation load)
template <const int BM, const int BN, const int WM, const int WN,
          const int WNITER, const int NUM_THREADS>
__global__ void esmm_hybrid_blockwise(...) {
    constexpr int BK = K17_BK;
    constexpr int TM = K17_TM;
    constexpr int TN = K17_TN;
    // ...
}
```

**Expected speedup: 1.5-2×**

#### 3. **Development vs Production Builds**

**Makefile/CMake changes:**

```makefile
# Development (fast compile, good-enough perf)
NVCC_FLAGS_DEV = -O2 -lineinfo -std=c++17

# Production (slow compile, max perf)
NVCC_FLAGS_PROD = -O3 --use_fast_math -std=c++17

# Current target
dev:
	nvcc $(NVCC_FLAGS_DEV) -o driver driver.cu

prod:
	nvcc $(NVCC_FLAGS_PROD) -o driver driver.cu
```

**Expected speedup: 2-3× for development**

#### 4. **Parallel Compilation**

```bash
# Use all CPU cores
make -j$(nproc)

# Or with explicit parallelism
nvcc -t=4 ...  # 4 parallel threads
```

**Expected speedup: 2-4× (linear with cores)**

#### 5. **ccache Integration**

```bash
# Install ccache
sudo yum install ccache

# Use with nvcc
export CCACHE_DIR=/tmp/ccache
nvcc -ccbin='ccache g++' ...
```

**Expected speedup: 10-100× on recompilation**

#### 6. **Precompiled Headers**

Create `include/pch.cuh`:
```cpp
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
// All common includes
```

Compile once:
```bash
nvcc -x cu-header include/pch.cuh -o include/pch.cuh.gch
```

Use:
```cpp
#include "pch.cuh"  // Uses precompiled version
```

**Expected speedup: 1.5-2×**

#### 7. **Incremental Compilation**

Separate kernels into individual .cu files:
```
src/kernels/
  ├── esmm_k17.cu         # Only K17
  ├── esmm_k18.cu         # Only K18
  └── esmm_k19.cu         # Only K19
```

Then compile individually and link:
```bash
nvcc -c src/kernels/esmm_k17.cu -o k17.o
nvcc -c src/kernels/esmm_k18.cu -o k18.o
nvcc k17.o k18.o driver.cu -o driver  # Fast linking
```

**Expected speedup: 5-10× when modifying single kernel**

---

## Recommended Action Plan

### Immediate (Critical Fix)
1. ✅ Update preprocessor to use WM=64 (matches new kernel config)

### Short-term (Quick Compilation Wins)
2. Add explicit template instantiation for K17
3. Create dev/prod build targets with -O2/-O3

### Medium-term (Comprehensive Speed-up)
4. Separate kernels into individual .cu files
5. Add ccache support
6. Enable parallel compilation (-j)

### Long-term (Architecture Improvement)
7. Refactor to reduce template parameters
8. Consider JIT compilation for parameter exploration
9. Precompiled headers for common includes

---

## Estimated Total Speedup

| Change | Speedup | Cumulative |
|--------|---------|------------|
| Explicit instantiation | 3× | 3× |
| -O2 for dev builds | 2× | 6× |
| Parallel compilation | 2× | 12× |
| ccache (recompiles) | 10× | 120× |
| Separate .cu files | 5× | 600× (incremental) |

**Bottom line:** 
- First compile: 3-12× faster
- Incremental recompiles: 100-600× faster with ccache + separation
