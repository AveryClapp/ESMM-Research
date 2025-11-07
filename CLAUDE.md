# ESMM Research: Emergent Sparsity Matrix Multiplication

## Project Overview

This research project aims to **exploit emergent sparsity in dense matrix multiplication at runtime** on CUDA GPUs. Unlike traditional sparse matrix libraries that require pre-converted sparse formats (CSR, COO, etc.), this work focuses on detecting and exploiting sparsity patterns dynamically during computation of standard dense matrices.

## The Core Insight

Modern neural networks and scientific computing workloads often exhibit **emergent sparsity** - matrices that are stored in dense format but contain significant zero patterns due to:
- Activation functions (ReLU, GELU) producing zeros
- Quantization and pruning creating structured sparsity
- Natural sparsity in certain problem domains

**Key Challenge**: Traditional sparse libraries have high overhead for format conversion. If we can detect and skip zero computation directly on dense matrices, we can:
1. Avoid format conversion overhead
2. Handle dynamically changing sparsity patterns
3. Exploit sparsity in both input matrices (A and B) simultaneously

## Architecture Overview

### Codebase Structure

```
ESMM-Research/
├── src/
│   ├── kernels/                    # CUDA kernel implementations
│   │   ├── esmm.cu                # Baseline ESMM (A-sparsity only)
│   │   ├── esmm_hybrid.cu         # Kernel 17: Warp-uniform A-sparsity
│   │   ├── esmm_hybrid_combined.cu # Kernel 18: A+B sparsity (prototype)
│   │   └── esmm_btranspose.cu     # Kernel 19: B-sparsity via transpose
│   └── preprocessors/              # Pattern analysis kernels
│       ├── a_preprocessor_hybrid.cu      # A-matrix pattern extraction
│       └── b_transpose_preprocessor.cu   # B-matrix pattern extraction
├── include/
│   ├── runners.cuh                # Kernel launch wrappers
│   ├── pattern_lut.cuh           # Pattern lookup tables (256 8-bit patterns)
│   └── metadata.cuh              # Metadata structures for patterns
├── driver.cu                      # Main benchmark harness
└── old_kernels/                   # Historical implementations
```

### Key Concepts

#### 1. Block-Level Sparsity Patterns

Instead of element-level sparsity tracking, we use **block-level patterns**:
- Divide matrix into small blocks (typically 8×32 for BK×WM or BK×WN)
- Encode which of the 8 K-dimension elements are non-zero as an 8-bit pattern
- Use lookup tables to map 8-bit patterns to offset arrays for sparse computation

**Why 8 bits?**
- Small enough for fast lookup (256 patterns)
- Large enough to capture meaningful sparsity
- Fits warp execution model (32 threads can share pattern)

#### 2. Warp-Uniform Execution

Critical for performance: all 32 threads in a warp must execute the same code path.

**Warp-Uniform Pattern Checking**:
- Group threads so they all need the same sparsity pattern
- All threads can skip the same zero blocks together
- Zero divergence when checking patterns
- Use `switch(count)` for compile-time unrolling based on non-zero count

#### 3. Preprocessing vs Runtime

Current approach uses lightweight preprocessing:
- **One-time cost**: Analyze A and/or B to extract patterns
- **Output**: Small metadata (1 byte per block, ~64KB for 4096×4096)
- **Runtime**: Main kernel loads patterns and skips zero blocks

## Kernel Evolution

### Kernel 10-16: A-Sparsity Exploration
Early kernels focused on exploiting sparsity in the A matrix (rows of output).

### Kernel 17: ESMM Hybrid (A-Sparsity, Current Best)
- **22.7 TFLOPS** on 4096×4096 @ 50% sparsity
- Warp-uniform A pattern checking
- Architecture: 32×64 warp tiles, TM=1, TN=8
- Each warp computes same rows → shares row patterns

### Kernel 18: Combined A+B Sparsity (Prototype)
- **12.3 TFLOPS** - slower due to complexity
- Attempts to check both A and B patterns
- Challenge: Hard to maintain warp-uniformity for both simultaneously

### Kernel 19: B-Transpose Approach (B-Sparsity)
- **4.0 TFLOPS** - significantly slower than K17
- Architecture: 64×32 warp tiles, TM=8, TN=1
- Transposes computation to exploit column sparsity in B
- Challenge: TM=8, TN=1 layout performs poorly vs. TM=1, TN=8

## The Central Problem

**How do we exploit sparsity in BOTH A and B matrices simultaneously while maintaining warp-uniform execution?**

### Architectural Constraints

1. **Warp Uniformity Requirement**
   - All threads in a warp must check the same pattern
   - A-sparsity: threads computing same rows → warp-uniform ✓
   - B-sparsity: threads computing same columns → warp-uniform ✓
   - Both simultaneously: threads need same row AND column patterns → conflict ✗

2. **Thread Tile Layout**
   - TM=1, TN=8: Better ILP, better performance, good for A-sparsity
   - TM=8, TN=1: Worse ILP, worse performance, good for B-sparsity
   - Can we find a layout that works for both?

3. **Memory Access Patterns**
   - A: Row-major (M×K), A-sparsity aligns naturally
   - B: Row-major (K×N), column patterns are harder to exploit
   - Transposing B → row patterns, but adds preprocessing cost

## Current State & Key Files

### Pattern Encoding
- `include/pattern_lut.cuh`: Lookup table mapping 8-bit patterns to offset arrays
- Patterns generated offline, compiled into kernel
- Maps pattern → {count, offsets[8]} for runtime use

### Shared Memory Layout
Current kernels use column-major shared memory with padding:
```cuda
__shared__ float As[(BK + PAD) * BM];   // Column-major for coalesced access
__shared__ float Bs[(BK + PAD) * BN];   // With padding to avoid bank conflicts
```

### Pattern Metadata Structure
```cuda
struct PatternMetadata {
    uint8_t* d_blockPatterns;  // Device pointer to pattern array
    int numRowBlocks;          // Number of row blocks
    int numKBlocks;            // Number of K blocks
}
```

## Performance Targets

- **Theoretical Peak**: ~19.5 TFLOPS (FP32 on A100)
- **cuBLAS Dense**: ~19.0 TFLOPS
- **Current Best (K17)**: ~22.7 TFLOPS @ 50% sparsity (effective speedup)
- **Goal**: Exploit both A and B sparsity for even higher effective throughput

## How to Get Started

### Building and Running
```bash
# Compile the driver
nvcc -arch=sm_80 -O3 -o driver driver.cu -lcublas -lcusparse -I.

# Run specific kernel (e.g., 17, 18, 19)
./driver 17 100 -v -n      # Kernel 17, 100 runs, verbose, no correctness check

# Run comparison
./driver 17,18,19 100 -v -n
```

### Understanding Performance
- Driver reports GFLOPS = (2 * M * N * K) / time
- With sparsity, effective GFLOPS is higher (less actual work)
- Compare kernel time, not just GFLOPS

### Key Parameters
```cuda
BM, BN = 128    // Block tile size
BK = 8          // K-dimension tile (matches 8-bit pattern)
WM, WN = 32/64  // Warp tile size
TM, TN = 1/8    // Thread tile size
NUM_THREADS = 256
```

## Open Research Questions

1. **Can we achieve warp-uniform checking for both A and B patterns simultaneously?**
   - Current approaches force a choice between A or B sparsity
   - Need architectural innovation to handle both

2. **Is B-transpose approach viable with better tuning?**
   - K19 is 5.7× slower than K17
   - Is this fundamental or fixable with optimization?

3. **What's the optimal thread tile layout for dual-sparsity?**
   - TM=1, TN=8 works great for A-sparsity
   - TM=8, TN=1 needed for B-sparsity (but performs poorly)
   - Can we use mixed layouts or 2D layouts (TM=4, TN=4)?

4. **Can we reduce preprocessing overhead?**
   - Current: Analyze patterns before kernel launch
   - Alternative: Online pattern detection during first few tiles?

## Development Tips

### Correctness Testing
Always test correctness first:
```bash
./driver 19 1 -v  # Single run with verification
```

### Profiling
Use nsys for detailed profiling:
```bash
nsys profile -o profile ./driver 19 100 -n
nsys stats profile.nsys-rep
```

### Common Pitfalls
1. **Warp divergence**: Use `-lineinfo` and `nsys` to check for divergence
2. **Bank conflicts**: Pad shared memory (add 1 element per row)
3. **Coalescing**: Ensure consecutive threads access consecutive memory
4. **Pattern lookup overhead**: Pre-compute and use compile-time dispatch

## Success Metrics

A successful solution should:
- [ ] Exploit both A and B sparsity simultaneously
- [ ] Maintain warp-uniform execution (zero divergence)
- [ ] Achieve >15 TFLOPS on 4096×4096 @ 50% A+B sparsity
- [ ] Have minimal preprocessing overhead (<1ms)
- [ ] Work with dynamic sparsity patterns

---

**Note**: This is active research. The "best" approach is still being discovered. Feel free to challenge assumptions, try radically different architectures, or explore new ideas. The goal is not to incrementally improve, but to find breakthrough approaches to dual-matrix sparsity exploitation.
