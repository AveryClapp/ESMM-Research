# ESMM CUDA Programming
This repository implements high-performance sparse matrix multiplication kernels using CUDA, with a focus on pattern-based sparsity optimization and warp-level acceleration techniques.

## Current Results/Experiments
View current testing results [here](https://docs.google.com/spreadsheets/d/1l7kVnpowxioqy-BX4UVK34Vqc7DbwsiFNjKI8jKklxw/edit?usp=sharing)  

Detailed walkthrough of our current kernel [here](https://github.com/AveryClapp/MMMResearch/blob/main/KernelWalkthrough.md)


## Directory Organization

```
├── driver.cu                    # Main test driver and benchmark harness
├── src/                         # Source code
│   ├── kernels/                 # CUDA kernel implementations
│   └── preprocessors/           # Matrix preprocessing kernels
├── include/                     # Header files (.cuh)
├── tests/                       # Test programs
├── scripts/                     # Python utilities and code generation
├── build/                       # Compiled binaries (gitignored)
├── profiles/                    # Profiling outputs (gitignored)
├── docs/                        # Documentation and visualizations
├── old_kernels/                 # Progressive kernel development hierarchy
├── tuning/                      # Autotuning scripts
└── tuner_results/               # Autotuning results
```

## Core Components

### **Driver System** (`driver.cu`)
- **Purpose**: Benchmark orchestration and performance measurement
- **Features**: Command-line argument parsing, multiple kernel selection, timing infrastructure
- **Matrix Sizes**: Configurable dimensions (default: 1024×1024×1024)
- **Sparsity Patterns**: 8-bit pattern generation

### **Kernel Management** (`runners.cuh`)
- **Architecture**: Unified interface for all kernel variants
- **Execution Control**: Grid/block dimension calculation, synchronization handling
- **Kernel Registry**: Switch-case routing for different optimization levels

### **Utilities** (`utils.cuh`)
- **Error Handling**: CUDA error checking macros with file/line reporting
- **Data Generation**: Pattern-based sparse matrix initialization
- **Performance Macros**: High-resolution timing infrastructure (`SETUP`, `START`, `END`, `RESULTS`)
- **Pattern Processing**: 8-element sparsity pattern handling

## B-Matrix Sparsity Investigation

### Summary
After extensive experimentation with B-matrix sparsity exploitation (kernels 19-25), we found that **B sparsity actually degrades performance** compared to A-only sparsity. All experimental B-sparsity kernels have been removed.

### Why B Sparsity Fails on GPUs

#### The Fundamental Problem: Warp Divergence
```
A Matrix (row-wise sparsity):
  - Processed by warps (32 threads)
  - Pattern check: ALL threads in warp see same pattern
  - Decision: Warp-uniform → skip K-blocks together
  - Result: Zero divergence ✅

B Matrix (column-wise sparsity):
  - Each thread handles different columns (TN=8)
  - Pattern check: Each thread sees DIFFERENT patterns
  - Decision: Per-thread → some threads skip, others don't
  - Result: Maximum divergence ❌
```

### Performance Results (4096×4096, 50% sparsity)

| Kernel | Approach | Time | GFLOPS | vs K17 |
|--------|----------|------|--------|---------|
| **K17** | A-only sparsity | 6.15 ms | **22,351** | 100% ⭐ |
| K18 | A+B flat encoding | 11.37 ms | 12,087 | 54% |
| K20 | A+B conditional load | 14.2 ms | ~9,700 | 43% |
| K22 | A+B inline multiply | 17.8 ms | ~7,700 | 35% |
| K23 | A+B 256-pattern | 12.9 ms | ~10,600 | 48% |
| K24 | A+B offset-based | 79.3 ms | 1,734 | 8% ❌ |
| K25 | A+B hierarchical | 19.0 ms | 7,247 | 32% ❌ |

### Attempted Solutions (All Failed)

1. **Offset-Based B Encoding (K24)**
   - Idea: Pre-compute offsets of non-zeros, use direct indexing
   - Problem: 64 template instantiations (8×8 A×B patterns) → massive code bloat
   - Result: 13x slower than A-only

2. **Hierarchical Checking (K25)**
   - Idea: Two-level hierarchy (32-col coarse + 8-col fine) for warp-uniform coarse skips
   - Problem: Pattern "11110000" means few 32-col blocks are entirely zero
   - Result: Added overhead without benefit → 3x slower than A-only

3. **Conditional Loading/Multiplying (K18, K20, K22)**
   - Idea: Predicate individual loads/FMAs based on pattern bits
   - Problem: 16 conditionals per K-block (8 loads + 8 multiplies) → instruction overhead
   - Result: 2x slower than A-only

### Root Cause Analysis

The overhead of B sparsity checking **exceeds the savings from skipped work**:

```
Cost of B sparsity per K-block:
  - 8 bit-checks for loading
  - 8 bit-checks for multiplication
  - Register pressure from pattern storage
  - Warp divergence penalty (50% efficiency at 50% sparsity)

Savings from 50% B sparsity:
  - Skip 4 out of 8 loads
  - Skip 4 out of 8 FMAs

Result: Overhead > Savings ❌
```

### Conclusion

**B-matrix sparsity cannot be profitably exploited on current GPU architectures** without fundamentally changing the thread-to-output mapping, which would break memory coalescing and reduce A-sparsity benefits.

**Best approach**: Kernel 17 (A-only sparsity) remains optimal. 

