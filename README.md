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

## Next Steps

### Improving Hybrid Kernel (K17)
After consolidating to our two best kernels, there are several low-hanging optimizations for the block-wise uniform approach:

- **Add kernel timing visibility**: Currently only preprocessing time is displayed, making it hard to see actual compute performance. Need to add timing around the kernel loop and report GFLOPS.

- **Profile offset reconstruction overhead**: The offset array reconstruction from bitmask happens every K-block (`for (int i = 0; i < BK; i++) if (pattern & (1 << i))`). Could potentially use `__popc()` for count and a lookup table for offsets since there are only 256 possible patterns.

- **Optimize pattern broadcast**: All 32 threads load the same pattern byte. While cache handles this well, using `__shfl_sync` to broadcast from lane 0 would reduce memory traffic.

- **Consider preprocessing caching**: For research workflows with repeated runs on the same matrix, preprocessing runs every time. Could cache results keyed by matrix pointer.

- **LUT-based offset lookup**: Replace runtime reconstruction with a 256-entry lookup table (1.25KB total). Each entry stores count + offsets, eliminating the bit-checking loop entirely.

