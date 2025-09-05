# ESMM CUDA Programming
This repository implements high-performance sparse matrix multiplication kernels using CUDA, with a focus on pattern-based sparsity optimization and warp-level acceleration techniques.

## Current Results/Experiments
View current testing results [here](https://docs.google.com/spreadsheets/d/1l7kVnpowxioqy-BX4UVK34Vqc7DbwsiFNjKI8jKklxw/edit?usp=sharing)  

Detailed walkthrough of our current kernel [here](https://github.com/AveryClapp/MMMResearch/blob/main/KernelWalkthrough.md)


## Directory Organization

```
├── driver.cu                    # Main test driver and benchmark harness
├── runners.cuh                  # Kernel wrapper functions and execution logic
├── utils.cuh                    # Utility functions, error handling, and matrix generation
├── esmm.cu                      # Main ESMM (Efficient Sparse Matrix Multiplication) kernel
├── esmm_warpskipping.cu         # ESMM with warp-level sparsity optimization
├── old_kernels/                 # Progressive kernel development hierarchy
├── images/                      # Architecture diagrams and visualizations
└── KernelWalkthrough.md        # Detailed implementation explanation
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

### **Progressive Optimization Pipeline**
The `old_kernels/` directory implements a systematic optimization progression:

1. **Baseline** (`basic.cu`): Naive thread-per-element approach
2. **Memory Optimization** (`gmem_coalesce.cu`): Coalesced global memory access
3. **Cache Utilization** (`smem_blocking.cu`): Shared memory blocking for data reuse
4. **Thread-Level Tiling** (`1D_Blocktiling.cu`): Multiple outputs per thread
5. **Advanced Tiling** (`2D_Blocktiling.cu`): 2D register blocking for compute intensity
6. **Vectorization** (`vectorized_blocktiling.cu`): SIMD memory operations
7. **Warp Specialization** (`warptiling.cu`): Warp-level cooperative computation

### **Sparse Kernels**
- **ESMM** (`esmm.cu`): Main sparse kernel with pattern detection
- **Warp Skipping** (`esmm_warpskipping.cu`): Dynamic warp-level sparsity optimization
- **Double Buffering** (`esmm_buffered.cu`): Experimental approach to ping-ponging memory and compute loads

## Research Integration

The codebase incorporates insights from cutting-edge research:
- **PIT Framework**: Permutation Invariant Transformation for dynamic sparsity
- **Micro-tile Optimization**: GPU-efficient sparse computation patterns  
- **Pattern-based Sparsity**: 8-bit pattern recognition and kernel specialization


## TODOs: 
- [ ] [KernelTuner](https://github.com/KernelTuner/kernel_tuner)
- [ ] [Cutlass](https://github.com/NVIDIA/cutlass) inspired double-buffering
- [ ] Padding to avoid bank conflicts (need to check our current relationship with bank conflicts)
- [ ] Revisit Inner Dimension Constraint Results
- [ ] Take a look at the [PIT Paper](https://arxiv.org/pdf/2301.10936)
- [ ] Generate 256 kernels
- [ ] Tuple (Offset, Value) for A-Sparsity??
- [ ] Build Switch Table
