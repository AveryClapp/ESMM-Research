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
 Investigating B sparisty still. The problem is that A is fast because of warp level skipping but B is just thread level skipping which doesnt really do anything. 

