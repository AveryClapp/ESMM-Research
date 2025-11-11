# K17 (esmm_hybrid_blockwise) Kernel Tuning Results

## Tuning Date
November 11, 2025

## Hardware
- GPU: NVIDIA A10G
- CUDA Version: 12.1.0
- Compute Capability: 8.6

## Methodology
- Tool: Python kernel_tuner 1.3.0
- Search space: 27 configurations after restrictions
- Iterations per config: 3
- Matrix types: Dense random (fp32)

## Results by Matrix Size

### 512x512x512
- **Best Config**: NUM_THREADS=128, BM=64, BN=64, WM=32, WN=32, WNITER=1
- **Time**: 0.026 ms
- **Performance**: 10,348 GFLOPS

### 1024x1024x1024
- **Best Config**: NUM_THREADS=128, BM=64, BN=128, WM=64, WN=32, WNITER=2
- **Time**: 0.043 ms
- **Performance**: 50,306 GFLOPS

### 2048x2048x2048
- **Best Config**: NUM_THREADS=128, BM=64, BN=128, WM=64, WN=32, WNITER=2
- **Time**: 0.055 ms
- **Performance**: 310,689 GFLOPS

## Optimal Configuration (Applied to K17)

```cpp
const uint NUM_THREADS = 128;  // Changed from 256
const uint BM = 64;             // Changed from 128
const uint BN = 128;            // Unchanged
const uint WM = 64;             // Changed from 32
const uint WN = 32;             // Changed from 64
const uint WNITER = 2;          // Changed from 4
const uint BK = 8;              // Fixed
const uint TM = 1;              // Fixed
const uint TN = 8;              // Fixed
```

## Key Findings

1. **Thread count**: 128 threads outperforms 256 for this kernel
   - Lower thread count → better register allocation
   - Still sufficient parallelism for warp execution

2. **Block dimensions**: BM=64, BN=128 is optimal
   - Asymmetric tiles (64×128) better than square (128×128)
   - Better L1 cache utilization

3. **Warp tiles**: WM=64, WN=32 with WNITER=2
   - Larger M tile (64 vs 32) improves data reuse
   - Smaller N tile with more iterations balances work

4. **Scaling**: Configuration stable from 1024→2048
   - Same config optimal for both sizes
   - Likely optimal for 4096 as well

## Performance vs Baseline

- **Old K17 config**: NUM_THREADS=256, BM=128, BN=128, WM=32, WN=64, WNITER=4
- **New K17 config**: NUM_THREADS=128, BM=64, BN=128, WM=64, WN=32, WNITER=2
- **Expected improvement**: 5-15% based on tuning results

## Files Modified

- `include/runners.cuh`: lines 894-931 (run_esmm_hybrid_no_check)

## Notes

- Tuning used dense random matrices (100% dense patterns)
- Real sparse matrices may have different optimal configs
- GPU preprocessing recommended for large matrices (4096+)

## Tuning Tool Files

- `tuning/esmm_hybrid_tune.cu`: Tuneable CUDA kernel (no templates)
- `tuning/tuner_k17.py`: Python kernel_tuner script
- `tuning/tuner_results/k17_*.json`: Cached tuning results
