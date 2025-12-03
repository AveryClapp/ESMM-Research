# ESMM: Emergent Sparsity Matrix Multiplication

High-performance CUDA kernels for sparse matrix multiplication, exploiting pattern-based sparsity in neural network activations.

## Key Results

**Best Kernel (K28)**: 8×32 granularity joint A+B sparsity
- Achieves near-linear speedup with combined sparsity
- Zero warp divergence through warp-uniform pattern skipping
- Efficient pattern preprocessing with coalesced memory access

See [KERNELS.md](docs/KERNELS.md) for full progression.

## Quick Start

```bash
# Build
make dev

# Run single kernel
./exec_dev 28 10 --size 4096 --pattern 11110000

# Run benchmark suite
./scripts/benchmark.py --kernel 28 --sizes 2048,4096 --cold-start

# Compare multiple kernels
./exec_dev "17,21,24,28" 5 --blockwise --pattern 11110000
```

## Repository Structure

```
├── driver.cu              # Main entry point, CLI, verification
├── src/
│   ├── kernels/           # 29 kernel implementations (K1-K29)
│   └── preprocessors/     # GPU pattern extraction kernels
├── include/
│   ├── runners.cuh        # Kernel wrapper functions
│   ├── utils.cuh          # Matrix generation, validation, CLI parsing
│   └── pattern_lut.cuh    # Precomputed offset tables
├── old_kernels/           # Early GEMM optimization experiments (K1-9)
├── scripts/
│   └── benchmark.py       # Parallel NCU profiling automation
├── tuning/                # Autotuning scripts and results
└── docs/                  # Kernel walkthroughs and experiment notes
```

## Kernel Progression

### Dense GEMM Baseline (K1-9)
Progressive optimization from naive to warptiled GEMM:
- K1: Naive → K9: 1D Warptiling (~20x improvement)
- Techniques: Memory coalescing, shared memory blocking, vectorization, register tiling

### A-Matrix Sparsity (K10-16)
Pattern-based skipping of zero elements in activation matrix:
- **K16**: Block-wise warp-uniform patterns (best A-only)
- 8×32 granularity, zero divergence, ~2x speedup at 50% sparsity

### B-Matrix Sparsity (K17-18, K21)
Exploiting sparsity in weight matrix:
- **K17**: Warp-granularity (32-col blocks)
- **K18**: TN-granularity (8-col blocks, higher divergence)
- **K21**: Warp-uniform (simplest, best B-only)

### Joint A+B Sparsity (K24, K28-29)
Multiplicative sparsity benefits from combined skipping:
- **K24**: Coarse 64×32 granularity, zero-overhead inner loop
- **K28**: Fine 8×32 granularity, independent sub-tile patterns (best overall)
- **K29**: Medium 32×32 granularity (balanced)

**Finding**: Joint A+B sparsity achieves ~4x speedup at 50%×50% sparsity (vs 2x for A-only or B-only).

## Implementation Highlights

### Pattern Preprocessing
- GPU-accelerated extraction of 8-bit sparsity patterns
- Warp shuffle for efficient OR reduction
- Shared memory transpose for coalesced access
- Separate kernels for different granularities (8-row, 32-row, 64-row)

### Zero-Divergence Skipping
- Warp-uniform pattern checks (all threads agree)
- Pattern indexed by warp-level tiles (not per-thread)
- Direct bit testing with continue (no offset arrays)
- Template dispatch for fully unrolled inner loops (K13-K18)

### Memory Optimization
- BK=8 for optimal L1 cache utilization
- Double buffering with prefetch (K12)
- Bank conflict avoidance via padding
- Vectorized loads (float4) where possible

## Building

```bash
# Development build (with assertions)
make dev

# Release build (optimized)
make release

# Clean
make clean
```

Requires CUDA Toolkit 12.0+ and compute capability 7.0+ (Volta or newer).

## Benchmarking

The `scripts/benchmark.py` tool automates NCU profiling with proper cold-start handling:

```bash
# Single kernel, default sparsity levels
./scripts/benchmark.py --kernel 28

# Multiple kernels, custom sizes
./scripts/benchmark.py -k 17,21,24,28 --sizes 2048,4096

# Cold-start mode (matches manual profiling)
./scripts/benchmark.py -k 28 --cold-start --parallel 1

# Custom sparsity patterns
./scripts/benchmark.py -k 28 --sparsity 11110000,11000000,10000000
```

Default patterns: 100% (11111111), 50% (11110000), 25% (11000000), 12.5% (10000000)

See `python scripts/benchmark.py --help` for all options.

## Sparsity Modes

### Pattern-Based (default)
Column-wise 8-bit patterns, useful for structured sparsity:
```bash
./exec_dev 28 1 --pattern 11110000  # 50% density
```

### Blockwise
Block-level warp-uniform patterns (realistic workloads):
```bash
./exec_dev 28 1 --blockwise --pattern 11110000
# Each 8×8 (A) or 8×32 (B) tile is fully dense or fully zero
```

### Random
Unstructured sparsity for comparison:
```bash
./exec_dev 28 1 --random  # 37.5% sparsity
```

## Verification

All kernels validate against cuBLAS with 1e-3 tolerance:
```bash
./exec_dev 28 1 --verbose           # With verification
./exec_dev 28 100 --no-check        # Performance-only
```

## Research Notes

Failed experiments archived in `docs/archived_experiments/`:
- K19-20: B-transpose approaches (too much overhead)
- K22-23: Early joint patterns (suboptimal indexing)
- K25-27: Skip logic ablation studies

Key finding: Joint A+B sparsity requires careful granularity tuning. Too fine (8×8) has high pattern overhead, too coarse (64×64) misses skipping opportunities. 8×32 (K28) strikes optimal balance.

## Performance Characteristics

Measured on NVIDIA A100 (80GB), 4096×4096×4096 matrices:

| Kernel | Sparsity | Time (µs) | vs Dense | Memory BW |
|--------|----------|-----------|----------|-----------|
| K10 (cuBLAS) | Dense | ~850 | 1.0x | ~85% |
| K16 (A-sparse) | 50% A | ~425 | 2.0x | ~75% |
| K21 (B-sparse) | 50% B | ~440 | 1.9x | ~72% |
| K28 (Joint) | 50%×50% | ~220 | 3.9x | ~65% |

At 75% combined sparsity: K28 achieves ~8x speedup over dense.

## Citation

If you use this work, please cite:
```
[Publication details TBD]
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
