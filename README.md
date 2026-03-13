# ESMM: Emergent Sparsity Matrix Multiplication

ESMM is a CUDA SGEMM research project exploiting *joint* block-structured sparsity in both operands. The key insight is that when both A (activations) and B (weights) have zero blocks at the same positions, that K-iteration contributes nothing to C and can be skipped entirely. This multiplicative effect makes joint sparsity far more exploitable than single-matrix sparsity.

The main contribution is **ESMM** — a warp-tiled SGEMM kernel with block- and warp-level skipping, smem-cached joint patterns, and tight templated shared memory allocation — benchmarked against cuBLAS on real LLM weight tensors.

## Quick Start

```bash
# Build
make release          # optimized (exec_prod)
make dev              # with assertions (exec_dev)

# Run a kernel vs cuBLAS at 4096×4096, 50% block density
./exec_prod 29,15 10 --blockwise --pattern 11110000 --size 4096

# Load a real weight matrix as B (random dense A generated automatically)
./exec_prod 29 10 --load-b path/to/weight.bin --dims 4096 4096 4096 --no-check

# Profile with NCU
sudo /usr/local/cuda-12.1/bin/ncu --set basic ./exec_prod 29 1 --blockwise --size 4096
```

## Where to Look

| What you want | Where to look |
|---|---|
| Kernel implementations | `src/kernels/` |
| Preprocessing (pattern extraction) | `src/preprocessors/ab_preprocessor.cu` |
| Kernel dispatch / runner wrappers | `include/runners.cuh` |
| CLI, matrix setup, verification | `driver.cu` |
| Synthetic benchmark suite | `scripts/benchmark.py` |
| Real LLM weight benchmark | `scripts/benchmark_real_weights.py` |
| Architecture diagrams | `docs/images/` |

## Approach

Sparsity is represented as 8-bit patterns over BK=8 K-slices. A pattern byte encodes which of the 8 columns in a tile are nonzero. Joint skipping fires when `a_pattern & b_pattern == 0` — i.e., no column is nonzero in both A and B simultaneously.

**Block-level skip**: one warp ORs all warp-pair joints into a shared byte; if zero, all threads skip the tile load entirely.

**Warp-level skip**: each warp checks its own joint; if zero, skips the inner accumulation loop.

**Smem-cached patterns**: joint patterns are precomputed into shared memory before the K-loop, avoiding repeated global memory reads per iteration.

Preprocessing runs as separate GPU kernels before the main GEMM. B patterns can be cached offline (weights are static); only A patterns need to be computed per call.

## Building

Requires CUDA 12.0+, compute capability 7.0+ (Volta/Turing/Ampere).

```bash
make release    # → exec_prod
make dev        # → exec_dev
make clean
```

## Benchmarking Real Weights

```bash
# Benchmark ESMM and cuBLAS on LLM weight tensors
python3 scripts/benchmark_real_weights.py \
  --weights /path/to/weights/ \
  --kernels 15,29 \
  --out results/real_weights_benchmark.csv
```

Weights should be `.pt` files (2D float tensors) organized under a directory tree that encodes pruner, group size, permutation type, and sparsity in path components. See `scripts/benchmark_real_weights.py` for the expected naming convention.

## Reproducing Results

The key result is ESMM vs cuBLAS at 4096×4096 across block densities. Build first (`make release`), then:

```bash
# Sweep block densities — reproduces the main performance curve
# (default sparsity patterns cover 100%, 50%, 25%, 12.5%)
python3 scripts/benchmark.py \
  --kernel 15,29 \
  --sizes 4096 \
  --blockwise \
  --runs 10

# Single point: ESMM vs cuBLAS at 12.5% density (expected ~2.6× speedup)
./exec_prod 29,15 10 --blockwise --pattern 10000000 --size 4096
```

For accurate timing (wall-clock includes host overhead), use NCU to extract pure compute durations:

```bash
sudo /usr/local/cuda-12.1/bin/ncu --set basic --csv \
  ./exec_prod 29 1 --blockwise --pattern 00010000 --size 4096
```

**Expected results at 4096×4096 (A100/similar Ampere GPU):**

| Density | ESMM (ms) | cuBLAS (ms) | Speedup |
|---------|-----------|-------------|---------|
| 100%    | 12.10     | 7.19        | 0.59×   |
| 50%     | 6.12      | 7.19        | 1.17×   |
| 25%     | 4.04      | 7.19        | 1.78×   |
| 12.5%   | 2.72      | 7.19        | 2.65×   |

Note: these are NCU compute-only times. Total end-to-end latency (including ~375 µs preprocessing) is ~2.32× at 12.5% density. Preprocessing for B can be done offline; only A patterns are computed at inference time.

## Citation

If you use this work, please cite:

```bibtex
@misc{esmm2026,
  title  = {ESMM: Exploiting Joint Block-Structured Sparsity in Matrix Multiplication},
  author = {Clapp, Avery and Burns, Randal},
  year   = {2026},
  note   = {\url{https://github.com/AveryClapp/ESMM-Research}}
}
```

A preprint will be linked here upon release.

## License

MIT — see [LICENSE](LICENSE).
