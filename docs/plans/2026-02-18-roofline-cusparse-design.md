# Design: Roofline Model Plot + cuSPARSE Comparison

**Date:** 2026-02-18
**Status:** Approved
**Scope:** Two new figures, one new benchmark binary. Nothing existing is modified.

---

## Background

The paper currently has Figures 1–5 covering performance vs sparsity, preprocessing overhead, fusion vs granularity, batch amortization, and size scaling. Two additions will make the experimental section more rigorous:

1. **Roofline model** — shows where K25 sits relative to theoretical hardware limits, making the paper feel grounded in hardware reality.
2. **cuSPARSE comparison** — the experimental setup mentions cuSPARSE as a baseline but currently has no measured results against it.

---

## GPU Hardware Constants (A10G, sm_86)

- Peak FP32: **31,240 GFLOPS**
- Peak memory bandwidth: **600 GB/s**
- Ridge point: **52.1 FLOP/byte**

---

## Part 1: Roofline Model

### Approach

Analytical arithmetic intensity + existing timing data. No new profiling runs needed.

**Arithmetic intensity formula:**
`AI = (2 × N³ × density) / (3 × N² × 4 bytes) = N × density / 6` FLOP/byte

This models: reads of A + reads of B + writes of C (three N² matrices at FP32). Conservative — doesn't assume cache reuse between preprocessing and GEMM phases.

### Data points

| Config | AI (FLOP/byte) | Region |
|--------|---------------|--------|
| N=4096, d=12.5% | 85 | compute-bound |
| N=4096, d=25% | 171 | compute-bound |
| N=4096, d=50% | 341 | compute-bound |
| N=4096, d=87.5% | 597 | compute-bound |
| N=1024, d=12.5% | 21 | memory-bound |
| N=1024, d=50% | 85 | compute-bound |
| N=2048, d=12.5% | 43 | memory-bound |
| N=2048, d=50% | 171 | compute-bound |
| N=8192, d=50% | 683 | compute-bound |

cuBLAS (K15) plotted separately for each size as reference.

### Files

- **`scripts/experiments/plot_roofline.py`** — reads from existing summary CSVs (Figures 1 and 5), computes AI analytically, plots roofline
- **`results/figures/roofline.pdf`** — output

---

## Part 2: cuSPARSE Comparison

### What is timed

| Label | Contents |
|-------|----------|
| `cuSPARSE SpMM only` | Just `cusparseSpMM()` call |
| `cuSPARSE total` | `dense→CSR conversion + cusparseSpMM()` |
| `K25 total` | preprocessing + GEMM (from existing data) |
| `cuBLAS` | K15 (from existing data) |

The two cuSPARSE measurements bound the best case (pre-converted sparse format) and the realistic inference case (dynamic activations, format converted at runtime). K25 is comparable to "cuSPARSE total" since K25's preprocessing is also done at runtime.

### Benchmark program

**`src/benchmarks/cusparse_benchmark.cu`** — standalone CUDA C++ program.

Responsibilities:
- Accept CLI args: `--size N --density D --runs R`
- Generate dense matrix A with blockwise sparsity matching our conventions
- Convert A to CSR using cuSPARSE generic API (`cusparseDenseToSparse_convert`)
- Run cusparseSpMM with warm-up + timed runs using CUDA events
- Time `spmm_only` and `total` (conversion + spmm) separately
- Print CSV row to stdout: `size,density,spmm_us,total_us`

**Makefile target:** `cusparse_bench` — uses same ARCH and PROD_FLAGS, links `-lcusparse`.

### Data collection

**`scripts/experiments/06_collect_cusparse_data.sh`** — runs the benchmark across:
- Sizes: 1024, 2048, 4096
- Densities: 12.5%, 25%, 50%
- Output: `results/cusparse_benchmark/cusparse_results.csv`

### Plot

**`scripts/experiments/plot_cusparse_comparison.py`** — grouped bar chart at N=4096 with density on x-axis. Four bars per group: cuSPARSE SpMM only, cuSPARSE total, K25 total, cuBLAS. Y-axis: time in ms. cuBLAS shown as horizontal reference line.

**`results/figures/figure_cusparse_comparison.pdf`** — output

---

## Files Created (summary)

| File | Purpose |
|------|---------|
| `src/benchmarks/cusparse_benchmark.cu` | Standalone cuSPARSE benchmark |
| `scripts/experiments/plot_roofline.py` | Roofline model figure |
| `scripts/experiments/06_collect_cusparse_data.sh` | cuSPARSE data collection |
| `scripts/experiments/plot_cusparse_comparison.py` | cuSPARSE vs K25 figure |
| `results/figures/roofline.pdf` | Output figure |
| `results/figures/figure_cusparse_comparison.pdf` | Output figure |

## Files Modified

None. All additions only.
