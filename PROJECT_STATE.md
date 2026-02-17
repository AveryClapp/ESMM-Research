# ESMM Project State
> Comprehensive context for continuing work in a fresh session (e.g. Claude.ai web)

---

## Paper Title (working)
**ESMM: Exploiting Emergent Activation Sparsity in Dense Matrix Multiplication via Fused Pattern Extraction**

## Core Contribution
A CUDA kernel (K25 "Simple Fused") that exploits joint A+B sparsity in matrix multiplication without converting to sparse formats. The key insight: **fusion > granularity**. Fusing preprocessing + computation into a single kernel launch outperforms fine-grained approaches because ~300µs of kernel launch overhead dominates at practical sparsity levels.

---

## Main Kernel: K25 (esmm_ab_simple_fused)

**Parameters:** BM=64, BN=128, BK=8, WM=64, WN=32, WNITER=2, TN=8, NUM_THREADS=128

**Single kernel launch does three things:**
1. `preprocess_a_inline` — scans A at 64-row granularity using ballot_sync, outputs [M/64 × K/8] pattern array
2. `preprocess_b_inline` — scans B at 32-col granularity using float4 loads, outputs [N/32 × K/8] pattern array
3. `esmm_ab_compute_inline` — computes joint = a_pattern & b_pattern, iterates only over active K-blocks

**Why 64-row granularity is "good enough":** Activation sparsity (ReLU, attention masks) is block-structured in practice. Coarser patterns have better cache behavior and lower metadata overhead. Pattern storage is only ~129KB for 4096³.

---

## Key Experimental Results (4096×4096, blockwise sparsity)

### Performance vs Density
| Density | Time (K25) | Speedup vs cuBLAS |
|---------|------------|-------------------|
| 100%    | 12.1ms     | 0.60× (overhead)  |
| 87.5%   | ~10ms      | ~0.72×            |
| 75%     | ~8ms       | ~0.90×            |
| 50%     | 6.8ms      | ~1.06×            |
| 25%     | 4.4ms      | ~1.64×            |
| 12.5%   | 1.9ms      | ~3.79×            |

**cuBLAS baseline: ~7210 µs (7.2ms) at 4096×4096**

### Fusion vs Granularity Ablation
| Kernel | Granularity | Approach              | 50% Sparsity      |
|--------|-------------|-----------------------|-------------------|
| K25    | 64×32       | **Fused**             | **6.4ms** ← Winner |
| K20    | 64×32       | Separate preproc      | 7–8ms             |
| K21    | 8×32        | Separate preproc      | 11.6ms            |
| K28    | 8×32        | Branchless fused      | 22.7ms            |
| K30    | 8×8         | SpInfer-style, sep.   | 14.9ms            |

**Key finding:** Branchless (K28) and finest-granularity (K30) are both SLOWER — the branch predictor handles uniform warp-level branches well, and fine granularity increases pattern overhead faster than it reduces compute.

### K25 Time Breakdown (4096×4096, 50% density)
| Component              | Time   |
|------------------------|--------|
| preprocess_a_inline    | 0.17ms |
| preprocess_b_inline    | 0.14ms |
| esmm_ab_compute_inline | 6.1ms  |
| **Total**              | **~6.4ms** |

### B-sparse-only baseline (K17)
~1.3ms at 0% density, scales linearly with sparsity.

---

## What We Are NOT Doing
- No sparse matrix formats (CSR, COO) — too much preprocessing overhead
- No cuSPARSE — requires format conversion, doesn't exploit joint A+B sparsity
- No unstructured sparsity — branch divergence kills GPU parallelism
- Not competing on random/unstructured sparsity — targeting emergent activation sparsity

---

## Target Application
Transformer inference: After ReLU/SiLU activations, intermediate FFN activations are 50–90% sparse. Both the activation matrix (A) and weight matrix (B, pruned offline) can be exploited simultaneously. Our approach handles **runtime** A-sparsity extraction + **offline** B-pattern reuse.

---

## Comparison to Prior Work
- **SpInfer** (8×8 granularity, separate preprocessing): Our K30 ≈ SpInfer style at 14.9ms. K25 at 6.4ms is 2.3× faster despite coarser granularity — because of fusion.
- **cuSPARSE**: Requires explicit sparse formats, high format-conversion overhead, can't exploit joint A+B sparsity simultaneously.
- **Magnitude pruning + SpMM**: Works for weight-only sparsity, doesn't handle dynamic activation sparsity.

---

## Figures Being Generated (5 total)

### Figure 1 — Performance vs Sparsity (line plot)
- X: Density (0%→100%), Y: Speedup over cuBLAS
- Lines: K17 (B-only), K20 (sep. preproc), K21 (8×32 sep.), K25 (MAIN)
- Shows K25 breaks even around 45% density, gets ~1.78× at 50%

### Figure 2 — Preprocessing Overhead (stacked bar)
- X: Matrix size (1024, 2048, 4096, 8192, 16384), Y: Time (ms)
- Stacked: preprocess_a / preprocess_b / compute
- Shows preprocessing is <5% of total time at all sizes

### Figure 3 — Fusion vs Granularity (grouped bar)
- Compares K20, K21, K25 at multiple sparsity levels
- Makes the "fusion > granularity" argument visually clear

### Figure 4 — Batch Amortization (line, log scale)
- X: Batch size, Y: Amortized preprocessing overhead per sample (%)
- Shows B-pattern preprocessing overhead → 0 with larger batches

### Figure 5 — Matrix Size Scaling (dual axis)
- X: Matrix size, Y: Speedup + TFLOPS
- K17, K25, cuBLAS lines; shows advantage holds across sizes

---

## Paper Sections Needed
1. Abstract
2. Introduction (sparsity in transformers, why existing approaches fall short)
3. Background (GPU architecture basics, roofline model, emergent sparsity)
4. Method (K25 design: pattern extraction, fusion approach, granularity choice)
5. Experiments (the 5 figures above)
6. Related Work (SpInfer, SparseGPT, cuSPARSE, warp-level sparsity approaches)
7. Conclusion

---

## Codebase Key Files
| File | Purpose |
|------|---------|
| `src/kernels/esmm_ab_simple_fused.cu` | K25 main kernel (inline preproc + compute) |
| `src/kernels/esmm_ab_sparse_optimized.cu` | K20 (compute logic K25 reuses) |
| `src/preprocessors/ab_preprocessor.cu` | Separate preprocessing kernels (K20–K24) |
| `include/runners.cuh` | Kernel runner wrappers with timing |
| `driver.cu` | CLI, matrix generation, kernel dispatch |
| `scripts/benchmark.py` | Parallel NCU profiling automation |
| `scripts/experiments/` | Data collection + plotting scripts |
| `results/figures/` | Output PDFs (generated after experiments) |

**Kernel numbering:**
- K10 = Dense ESMM (internal reference)
- K15 = cuBLAS (actual baseline)
- K17 = B-sparse-only warp baseline
- K20 = A+B sparse, separate preprocessing, 64×32
- K21 = A+B sparse, separate preprocessing, 8×32
- K25 = **MAIN CONTRIBUTION** (fused, 64×32)
- K28 = Branchless fused (slower than K25)
- K30 = SpInfer-style 8×8 (slower than K25)

---

## Current Status
| Task | Status |
|------|--------|
| Figure 1 ESMM kernels (24 NCU profiles) | ✅ COMPLETE |
| cuBLAS baseline (K15, 4096×4096) | ✅ COMPLETE — 7210 µs |
| Figure 2 data | 🔄 IN PROGRESS |
| Figure 3 data | ⏳ PENDING |
| Figure 4 data | ⏳ PENDING |
| Figure 5 data | ⏳ PENDING |
| Plot generation (all 5 figures) | ⏳ PENDING |
| Writing | ❌ NOT STARTED |

---

## Writing Notes

When writing this paper, keep in mind:
- This is a **systems/architecture paper**, not a deep learning paper — focus on GPU efficiency arguments
- The audience knows CUDA and cares about roofline analysis, memory coalescing, warp utilization
- The main claim: for emergent block-structured activation sparsity at 25–75% density, fused lightweight pattern extraction + joint A+B skipping gives **1.5–3× speedup with <5% preprocessing overhead**
- Be honest about when K25 is NOT worth it: dense inputs, unstructured sparsity, small matrices (<1024×1024)
- The breakeven point is ~45% density — below that, K25 wins; above that, use cuBLAS
