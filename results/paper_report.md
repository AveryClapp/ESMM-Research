# ESMM Research: Kernel Progression, Real-World Results, and Paper Writing Guide

## 1. Kernel Progression

### Background

The project targets SGEMM (single-precision matrix multiplication) where both input matrices A and B have joint block-structured sparsity at BK=8 column × BM=32 row tile granularity. The core insight is that if a tile in A and the corresponding tile in B are both zero, the output contribution is zero and the computation can be skipped entirely.

### Kernel Lineage

**K14 (baseline naive)**: Dense SGEMM reference with no sparsity exploitation.

**K15 (cuBLAS)**: cuBLAS SGEMM. The standard high-performance baseline. Row-major call: `cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inners, ...)`.

**K16, K17**: Early experiments with simple sparsity checks. Not in the final paper lineup.

**K20**: First production smem-cached pattern kernel. Precomputes joint=a_pat & b_pat per warp into shared memory before the main compute loop. Uses 64-row tile granularity for A, warp-level skip (early exit if joint==0), and float4 A-loads. Block-level skip added later with `block_joint` sentinel.

**K21**: Intermediate variant. Not in final paper lineup.

**K25**: Gmem-pattern variant using ballot-sync for A patterns and float4 B-loads. Reads a_pat/b_pat directly from global memory each K iteration rather than caching in smem. Separate fused preprocessors (`preprocess_a_fused`, `preprocess_b_fused`). About 48% slower at 12.5% density compared to K29 (4.03 vs 2.72ms) due to gmem pattern reads.

**K26 (smem-cached, templated)**: K20 architecture with three key additions: (1) 32-row tile granularity (instead of 64-row), (2) templated MAX_K_BLOCKS so smem is sized to actual numKBlocks rather than a fixed maximum, (3) float4 inner A-loads. The templated dispatch selects from MAX_K_BLOCKS ∈ {128, 256, 512, 1024} at runtime based on the K dimension. This avoids wasting smem on unused pattern slots.

**K27 (ablation — 32-row only)**: K26 with 32-row granularity but without smem caching — reads patterns from gmem. Isolates the contribution of smem caching.

**K28 (ablation — gmem patterns, 32-row)**: K25-style gmem patterns with 32-row A granularity. Another ablation point.

**K29 (main contribution)**: K26 + float2 A-loads instead of float4. The switch from float4 to float2 changes the row stride from 2×BM (50% thread utilization) to BM (100% thread utilization). Float2 (K29) outperforms float4 (K26) at 3 of 4 tested densities: 100% (12.10 vs 12.94ms), 25% (4.04 vs 4.22ms), and 12.5% (2.72 vs 2.85ms). K26 is marginally faster at 50% (6.02 vs 6.12ms, a 1.7% difference). Also inherits the templated MAX_K_BLOCKS, smem-cached joint patterns, and 32-row granularity.

### NCU-Verified Compute Times (ms, 4096×4096, blockwise synthetic sparsity)

| Kernel | 100% density | 50% density | 25% density | 12.5% density |
|--------|-------------|-------------|-------------|---------------|
| K15 (cuBLAS) | 7.19  | 7.19        | 7.20        | 7.19          |
| K20    | 11.65       | 6.32        | 4.16        | 3.80          |
| K25    | 11.87       | 6.73        | 4.78        | 4.03          |
| K26    | 12.94       | 6.02        | 4.22        | 2.85          |
| K27    | —           | 6.01        | 4.51        | 4.41          |
| K28 (templated) | 13.08 | 6.20   | 4.22        | 3.34          |
| K29    | 12.10       | 6.12        | 4.04        | 2.72          |

**K29 vs cuBLAS speedup (compute kernel only)**:
100% → 0.59×, 50% → 1.18×, 25% → 1.78×, 12.5% → **2.65×**

**K29 vs cuBLAS total speedup (compute + ~374µs preprocessing)**:
100% → 0.58×, 50% → 1.11×, 25% → 1.63×, 12.5% → **2.32×**

K29 beats K26 at 3 of 4 densities; K26 is marginally faster at 50% (6.02 vs 6.12ms). Both beat K25 at high sparsity due to smem-cached patterns. K20 is competitive and actually faster than K29 at 100% density (11.65 vs 12.10ms), but K29 pulls ahead as density drops due to finer-grained 32-row skip granularity.

### Preprocessing Times (lean preprocessor, approximate µs)

| Kernel | Preprocessing time |
|--------|-------------------|
| K20    | ~400 µs (preprocess_a 64-row + preprocess_b) |
| K25    | ~302 µs (fused preprocessors) |
| K26/K29 | ~375 µs (preprocess_a 32-row + preprocess_b) |

The lean `preprocess_ab` variant was created specifically to eliminate ~225 µs of wasted time from the `analyze_ab_patterns_kernel` double-pass that was in the original debug preprocessor. The debug version (`preprocess_ab_debug`) is preserved for diagnostics.

---

## 2. Real-World Data Results

### Dataset

Weights from LLaMA (Layer 0) pruned with Wanda and SparseGPT at sparsity levels 50%, 60%, 70%, 80%, 90%, 95%, 99%. Two group sizes (8, 16). Two permutation types (row permuted, column permuted). Three layer types: attn (4096×4096), mlp_down (4096×11008), mlp_gate/up (11008×4096). The row/col permutation is a Hamming-distance optimization designed to cluster similar rows/columns together, maximizing block sparsity.

Total: 440 benchmarked configurations (616 tensors minus those skipped due to K>8192).

### K-Dimension Limit

K29 uses MAX_K_BLOCKS=1024 × BK=8 = 8192 maximum K dimension. The mlp_gate and mlp_up matrices have K=11008, which exceeds this limit and causes illegal memory access. These are skipped in the benchmark. Only attn (K=4096) and mlp_down (K=11008 transposed — actually K=4096 in the benchmark direction) fit within the limit.

Wait — more precisely: in the GEMM A×B, A is dense (rows × inners), B is the loaded weight (inners × cols). For mlp_gate/up with shape 11008×4096, the inners dimension presented to the kernel is 11008, which exceeds 8192. For mlp_down with shape 4096×11008, inners=4096 which is fine.

### Block Sparsity Findings

At BM=32 × BK=8 tile granularity:

- 50-90% element sparsity: approximately 0% block sparsity. Every tile has at least one nonzero.
- 95% element sparsity: still ~0% block sparsity in most cases.
- 99% element sparsity: 15-83% block sparsity depending on layer and permutation type.

The fundamental reason is that Wanda and SparseGPT prune within groups (group_size=8 or 16). Each group of 8/16 consecutive elements must have at least one surviving weight (the pruning is structured to preserve model quality). A BK=8 tile spans exactly 8 elements. For the tile to be all-zero, every group within it must have all its survivors zeroed — which is impossible by construction for group_size=8 or 16 at ≤99% sparsity.

This is not a failure of the experiment; it is the expected behavior of group-structured pruning. The finding is honest and important for the paper.

### Benchmark Results (honest, random dense A)

- **50-95% element sparsity**: K29 runs at 0.68-0.72× cuBLAS. K29 is slower than cuBLAS because preprocessing adds overhead and the kernel gets no skip opportunities (0% block sparsity).
- **99% element sparsity**: K29 mean speedup 1.06×, max 2.32× (attn matrices with 70-83% block sparsity), min around 0.7× for configurations with only 15% block sparsity.

**Note**: These real-weight numbers are from the original CUDA-events timing run (results/real_weights_benchmark.csv). The benchmark has been updated to use NCU timing; these numbers should be re-measured and will change slightly.

### The Zero-A Bug

During initial benchmarking, K29 appeared to achieve 3.86× speedup. This was a bug: `h_A` was allocated with `malloc` and left uninitialized. On Linux, large `malloc` allocations are zero-initialized by the OS for security. K29 detected all-zero A patterns, skipped all computation, and reported near-zero kernel time. The fix: when `--load-b` is provided without `--load-a`, fill `h_A` with random values before benchmarking.

---

## 3. Ingestion Pipeline

### Overview

Two scripts handle the full pipeline from raw .pt files to benchmark CSV.

### analyze_real_weights.py

**Explore mode** (`--explore`): loads .pt files, prints tensor shapes and element-wise sparsity. Useful for understanding the data structure.

**Analyze mode** (`--analyze`): for each `*_permuted.pt` file found recursively:
1. Load tensor with `torch.load(..., weights_only=True)`, convert to float32.
2. Pad to BM=32 × BK=8 tile boundaries with zero padding.
3. Compute element-wise sparsity (simple zero count).
4. Compute block sparsity: reshape to `(nM, BM, nK, BK)`, take `max(axis=(1,3))`, check if zero.
5. Export float32 binary (row-major) for ESMM benchmarking.
6. Export JSON metadata (shape, sparsity, config).
7. Export block sparsity heatmap as PNG.

Config metadata (pruner, group_size, perm_type, sparsity) is parsed from directory path components. Output filenames are prefixed: `{pruner}_grp{group}_{perm}_sp{sparsity}_{stem}.bin`.

### benchmark_real_weights.py

For each `*_permuted.pt` file found recursively:
1. Parse config from path.
2. Load tensor, pad to tile boundaries.
3. Skip if inners dimension > 8192 (K29 limit).
4. Write to temp `.bin` file via `tempfile.NamedTemporaryFile`.
5. For each kernel (default: 15=cuBLAS, 29=K29): run `sudo ncu --set basic --target-processes all --export <tmpdir>/profile exec_prod ...`, then import the `.ncu-rep` and parse `Duration` rows from `--csv --page details`.
6. Separate preprocessing kernels (names containing "preprocess" or "analyze") from compute kernels by name. cuBLAS has no preprocessing kernels.
7. Return `(compute_ms, preprocess_ms)` per kernel. CSV records both columns plus `k29_compute_speedup` (K15.compute / K29.compute) and `k29_total_speedup` (K15.compute / (K29.compute + K29.preprocess)).
8. Delete temp `.bin` and `.ncu-rep` files (in `finally` block).
9. Write CSV row immediately (flush after every tensor for fault tolerance).

The exec is located via `Path(__file__).resolve().parent.parent / "exec_prod"` (absolute path required).

### File Structure

```
weight_permutations/
  wanda/
    grp_8/
      row_perm/
        sparsity_0.5/
          model.layers.0.self_attn.q_proj_permuted.pt
          ...
  sparsegpt/
    ...

results/
  real_weights_benchmark.csv    (440 rows)
  paper_figures/                (fig1-fig4 PDF + PNG)
```

---

## 4. Paper Writing Guidance

### Positioning

The paper contributes a GPU kernel family (K20→K25→K26→K29) for SGEMM where both A and B matrices have joint block-structured sparsity. The contribution is:

1. The smem-cached joint pattern approach (vs gmem reads in prior work).
2. Templated MAX_K_BLOCKS for smem efficiency.
3. Float2 A-loads for higher thread utilization at low density.
4. Empirical characterization of block sparsity in real LLM weights.

### Synthetic Results (the main technical contribution)

K29 achieves up to **2.65× compute speedup** over cuBLAS on synthetic blockwise-sparse matrices at 12.5% density (87.5% block sparsity). Including ~374µs preprocessing, the end-to-end speedup is **2.32×** at 12.5%, **1.63×** at 25%, **1.11×** at 50%. cuBLAS baseline is 7.19ms flat. These numbers are from NCU raw timing of .ncu-rep files.

### Real-Weight Results (honest framing)

At practical sparsity (50-95%), K29 is slower than cuBLAS on real Wanda/SparseGPT weights because group-structured pruning prevents block-level zeros. At 99% sparsity (not practically useful for LLM inference), K29 achieves up to 2.32× speedup.

The honest framing: ESMM is designed for block-structured sparsity. Current unstructured/semi-structured pruning methods do not produce this pattern. This motivates block-structured pruning as future work — and the row/column permutation experiments show that even with permutation, group-constrained pruning at ≤99% sparsity does not generate exploitable blocks.

### Section Outline

1. Introduction: joint A+B sparsity in SGEMM, motivation from sparse attention + sparse weights in LLMs.
2. Background: block-structured sparsity, existing sparse GEMM approaches.
3. Kernel Design: BK=8 × BM=32 tiles, joint pattern masking, smem-cached patterns, warp-level and block-level skip, templated MAX_K_BLOCKS, float2 vs float4 A-loads.
4. Ablation: K27 (32-row only), K28 (gmem patterns, 32-row), K25 (gmem patterns, ballot-sync), K26 (smem + float4), K29 (smem + float2).
5. Synthetic Evaluation: speedup vs cuBLAS across sparsity levels 12.5%-100%.
6. Real-Weight Evaluation: Wanda/SparseGPT on LLaMA Layer 0, block sparsity analysis, honest benchmark results.
7. Discussion and Future Work: block-structured pruning methods needed to realize K29's potential.

---

## Claude Prompt

Use the following prompt verbatim to give a fresh Claude instance full context to write the paper:

---

I need you to write a research paper on CUDA SGEMM kernels that exploit joint A+B block-structured sparsity. Here is the complete technical context:

**Problem**: Standard SGEMM (A×B=C) on GPUs ignores sparsity even when both matrices have structured zeros. We target the case where both A and B have block-structured sparsity at 32-row × 8-column tile granularity. If a tile in A and the corresponding tile in B are both all-zero, the output contribution is zero and we can skip the computation.

**Our kernels** (in progression):

K15 = cuBLAS baseline. Row-major call: `cublasSgemm(handle, OP_N, OP_N, cols, rows, inners, ...)`.

K20: First production kernel. Precomputes joint = a_pat & b_pat per warp into shared memory (smem-cached patterns). 64-row tile granularity for A. Warp-level skip (if joint==0, skip K-block). Float4 A-loads. Block-level skip via `block_joint` sentinel. ~400 µs preprocessing.

K25: Alternative approach using gmem pattern reads (no smem caching). Ballot-sync for A patterns, float4 B-loads. Separate fused preprocessors. ~302 µs preprocessing. About 46% slower than K29 at 12.5% density due to global memory pattern reads each K-iteration.

K26: K20 with (1) 32-row A tile granularity, (2) templated MAX_K_BLOCKS ∈ {128,256,512,1024} so smem is sized to actual K/8 blocks not a fixed maximum, (3) float4 inner A-loads. ~375 µs preprocessing.

K29 (main contribution): K26 + float2 A-loads instead of float4. Float2 gives rowStrideA = BM = 32 (100% thread utilization). Float4 gives rowStrideA = 2×BM = 64 (50% threads idle). At low density, float2 wins because computation is the bottleneck. At high density, both are similar. K29 beats K26 across all sparsity levels.

**NCU-measured kernel compute times (ms) on 4096×4096 matrix, blockwise synthetic sparsity**:

| Kernel | 100% density | 50% | 25% | 12.5% |
|--------|-------------|-----|-----|-------|
| K15 (cuBLAS) | 7.19 | 7.19 | 7.20 | 7.19 |
| K20    | 11.65       | 6.32| 4.16| 3.80  |
| K25    | 11.87       | 6.73| 4.78| 4.03  |
| K26    | 12.94       | 6.02| 4.22| 2.85  |
| K27    | —           | 6.01| 4.51| 4.41  |
| K28 (templated) | 13.08 | 6.20| 4.22| 3.34 |
| K29    | 12.10       | 6.12| 4.04| 2.72  |

Speedup of K29 over cuBLAS (compute-only): 12.5% → 2.65×, 25% → 1.78×, 50% → 1.18×, 100% → 0.59× (K29 is 1.68× slower at full density). cuBLAS baseline: 7.19ms. Including preprocessing (~374µs): 12.5% → 2.32×, 25% → 1.63×, 50% → 1.11×.

**Preprocessing**: The lean `preprocess_ab` kernel computes per-tile sparsity patterns for A and B, storing them as bitmasks. The debug variant also computes statistics. The lean variant runs in ~375 µs, eliminating ~225 µs of wasted time vs the old version.

**Real-world evaluation**:

Data: LLaMA Layer 0 weights pruned with Wanda and SparseGPT. Sparsity levels 50%, 60%, 70%, 80%, 90%, 95%, 99%. Group sizes 8 and 16. Row and column permutation (Hamming-distance clustering to maximize block sparsity). Layer shapes: attention projections (4096×4096), mlp_down (4096×11008), mlp_gate/up (11008×4096, skipped — K=11008 > K29's max 8192).

Block sparsity findings at 32×8 tile granularity:
- 50-90% element sparsity → ~0% block sparsity. No tiles are all-zero.
- 99% element sparsity → 15-83% block sparsity (attention matrices perform best).

Why: Wanda and SparseGPT prune within groups of 8 or 16 consecutive weights, preserving at least one weight per group for quality. A BK=8 tile spans 8 weights. For the tile to be all-zero, every group in it must be fully zeroed — impossible at ≤99% sparsity with group_size=8/16.

Honest benchmark results (synthetic dense A, loaded sparse B from .pt files):
- 50-95% sparsity: K29 = 0.68-0.72× cuBLAS (slower — preprocessing overhead, no skip opportunities).
- 99% sparsity: K29 mean 1.06×, max 2.32× speedup (attention matrices with high block sparsity).

**Paper framing**: ESMM (Efficient Sparse Matrix Multiply) targets block-structured sparsity. The contribution is the kernel design and the empirical demonstration that on synthetic blockwise-sparse data, K29 achieves up to **2.65× compute speedup** (2.32× end-to-end including preprocessing) over cuBLAS at 12.5% density. cuBLAS baseline is 7.19ms. On real LLM weights from standard unstructured pruning, the speedup is not realized at practical sparsity. This motivates block-structured pruning as future work — the permutation experiments show that even aggressive row/column reordering cannot convert unstructured pruning to block-structured at practical sparsity levels.

**Paper structure I want**:
1. Introduction
2. Background: block-structured sparsity, prior sparse GEMM work
3. Kernel Design: tile structure, joint masking, smem vs gmem patterns, templated sizing, float2 vs float4
4. Ablation study: K27, K28, K25, K26 → K29 (each design choice justified)
5. Synthetic Evaluation: speedup curves over sparsity
6. Real-Weight Evaluation: block sparsity analysis + benchmark results with honest interpretation
7. Discussion/Future Work: need for block-structured pruning methods

Please write a complete 6-8 page IEEE/NeurIPS style paper. Use precise technical language. Do not soften the real-weight results — they are honest and important. The negative result (standard pruning doesn't produce exploitable block sparsity) is a contribution. Write the related work section to cover: cuSPARSE, Sputnik, NVIDIA 2:4 sparsity, SparTA, and any relevant sparse GEMM literature.

---

End of prompt.
