# ESMM Research: Paper Writing Guide

## Core Value Proposition

### The Gap You're Filling
- **Existing work focuses on single-matrix sparsity** (either A or B sparse, not both)
- **Real-world workloads often have structured sparsity in BOTH matrices** (e.g., pruned neural networks, block-sparse attention)
- **Pattern-based sparsity enables compile-time optimizations** that traditional sparse formats (CSR/COO) can't exploit
- **Granularity matters**: Coarse-grained skipping (entire warps) vs fine-grained (8-row tiles) has different tradeoffs

### Your Contribution
1. **Joint A+B exploitation**: First to systematically explore different granularities for dual-sparse GEMM
2. **K-dimension skipping**: When corresponding K-blocks in A and B are both zero, skip the entire FMA tile
3. **Practical pattern preprocessing**: GPU-based pattern extraction that's cheap enough to amortize
4. **Granularity analysis**: 8×32 tiles (K28) vs 64-row blocks (K22-K24) - when does fine-grained pay off?

## Key Research Questions to Answer

1. **When does joint sparsity beat single-matrix sparsity?**
   - At what sparsity level does K28 beat K17 (B-only)?
   - What's the crossover point?

2. **What's the cost-benefit of fine-grained skipping?**
   - K28 (8-row) has more pattern metadata overhead than K24 (64-row)
   - But can skip more precisely
   - Where's the sweet spot?

3. **How close to theoretical roofline can you get?**
   - At 50% sparsity, theoretical speedup is 2× (if zero overhead)
   - What's your achieved speedup?
./scripts/benchmark.py -k 10,17,24,28 \
  --sparsity 11111111,11110000,11000000,10000000 \
  --sizes 2048,4096,8192 --cold-start
```

**Expected insights**:
- K28 should beat K24 at high sparsity (more precise skipping)
- K24 might beat K28 at low sparsity (less metadata overhead)
- Both should beat K17 (B-only) at moderate-to-high sparsity

**Plot**: Speedup vs sparsity level (12.5%, 25%, 50%, 100%) for each kernel

### 3. Granularity Analysis (Novel Contribution)

Compare K22 (64-row, simple), K24 (64-row, optimized), K28 (8-row):

```bash
./scripts/benchmark.py -k 22,24,28 \
  --sparsity 11110000 --sizes 4096 --cold-start
```
   - What's the gap, and what causes it? (pattern overhead, divergence, memory traffic)

4. **Does preprocessing cost amortize?**
   - One-time preprocessing cost vs repeated kernel savings
   - How many GEMM calls needed to break even?

## Critical Experiments

### 1. Baseline Comparisons (Essential)

```bash
# Dense baseline
cuBLAS vs K10 (your dense kernel)

# Single-matrix sparse
cuSPARSE (if applicable) vs K17 (B-sparse) vs K14 (A-sparse)

# Your joint sparse kernels
K22, K24, K28 at different sparsities
```

**Metrics**:
- Absolute GFLOPS
- Speedup over cuBLAS
- % of theoretical peak (memory bandwidth, compute throughput)

### 2. Sparsity Scaling Study (Core Result)

```bash
./scripts/benchmark.py -k 10,17,24,28 \
  --sparsity 11111111,11110000,11000000,10000000 \
  --sizes 2048,4096,8192 --cold-start
```

**Expected insights**:
- K28 should beat K24 at high sparsity (more precise skipping)
- K24 might beat K28 at low sparsity (less metadata overhead)
- Both should beat K17 (B-only) at moderate-to-high sparsity

**Plot**: Speedup vs sparsity level (12.5%, 25%, 50%, 100%) for each kernel

### 3. Granularity Analysis (Novel Contribution)

Compare K22 (64-row, simple), K24 (64-row, optimized), K28 (8-row):

```bash
./scripts/benchmark.py -k 22,24,28 \
  --sparsity 11110000 --sizes 4096 --cold-start
```

**Profile with NCU**:
- Memory throughput (% of peak)
- Warp stall reasons (memory dependency, execution dependency)
- Register pressure differences

**Key question**: Does fine-grained pattern metadata (8× more entries for K28) hurt L1/L2 cache?

### 4. Preprocessing Overhead Analysis

Measure preprocessing time separately:
```bash
# Extract from benchmark.py output (it tracks preprocessing)
# Or manually profile preprocessing kernels
```

**Calculate amortization**:
- One-time preprocessing: X ms
- Per-kernel savings at 50% sparsity: Y ms
- Break-even point: X/Y iterations

**Real-world context**: How many GEMM calls per preprocessing in typical workloads? (e.g., transformer layer processes same weight matrix for entire batch)

### 5. Roofline Analysis (Credibility Check)

For each kernel at each sparsity level:

**Theoretical peak**:
- Memory bandwidth: ~900 GB/s (A100)
- Compute: ~19.5 TFLOPS (FP32, A100)

**Arithmetic intensity**:
```
Dense GEMM: (2*M*N*K) FLOPs / ((M*K + K*N + M*N) * 4 bytes)
50% sparse: Same numerator / (0.5*M*K + 0.5*K*N + M*N + pattern_metadata) bytes
```

**Plot**: Achieved performance vs roofline model
- Show where each kernel lands (memory-bound vs compute-bound)
- Highlight gap to theoretical peak

### 6. Real-World Pattern Study (If time permits)

Test with realistic sparsity patterns from:
- Pruned neural network weights (e.g., BERT-family models)
- Block-sparse attention masks
- Scientific computing (FEM matrices, etc.)

**Show**: Structured patterns (blockwise) perform better than random sparsity

## Paper Structure Suggestion

### 1. Introduction
- Motivation: Joint sparsity is common (pruned NNs, structured attention)
- Gap: Existing libraries don't exploit dual sparsity efficiently
- Contribution: Granularity-aware joint A+B sparse GEMM

### 2. Background
- CUDA memory hierarchy, warp execution model
- Existing sparse GEMM approaches (CSR, COO, pattern-based)
- Roofline model for performance analysis

### 3. Design Space Exploration
- **Pattern preprocessing**: GPU-based extraction (why it's cheap enough)
- **Granularity choices**: 64-row blocks vs 8-row tiles
- **K-dimension skipping**: When A[m, k] and B[k, n] both zero → skip
- **Memory layout**: Pattern metadata overhead analysis

### 4. Implementation
- K22/K24 (coarse 64-row) architecture
- K28 (fine 8-row) architecture
- Key optimizations: coalescing, shared memory reuse, warp-level skipping

### 5. Experimental Results
- Setup: A100 GPU, CUDA 12.1, matrix sizes, sparsity patterns
- **Result 1**: Scaling across sparsity (12.5% to 100%)
- **Result 2**: Granularity comparison (8-row vs 64-row)
- **Result 3**: Vs cuBLAS/cuSPARSE (if applicable)
- **Result 4**: Roofline analysis (where's the bottleneck?)
- **Result 5**: Preprocessing amortization

### 6. Discussion
- When to use fine-grained (K28) vs coarse-grained (K24)?
- Limitations: Pattern metadata overhead, divergence at boundaries
- Future work: Other granularities, different sparsity formats, multi-GPU

### 7. Related Work
- cuSPARSE, Sputnik (single-matrix sparse)
- DeepSparse, TVM (compiler-based approaches)
- Sparse tensor algebra compilers (TACO)

### 8. Conclusion
- Joint A+B sparsity provides X% speedup at Y% sparsity
- Fine-grained skipping wins at high sparsity, coarse-grained at low
- Preprocessing amortizes over Z iterations

## Red Flags to Avoid

### Don't oversell
- If K28 only beats K24 marginally, acknowledge it
- If you're not beating cuBLAS at 100% density, explain why (not the goal)
- If preprocessing is expensive, quantify the amortization point

### Be honest about bottlenecks
- Show roofline analysis - are you memory-bound or compute-bound?
- If you're hitting 80% of theoretical peak, that's great! Say so.
- If you're at 40%, explain the gap (pattern overhead, divergence, etc.)

### Statistical rigor
- Run benchmarks multiple times (median of 5-10 runs)
- Report variance or confidence intervals
- Use cold-start mode for fair comparison

## Immediate Next Steps

### 1. Run comprehensive benchmark sweep

```bash
./scripts/benchmark.py -k 10,17,22,24,28 \
  --sizes 1024,2048,4096,8192 \
  --sparsity 11111111,11110000,11100000,11000000,10100000,10000000 \
  --cold-start --parallel 1
```

### 2. Extract NCU profiles for key configurations

```bash
# K28 at 50% sparsity, size 4096
sudo LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH \
  /usr/local/cuda-12.1/bin/ncu --set full -o k28_50pct \
  ./exec_dev 28 1 --size 4096 --pattern 11110000 --blockwise --no-check
```

### 3. Analyze results
- Create plots: speedup vs sparsity, roofline diagrams
- Identify where K28 wins/loses vs K24
- Calculate preprocessing amortization

### 4. Write draft focusing on
- Clear problem statement (joint sparsity gap)
- Novel contribution (granularity analysis)
- Honest experimental analysis (when it works, when it doesn't)

## The Story

**Elevator pitch**: "We systematically explore fine-grained vs coarse-grained skipping for joint A+B sparse GEMM, showing that 8×32 tiles beat 64-row blocks at >X% sparsity, achieving Y% of theoretical peak at realistic sparsity levels."

## Key Metrics to Report

### Performance Metrics
- **Absolute performance**: GFLOPS achieved
- **Relative speedup**: vs cuBLAS (dense baseline)
- **Efficiency**: % of theoretical peak (both memory and compute)
- **Scaling**: How performance varies with matrix size and sparsity

### Overhead Metrics
- **Pattern metadata size**: Bytes per matrix dimension
- **Preprocessing time**: Absolute time + amortization analysis
- **Memory traffic**: Total bytes transferred (from NCU)

### Architecture Metrics
- **Occupancy**: Achieved vs theoretical
- **Warp efficiency**: % of active threads
- **Register usage**: Per thread
- **Shared memory usage**: Per block

## Comparison Table Template

| Kernel | Granularity | Sparsity | Size | GFLOPS | Speedup vs Dense | % Peak BW | % Peak Compute |
|--------|-------------|----------|------|--------|------------------|-----------|----------------|
| K10    | Dense       | 100%     | 4096 | X      | 1.00×            | Y%        | Z%             |
| K17    | B-only      | 50%      | 4096 | X      | A.BC×            | Y%        | Z%             |
| K24    | 64-row A+B  | 50%      | 4096 | X      | A.BC×            | Y%        | Z%             |
| K28    | 8-row A+B   | 50%      | 4096 | X      | A.BC×            | Y%        | Z%             |

## Questions to Answer with Data

1. **At what sparsity level does joint A+B beat B-only?**
   - Plot: K17 vs K24/K28 across sparsity levels
   - Expected: Crossover around 30-40% sparsity

2. **Does finer granularity always win?**
   - Plot: K24 vs K28 across sparsity levels
   - Expected: K28 wins at high sparsity, K24 might win at low sparsity

3. **Is preprocessing worth it?**
   - Calculate: preprocessing_time / (kernel_speedup × num_iterations)
   - Show: Break-even point (number of iterations needed)

4. **Are we memory-bound or compute-bound?**
   - From NCU: Compare achieved vs theoretical bandwidth/compute
   - Identify bottleneck and explain why

5. **How does performance scale with matrix size?**
   - Plot: GFLOPS vs matrix size for different kernels
   - Check: Does larger size improve efficiency (better amortization)?
