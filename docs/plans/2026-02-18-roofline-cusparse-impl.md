# Roofline Model + cuSPARSE Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a roofline model plot and a cuSPARSE comparison figure without touching any existing files.

**Architecture:** (1) A pure-Python plot script reads existing timing CSVs and computes arithmetic intensity analytically to produce a roofline figure. (2) A standalone CUDA C++ program benchmarks cuSPARSE SpMM (timing conversion separately), writes CSV output, a shell script collects data, and a plot script generates the comparison figure.

**Tech Stack:** CUDA 12.1, cuSPARSE generic API, Python 3 (pandas, matplotlib, numpy), existing Makefile (-lcusparse already linked).

**Hardware constants (A10G, sm_86):**
- Peak FP32: 31,240 GFLOPS
- Peak memory bandwidth: 600 GB/s
- Ridge point: 52.1 FLOP/byte

---

## Task 1: Roofline Model Plot

**Files:**
- Create: `scripts/experiments/plot_roofline.py`
- Output: `results/figures/roofline.pdf` and `roofline.png`

**Data sources (already exist):**
- `results/figure1_performance_vs_sparsity/esmm_kernels/summary.csv` — K25 at N=4096, 6 densities + K15
- `results/figure5_matrix_scaling/esmm_kernels/summary.csv` — K25 at 4 sizes, 50% density + K15

**Arithmetic intensity formula:**
```
AI(N, density) = (2 * N^3 * density) / (3 * N^2 * 4 bytes)
               = N * density / 6   [FLOP/byte]
```

Numerator: effective FLOPs done (sparse kernel skips (1-density) fraction).
Denominator: full A + B + C traffic (conservative — no cache reuse assumption).

**Step 1: Write `plot_roofline.py`**

```python
#!/usr/bin/env python3
"""
Roofline Model: K25 vs hardware limits (A10G)
Analytical arithmetic intensity + timing from existing summary CSVs.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A10G hardware limits
PEAK_GFLOPS    = 31_240.0   # FP32 GFLOPS
PEAK_BW_GBs    = 600.0      # GB/s
RIDGE_POINT    = PEAK_GFLOPS / PEAK_BW_GBs  # ≈ 52.1 FLOP/byte

PATTERN_TO_DENSITY = {
    "00000000": 0.0, "10000000": 12.5, "11000000": 25.0, "11100000": 37.5,
    "11110000": 50.0, "11111000": 62.5, "11111100": 75.0, "11111110": 87.5,
    "11111111": 100.0,
}

def pattern_to_density(pattern):
    p = str(pattern).strip()
    if p in PATTERN_TO_DENSITY:
        return PATTERN_TO_DENSITY[p]
    return (p.count('1') / 8.0) * 100.0

def arithmetic_intensity(n, density_pct):
    """AI = N * density / 6  [FLOP/byte]"""
    return n * (density_pct / 100.0) / 6.0

def effective_gflops(n, density_pct, time_us):
    """Effective GFLOPS: only count FLOPs actually performed."""
    flops = 2.0 * n**3 * (density_pct / 100.0)
    return (flops / time_us) / 1e3  # µs → s, FLOPs → GFLOPS

def load_fig1_data():
    """K25 + K15 at N=4096, varying density."""
    f = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity" / "esmm_kernels" / "summary.csv"
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel']).copy()
    df['kernel'] = df['kernel'].astype(int)
    df['density'] = df['pattern'].astype(str).str.strip().apply(pattern_to_density)
    return df

def load_fig5_data():
    """K25 + K15 at 50% density, varying N."""
    f = PROJECT_ROOT / "results" / "figure5_matrix_scaling" / "esmm_kernels" / "summary.csv"
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel']).copy()
    df['kernel'] = df['kernel'].astype(int)
    df['density'] = 50.0  # fig5 is fixed at 50% density (pattern 11110000)
    return df

def plot_roofline():
    df1 = load_fig1_data()
    df5 = load_fig5_data()

    # K25 points: densities at N=4096 (from fig1)
    k25_fig1 = df1[(df1['kernel'] == 25) & (df1['density'] > 0)].copy()
    k25_fig1 = k25_fig1.groupby('density')['kernel_time_us'].mean().reset_index()
    k25_fig1['n'] = 4096
    k25_fig1['ai']    = k25_fig1.apply(lambda r: arithmetic_intensity(r['n'], r['density']), axis=1)
    k25_fig1['gflops'] = k25_fig1.apply(lambda r: effective_gflops(r['n'], r['density'], r['kernel_time_us']), axis=1)

    # K25 points: sizes at 50% density (from fig5)
    k25_fig5 = df5[df5['kernel'] == 25].copy()
    k25_fig5 = k25_fig5.groupby('size')['kernel_time_us'].mean().reset_index()
    k25_fig5['density'] = 50.0
    k25_fig5['ai']    = k25_fig5.apply(lambda r: arithmetic_intensity(r['size'], r['density']), axis=1)
    k25_fig5['gflops'] = k25_fig5.apply(lambda r: effective_gflops(r['size'], r['density'], r['kernel_time_us']), axis=1)
    # Remove N=4096 duplicate (already in k25_fig1)
    k25_fig5 = k25_fig5[k25_fig5['size'] != 4096]

    # cuBLAS (K15) points from fig5 (dense: density=100%)
    k15 = df5[df5['kernel'] == 15].copy()
    k15 = k15.groupby('size')['kernel_time_us'].mean().reset_index()
    k15['density'] = 100.0
    k15['ai']    = k15.apply(lambda r: arithmetic_intensity(r['size'], r['density']), axis=1)
    k15['gflops'] = k15.apply(lambda r: effective_gflops(r['size'], r['density'], r['kernel_time_us']), axis=1)

    # Roofline curve
    ai_range = np.logspace(-1, 4, 500)
    roofline  = np.minimum(PEAK_GFLOPS, PEAK_BW_GBs * ai_range)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline ceilings
    ax.plot(ai_range, roofline, 'k-', linewidth=2.5, label='Roofline (A10G)', zorder=2)
    ax.axhline(PEAK_GFLOPS, color='black', linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(RIDGE_POINT, color='black', linestyle='--', linewidth=1, alpha=0.4,
               label=f'Ridge point ({RIDGE_POINT:.0f} FLOP/byte)')

    # Shade memory-bound / compute-bound regions
    ax.fill_betweenx([0, PEAK_GFLOPS], 0.1, RIDGE_POINT, alpha=0.05, color='blue')
    ax.fill_betweenx([0, PEAK_GFLOPS], RIDGE_POINT, 10000, alpha=0.05, color='red')

    # K25 points: varying density at N=4096
    sc1 = ax.scatter(k25_fig1['ai'], k25_fig1['gflops'],
                     c=k25_fig1['density'], cmap='RdYlGn_r',
                     s=120, marker='D', zorder=5,
                     vmin=0, vmax=100, label='AB-Fused (N=4096, vary density)')
    for _, row in k25_fig1.iterrows():
        ax.annotate(f"{row['density']:.0f}%",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # K25 points: varying size at 50% density
    ax.scatter(k25_fig5['ai'], k25_fig5['gflops'],
               color='red', s=100, marker='D', alpha=0.6, zorder=5,
               label='AB-Fused (50% density, vary N)')
    for _, row in k25_fig5.iterrows():
        ax.annotate(f"N={int(row['size'])}",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, -12), textcoords='offset points', fontsize=8)

    # cuBLAS reference points
    ax.scatter(k15['ai'], k15['gflops'],
               color='gray', s=100, marker='s', zorder=5,
               label='cuBLAS (dense, vary N)')
    for _, row in k15.iterrows():
        ax.annotate(f"N={int(row['size'])}",
                    xy=(row['ai'], row['gflops']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color='gray')

    # Colorbar for density
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Matrix Density (%)', fontsize=11)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(10, PEAK_GFLOPS * 2)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Effective Performance (GFLOPS)', fontsize=13, fontweight='bold')
    ax.set_title('Roofline Model: AB-Fused (K25) on A10G\n(Analytical AI, NCU-timed execution)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.2)

    # Annotations for regions
    ax.text(2, 200, 'Memory-bound', fontsize=10, color='blue', alpha=0.6, style='italic')
    ax.text(200, 200, 'Compute-bound', fontsize=10, color='red', alpha=0.6, style='italic')

    plt.tight_layout()
    out = OUTPUT_DIR / "roofline.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")

    # Summary table
    print("\n===== Roofline Summary =====")
    print(f"Peak FP32:    {PEAK_GFLOPS:,.0f} GFLOPS")
    print(f"Peak BW:      {PEAK_BW_GBs:.0f} GB/s")
    print(f"Ridge point:  {RIDGE_POINT:.1f} FLOP/byte")
    print(f"\n{'Config':30s} | {'AI':>10} | {'GFLOPS':>10} | {'% of Peak':>10}")
    print("-" * 68)
    for _, row in k25_fig1.sort_values('density').iterrows():
        pct = row['gflops'] / PEAK_GFLOPS * 100
        print(f"N=4096, density={row['density']:5.1f}% | {row['ai']:>10.1f} | {row['gflops']:>10.1f} | {pct:>9.1f}%")

if __name__ == "__main__":
    plot_roofline()
```

**Step 2: Run the script**

```bash
cd /home/ec2-user/ESMM-Research
python3 scripts/experiments/plot_roofline.py
```

Expected output:
```
✓ Saved: results/figures/roofline.pdf
===== Roofline Summary =====
Peak FP32:    31,240 GFLOPS
...
```

Expected: No errors, PDF created at `results/figures/roofline.pdf`.

**Step 3: Verify the PDF exists and is non-empty**

```bash
ls -lh results/figures/roofline.pdf
```

Expected: File exists, size > 20K.

**Step 4: Commit**

```bash
git add scripts/experiments/plot_roofline.py results/figures/roofline.pdf results/figures/roofline.png
git commit -m "Add roofline model plot for K25 on A10G"
```

---

## Task 2: cuSPARSE Benchmark Program

**Files:**
- Create: `src/benchmarks/cusparse_benchmark.cu`

**What the program does:**
1. Parse CLI: `--size N`, `--density D` (0.0–1.0), `--runs R` (default 10)
2. Allocate host/device matrices A (M×K sparse), B (K×N dense), C (M×N dense). Use M=N=K (square).
3. Fill A with blockwise sparsity: tile A into 64×8 blocks, each block either all-zero or all-random based on density.
4. Fill B with random values.
5. **Timing band 1 — SpMM only:** Convert A to CSR once (not timed), warm up, then time `cusparseSpMM` R times.
6. **Timing band 2 — total:** Time (`cusparseDenseToSparse_convert` + `cusparseSpMM`) R times each run.
7. Print one CSV row to stdout: `size,density,spmm_us,total_us,spmm_gflops,total_gflops`

**Step 1: Create `src/benchmarks/cusparse_benchmark.cu`**

```cuda
// src/benchmarks/cusparse_benchmark.cu
// Standalone cuSPARSE SpMM benchmark for comparison against K25.
// Outputs one CSV row: size,density,spmm_us,total_us,spmm_gflops,total_gflops

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(x) do { \
    cusparseStatus_t err = (x); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// Fill host matrix with blockwise sparsity: 64-row × 8-col tiles
// Matching the K25 A-matrix convention.
void fill_blockwise_sparse(float* A, int M, int K, float density) {
    const int TILE_M = 64, TILE_K = 8;
    for (int bm = 0; bm < (M + TILE_M - 1) / TILE_M; bm++) {
        for (int bk = 0; bk < (K + TILE_K - 1) / TILE_K; bk++) {
            float r = (float)rand() / RAND_MAX;
            bool active = (r < density);
            for (int m = bm * TILE_M; m < (bm + 1) * TILE_M && m < M; m++) {
                for (int k = bk * TILE_K; k < (bk + 1) * TILE_K && k < K; k++) {
                    A[m * K + k] = active ? ((float)rand() / RAND_MAX - 0.5f) : 0.0f;
                }
            }
        }
    }
}

void fill_random(float* X, int n) {
    for (int i = 0; i < n; i++) X[i] = (float)rand() / RAND_MAX - 0.5f;
}

int main(int argc, char* argv[]) {
    int    N    = 4096;
    float  density = 0.5f;
    int    runs = 10;
    bool   print_header = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc)    N = atoi(argv[++i]);
        if (strcmp(argv[i], "--density") == 0 && i + 1 < argc) density = atof(argv[++i]);
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)    runs = atoi(argv[++i]);
        if (strcmp(argv[i], "--header") == 0)                   print_header = true;
    }

    if (print_header) {
        printf("size,density,spmm_us,total_us,spmm_gflops,total_gflops\n");
        return 0;
    }

    int M = N, K = N;
    long long flops = 2LL * N * N * N;

    srand(42);

    // ---- Host allocation and fill ----
    float* hA = (float*)malloc((long long)M * K * sizeof(float));
    float* hB = (float*)malloc((long long)K * N * sizeof(float));
    fill_blockwise_sparse(hA, M, K, density);
    fill_random(hB, K * N);

    // ---- Device allocation ----
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, (long long)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, (long long)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, (long long)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, (long long)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, (long long)K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, (long long)M * N * sizeof(float)));

    // ---- cuSPARSE setup ----
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Dense descriptor for A (for conversion)
    cusparseDnMatDescr_t dnA;
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnA, M, K, K, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Sparse CSR descriptor for A (values/indices allocated after analysis)
    cusparseSpMatDescr_t spA;
    int64_t* d_csr_offsets; int64_t* d_csr_cols; float* d_csr_vals;
    CHECK_CUDA(cudaMalloc(&d_csr_offsets, (M + 1) * sizeof(int64_t)));

    CHECK_CUSPARSE(cusparseCreateCsr(&spA, M, K, 0,
                                     d_csr_offsets, nullptr, nullptr,
                                     CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Analysis pass (counts nnz, fills offsets)
    size_t conv_buf_size = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, dnA, spA,
                                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                    &conv_buf_size));
    void* conv_buf;
    CHECK_CUDA(cudaMalloc(&conv_buf, conv_buf_size));
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, dnA, spA,
                                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                   conv_buf));

    int64_t nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(spA, nullptr, nullptr, &nnz));

    CHECK_CUDA(cudaMalloc(&d_csr_cols, nnz * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc(&d_csr_vals, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(spA, d_csr_offsets, d_csr_cols, d_csr_vals));

    // Pre-convert once for SpMM-only baseline
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                  conv_buf));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Dense descriptors for B (K×N) and C (M×N)
    cusparseDnMatDescr_t dnB, dnC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnB, K, N, N, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnC, M, N, N, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // SpMM buffer
    size_t spmm_buf_size = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, spA, dnB, &beta, dnC,
                                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                            &spmm_buf_size));
    void* spmm_buf;
    CHECK_CUDA(cudaMalloc(&spmm_buf, spmm_buf_size));

    // ---- Timing: SpMM only (A already in CSR) ----
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // Warm up
    for (int i = 0; i < 3; i++) {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float spmm_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&spmm_ms, t0, t1));
    double spmm_us = (spmm_ms / runs) * 1000.0;

    // ---- Timing: total = conversion + SpMM ----
    // Warm up
    for (int i = 0; i < 2; i++) {
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                      CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                      conv_buf));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < runs; i++) {
        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dnA, spA,
                                                      CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                      conv_buf));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, spA, dnB, &beta, dnC,
                                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                     spmm_buf));
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, t0, t1));
    double total_us = (total_ms / runs) * 1000.0;

    double spmm_gflops  = (double)flops / (spmm_us  * 1e-6) / 1e9;
    double total_gflops = (double)flops / (total_us * 1e-6) / 1e9;

    printf("%d,%.4f,%.3f,%.3f,%.2f,%.2f\n",
           N, density, spmm_us, total_us, spmm_gflops, total_gflops);

    // Cleanup
    cusparseDestroyDnMat(dnA); cusparseDestroyDnMat(dnB); cusparseDestroyDnMat(dnC);
    cusparseDestroySpMat(spA);
    cusparseDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(d_csr_offsets); cudaFree(d_csr_cols); cudaFree(d_csr_vals);
    cudaFree(conv_buf); cudaFree(spmm_buf);
    free(hA); free(hB);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
```

**Step 2: Add Makefile target for `cusparse_bench`**

Add to the bottom of `Makefile` (before the `.PHONY: info` line):

```makefile
# cuSPARSE benchmark (standalone, for comparison against K25)
CUSPARSE_BENCH := cusparse_bench
CUSPARSE_SRC   := src/benchmarks/cusparse_benchmark.cu

.PHONY: cusparse_bench
cusparse_bench: $(CUSPARSE_BENCH)

$(CUSPARSE_BENCH): $(CUSPARSE_SRC)
	@echo "Building cuSPARSE benchmark..."
	$(NVCC) $(PROD_FLAGS) $(CUSPARSE_SRC) -o $(CUSPARSE_BENCH) $(LIBS)
	@echo "Build complete: ./$(CUSPARSE_BENCH)"
```

Also add `cusparse_bench` to the `clean` target:
```makefile
# existing clean target — add cusparse_bench:
clean:
	rm -f $(TARGET) $(PROD_TARGET) $(TEST_TARGET) $(PROFILE_TARGET) $(CUSPARSE_BENCH) *.o profile.ncu-rep
```

**Step 3: Build and smoke-test**

```bash
mkdir -p src/benchmarks
make cusparse_bench
```

Expected: Compiles without errors, binary `./cusparse_bench` created.

```bash
./cusparse_bench --size 1024 --density 0.5 --runs 5
```

Expected: One CSV line printed, e.g.:
```
1024,0.5000,45.234,89.123,46.23,23.45
```

**Step 4: Test edge cases**

```bash
./cusparse_bench --size 1024 --density 0.125 --runs 5
./cusparse_bench --size 2048 --density 0.25 --runs 5
./cusparse_bench --header
```

Expected: All produce valid output without CUDA errors.

**Step 5: Commit**

```bash
git add src/benchmarks/cusparse_benchmark.cu Makefile
git commit -m "Add standalone cuSPARSE SpMM benchmark (timing SpMM-only and total with conversion)"
```

---

## Task 3: cuSPARSE Data Collection Script

**Files:**
- Create: `scripts/experiments/06_collect_cusparse_data.sh`
- Output: `results/cusparse_benchmark/cusparse_results.csv`

**Step 1: Create `06_collect_cusparse_data.sh`**

```bash
#!/bin/bash
# Experiment 6: cuSPARSE comparison (Figure: cusparse_comparison)
# Benchmarks cuSPARSE SpMM vs K25 at high sparsity levels.
# Reports both SpMM-only time and total (conversion + SpMM) time.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/cusparse_benchmark"
OUTPUT_CSV="$OUTPUT_DIR/cusparse_results.csv"

echo "===== cuSPARSE Benchmark ====="
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

# Build if not already built
if [ ! -f "./cusparse_bench" ]; then
    echo "Building cusparse_bench..."
    make cusparse_bench
fi

SIZES="1024 2048 4096"
# Densities to test (cuSPARSE shines at high sparsity = low density)
# 0.0625 = 6.25%, 0.125 = 12.5%, 0.25 = 25%, 0.5 = 50%
DENSITIES="0.0625 0.125 0.25 0.5"
RUNS=20

echo "size,density,spmm_us,total_us,spmm_gflops,total_gflops" > "$OUTPUT_CSV"

for SIZE in $SIZES; do
    for DENSITY in $DENSITIES; do
        echo "  Running: size=$SIZE density=$DENSITY"
        ./cusparse_bench --size "$SIZE" --density "$DENSITY" --runs "$RUNS" >> "$OUTPUT_CSV"
    done
done

echo ""
echo "===== Complete ====="
echo "Results: $OUTPUT_CSV"
echo ""
cat "$OUTPUT_CSV"
echo ""
echo "Next steps:"
echo "  python3 scripts/experiments/plot_cusparse_comparison.py"
```

**Step 2: Make executable and run**

```bash
chmod +x scripts/experiments/06_collect_cusparse_data.sh
bash scripts/experiments/06_collect_cusparse_data.sh
```

Expected: Takes ~5 minutes. Produces `results/cusparse_benchmark/cusparse_results.csv` with 12 data rows (3 sizes × 4 densities).

**Step 3: Sanity-check the output**

```bash
cat results/cusparse_benchmark/cusparse_results.csv
```

Expected: 13 lines (header + 12 data rows). At `size=4096, density=0.125`, `spmm_us` should be well below K25's 4685 µs (cuSPARSE is fast for very sparse matrices). At `density=0.5`, cuSPARSE should be closer to or slower than K25.

**Step 4: Commit**

```bash
git add scripts/experiments/06_collect_cusparse_data.sh results/cusparse_benchmark/cusparse_results.csv
git commit -m "Add cuSPARSE data collection script and initial benchmark results"
```

---

## Task 4: cuSPARSE Comparison Plot

**Files:**
- Create: `scripts/experiments/plot_cusparse_comparison.py`
- Output: `results/figures/figure_cusparse_comparison.pdf`

**Data sources:**
- `results/cusparse_benchmark/cusparse_results.csv` — cuSPARSE timings
- `results/figure1_performance_vs_sparsity/esmm_kernels/summary.csv` — K25 and K15 at N=4096

**Step 1: Create `plot_cusparse_comparison.py`**

```python
#!/usr/bin/env python3
"""
cuSPARSE vs K25 comparison at high sparsity (N=4096).
Shows cuSPARSE SpMM-only, cuSPARSE total (conv+SpMM), K25 total, cuBLAS.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITY_LABELS = {
    0.0625: "6.25%",
    0.125:  "12.5%",
    0.25:   "25%",
    0.5:    "50%",
}

PATTERN_TO_DENSITY = {
    "10000000": 12.5, "11000000": 25.0, "11110000": 50.0,
    "11111100": 75.0, "11111110": 87.5,
}

def load_cusparse(n=4096):
    f = PROJECT_ROOT / "results" / "cusparse_benchmark" / "cusparse_results.csv"
    df = pd.read_csv(f)
    return df[df['size'] == n].copy()

def load_k25_k15(n=4096):
    f = PROJECT_ROOT / "results" / "figure1_performance_vs_sparsity" / "esmm_kernels" / "summary.csv"
    df = pd.read_csv(f, comment='#')
    df['kernel'] = df['kernel'].astype(str).str.strip()
    df = df[df['kernel'] != 'PREPROCESS'].copy()
    df['kernel'] = pd.to_numeric(df['kernel'], errors='coerce')
    df = df.dropna(subset=['kernel'])
    df['kernel'] = df['kernel'].astype(int)
    df['density_pct'] = df['pattern'].astype(str).str.strip().map(PATTERN_TO_DENSITY)
    df = df.dropna(subset=['density_pct'])
    df['density'] = df['density_pct'] / 100.0
    return df[df['size'] == n] if 'size' in df.columns else df

def plot_cusparse_comparison():
    sp = load_cusparse(n=4096)
    esmm = load_k25_k15(n=4096)

    k25 = esmm[esmm['kernel'] == 25].groupby('density')['kernel_time_us'].mean().reset_index()
    k15 = esmm[esmm['kernel'] == 15]['kernel_time_us'].mean()  # cuBLAS (density-independent)

    # Align on common densities
    densities = sorted(sp['density'].unique())
    density_labels = [DENSITY_LABELS.get(d, f"{d*100:.1f}%") for d in densities]

    x = np.arange(len(densities))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def get_k25_time(d):
        row = k25[np.isclose(k25['density'], d, atol=0.01)]
        return row['kernel_time_us'].values[0] / 1000 if not row.empty else np.nan

    spmm_times    = sp['spmm_us'].values / 1000
    total_times   = sp['total_us'].values / 1000
    k25_times     = [get_k25_time(d) for d in densities]
    cublas_time   = k15 / 1000

    # Left: absolute time (ms)
    b1 = ax1.bar(x - 1.5*width, spmm_times,  width, label='cuSPARSE SpMM only', color='steelblue',  alpha=0.85, edgecolor='black')
    b2 = ax1.bar(x - 0.5*width, total_times, width, label='cuSPARSE total (conv+SpMM)', color='cornflowerblue', alpha=0.85, edgecolor='black')
    b3 = ax1.bar(x + 0.5*width, k25_times,   width, label='AB-Fused ★ (K25)',  color='red',        alpha=0.85, edgecolor='black')
    ax1.axhline(y=cublas_time, color='gray', linestyle='--', linewidth=2,
                label=f'cuBLAS ({cublas_time:.1f} ms)')

    ax1.set_xlabel('Matrix Density', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kernel Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Time (N=4096)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(density_labels)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bar value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                         f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    # Right: speedup vs cuBLAS
    spmm_speedup  = cublas_time / spmm_times
    total_speedup = cublas_time / total_times
    k25_speedup   = np.array([cublas_time / t if not np.isnan(t) else np.nan for t in k25_times])

    ax2.plot(density_labels, spmm_speedup,  's-', color='steelblue',      linewidth=2, markersize=9, label='cuSPARSE SpMM only')
    ax2.plot(density_labels, total_speedup, 'o-', color='cornflowerblue', linewidth=2, markersize=9, label='cuSPARSE total')
    ax2.plot(density_labels, k25_speedup,   'D-', color='red',            linewidth=2.5, markersize=9, label='AB-Fused ★')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='cuBLAS (baseline)')

    ax2.set_xlabel('Matrix Density', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup vs cuBLAS (N=4096)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('cuSPARSE vs AB-Fused: Sparse Matrix Multiply (4096×4096)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "figure_cusparse_comparison.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")

    print("\n===== cuSPARSE vs K25 Summary (N=4096) =====")
    print(f"{'Density':>10} | {'cuSP SpMM':>12} | {'cuSP Total':>12} | {'K25':>10} | {'cuBLAS':>10}")
    print("-" * 65)
    for d, sl, tl, k25t in zip(densities, spmm_times, total_times, k25_times):
        lbl = DENSITY_LABELS.get(d, f"{d*100:.1f}%")
        k25_str = f"{k25t:.2f} ms" if not np.isnan(k25t) else "  N/A"
        print(f"{lbl:>10} | {sl:>10.2f} ms | {tl:>10.2f} ms | {k25_str:>10} | {cublas_time:>8.2f} ms")

if __name__ == "__main__":
    plot_cusparse_comparison()
```

**Step 2: Run the script**

```bash
python3 scripts/experiments/plot_cusparse_comparison.py
```

Expected: Produces `results/figures/figure_cusparse_comparison.pdf` and prints a comparison table.

**Step 3: Inspect the figure**

Check:
- At 6.25% density, cuSPARSE SpMM-only should clearly beat cuBLAS (many zeros skipped).
- At 50% density, cuSPARSE should be near or below cuBLAS (not enough zeros).
- K25 total should be competitive with cuSPARSE total at all densities (they include different overheads).

**Step 4: Commit**

```bash
git add scripts/experiments/plot_cusparse_comparison.py \
        results/figures/figure_cusparse_comparison.pdf \
        results/figures/figure_cusparse_comparison.png
git commit -m "Add cuSPARSE comparison plot"
```

---

## Task 5: Final Verification

**Step 1: Regenerate all figures to confirm nothing broken**

```bash
python3 scripts/experiments/plot_roofline.py
python3 scripts/experiments/plot_cusparse_comparison.py
python3 scripts/experiments/plot_figure1.py
python3 scripts/experiments/plot_figure2.py
python3 scripts/experiments/plot_figure3.py
python3 scripts/experiments/plot_figure5.py
```

Expected: All complete without errors.

**Step 2: List all figures**

```bash
ls -lh results/figures/*.pdf
```

Expected: 7 PDFs — figure1 through figure5, roofline, figure_cusparse_comparison.

**Step 3: Final commit**

```bash
git add results/figures/
git commit -m "Generate all figures including roofline and cuSPARSE comparison"
```
