# ESMM Paper Experiments

This directory contains all scripts needed to reproduce the figures in the ESMM paper.

## Quick Start

```bash
# Run all experiments and generate all figures (~2 hours)
bash scripts/experiments/run_all_experiments.sh

# Or run individual experiments:
bash scripts/experiments/01_collect_figure1_data.sh
python3 scripts/experiments/plot_figure1.py
```

Output will be in `results/figures/*.pdf`

## Experiments Overview

### Figure 1: Performance vs Sparsity (Main Result)
**Purpose:** Shows K17/K20/K21/K25 speedup over cuBLAS across sparsity levels, highlighting K25 (fused) as main contribution

**Data Collection:**
```bash
bash scripts/experiments/01_collect_figure1_data.sh
```
- Runs K17 (B-only), K20 (separate), K21 (8×32), K25 (fused) at 4096×4096
- Sparsity levels: 0%, 12.5%, 25%, 50%, 75%, 87.5%
- Runs cuBLAS baseline (dense, constant across sparsity)
- Optionally runs cuSPARSE baseline (if implemented)
- Runtime: ~15 minutes

**Plotting:**
```bash
python3 scripts/experiments/plot_figure1.py
```
- Generates line plot with sparsity (x-axis) vs speedup (y-axis)
- Shows cuBLAS (1.0× baseline), cuSPARSE (if available), K17, K20, K21, K25
- Annotates peak speedups for K21 and K25

**Expected Output:**
- **K25 peak speedup: ~1.78× at 50% density** ⭐ (MAIN KERNEL)
- K21 peak speedup: ~1.4× (fine granularity, separate preprocessing)
- K20 peak speedup: ~1.6× (coarse granularity, separate preprocessing)
- K17 peak speedup: ~1.5× (B-sparse only)

---

### Figure 2: Preprocessing Overhead
**Purpose:** Shows preprocessing overhead decreases with matrix size (K25 fused approach)

**Data Collection:**
```bash
bash scripts/experiments/02_collect_figure2_data.sh
```
- Runs K25 (fused) at sizes: 1024, 2048, 4096, 8192, 16384
- 50% sparsity (representative)
- Separates inline preprocessing time from GEMM time
- Runtime: ~20 minutes

**Plotting:**
```bash
python3 scripts/experiments/plot_figure2.py
```
- Left subplot: Stacked bar chart (preprocessing + GEMM)
- Right subplot: Overhead percentage vs matrix size
- Horizontal line at 5% threshold

**Expected Output:**
- 1024×1024: ~23% overhead
- 4096×4096: ~2.4% overhead
- 16384×16384: ~0.6% overhead

---

### Figure 3: Fusion vs Granularity Tradeoff
**Purpose:** Shows that fusion (K25) beats fine granularity (K21) across all sparsity levels

**Data Collection:**
```bash
bash scripts/experiments/03_collect_figure3_data.sh
```
- Runs K20 (64-row separate), K21 (8×32 separate), K25 (64×32 fused) at 4096×4096
- Full sparsity range: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
- Runtime: ~25 minutes

**Plotting:**
```bash
python3 scripts/experiments/plot_figure3.py
```
- Left subplot: K20 vs K21 vs K25 speedup across sparsity
- Right subplot: Absolute performance at 50% density (bar chart)
- Shows key insight: Fusion > Granularity

**Expected Output:**
- K25 (64×32 fused): ~6.4ms at 50% density ⭐ FASTEST
- K20 (64×32 separate): ~7-8ms (similar compute, worse launch overhead)
- K21 (8×32 separate): ~11.6ms (fine granularity doesn't compensate for overhead)
- K25 consistently beats K21 across all sparsity levels

---

### Figure 4: Batch Amortization
**Purpose:** Shows preprocessing overhead becomes negligible with batching (K25 fused approach)

**Data Collection:**
```bash
bash scripts/experiments/04_collect_figure4_data.sh
```
- Uses preprocessing + GEMM times from K25 Figure 2 data (4096×4096, 50% density)
- No actual benchmarking - just arithmetic calculation
- Runtime: <1 minute

**Plotting:**
```bash
python3 scripts/experiments/plot_figure4.py
```
- Line plot: batch size (x-axis, log scale) vs overhead % (y-axis)
- Calculates: `overhead = preprocess / (preprocess + GEMM * batch_size)`
- Shades typical LLM batch range (32-128)
- Horizontal lines at 1% and 5% thresholds

**Expected Output:**
- Batch size 1: ~2.4% overhead
- Batch size 32: ~0.08% overhead
- Batch size 128: ~0.02% overhead
- Conclusion: Negligible overhead at production batch sizes

---

### Figure 5: Matrix Size Scaling (Optional)
**Purpose:** Shows K17/K25 maintain speedup across matrix sizes

**Data Collection:**
```bash
bash scripts/experiments/05_collect_figure5_data.sh
```
- Runs K17 (B-only), K25 (fused), cuBLAS at sizes: 1024, 2048, 4096, 8192, 16384
- 50% sparsity (representative)
- Runtime: ~30 minutes

**Plotting:**
```bash
python3 scripts/experiments/plot_figure5.py
```
- Left subplot: Speedup vs matrix size
- Right subplot: Absolute TFLOPS vs matrix size
- Shows K25 approach generalizes beyond single problem size

**Expected Output:**
- K25 maintains ~1.78× speedup across all sizes
- TFLOPS increases with matrix size (better GPU utilization)
- K17 (B-only) baseline for comparison

---

## Directory Structure

```
scripts/experiments/
├── README.md                          # This file
├── run_all_experiments.sh             # Master script
│
├── 01_collect_figure1_data.sh         # Data collection scripts
├── 02_collect_figure2_data.sh
├── 03_collect_figure3_data.sh
├── 04_collect_figure4_data.sh
├── 05_collect_figure5_data.sh
│
├── plot_figure1.py                    # Plotting scripts
├── plot_figure2.py
├── plot_figure3.py
├── plot_figure4.py
└── plot_figure5.py

results/
├── figure1_performance_vs_sparsity/   # Raw benchmark data
│   └── esmm_kernels/
│       └── summary.csv
├── figure2_preprocessing_overhead/
├── figure3_granularity_tradeoff/
├── figure4_batch_amortization/
├── figure5_matrix_scaling/
│
└── figures/                           # Final outputs
    ├── figure1_performance_vs_sparsity.pdf
    ├── figure1_performance_vs_sparsity.png
    ├── figure2_preprocessing_overhead.pdf
    └── ...
```

## Dependencies

- Python 3.6+
- pandas
- matplotlib
- numpy
- CUDA toolkit with `ncu` (NVIDIA Nsight Compute)
- ESMM executable (`./exec_dev`)

## Troubleshooting

### Problem: "exec_dev not found"
**Solution:** Run `make dev` in project root

### Problem: NCU permission denied
**Solution:** Run benchmark scripts with sudo, or configure NCU permissions:
```bash
sudo chmod +x /usr/local/cuda-*/bin/ncu
```

### Problem: cuBLAS/cuSPARSE baselines not found
**Solution:** The scripts will use K10 (dense GEMM) as a fallback. To implement proper cuBLAS baseline:
1. Edit `scripts/run_cublas_baseline.py`
2. Call `cublasSgemm` directly or via wrapper

### Problem: Out of memory
**Solution:** Reduce matrix sizes or run experiments individually

### Problem: Inconsistent results
**Solution:** Use `--cold-start` flag (already enabled in scripts) and ensure GPU is idle

## Advanced Usage

### Run Single Experiment
```bash
# Collect data
bash scripts/experiments/01_collect_figure1_data.sh

# Generate plot
python3 scripts/experiments/plot_figure1.py
```

### Custom Matrix Sizes
Edit the data collection scripts and change the `SIZES` variable:
```bash
SIZES="1024,2048,4096"  # Your custom sizes
```

### Custom Sparsity Patterns
Edit the data collection scripts and change the `SPARSITY_PATTERNS` variable:
```bash
SPARSITY_PATTERNS="11110000,11000000"  # 50% and 25% density
```

### Re-run Specific Kernel
```bash
cd /path/to/ESMM-Research
python3 scripts/benchmark.py \
  --kernel 28 \
  --sizes 4096 \
  --sparsity "11110000" \
  --cold-start \
  --output-dir results/manual_run
```

### Extract Specific Metrics from NCU Reports
```bash
# Find the .ncu-rep file
NCU_FILE="results/figure1_*/k28_4096_50pct.ncu-rep"

# Extract metrics
/usr/local/cuda-*/bin/ncu --import $NCU_FILE --csv --page details > metrics.csv

# Parse metrics
grep "Duration" metrics.csv
grep "Memory Throughput" metrics.csv
```

## Notes

- All experiments use `--cold-start` for reproducibility
- Preprocessing time is automatically tracked and separated from GEMM time
- Speedup is always relative to cuBLAS (or K10 dense GEMM as proxy)
- Figures are saved as both PDF (high-quality) and PNG (preview)

## Citation

If you use these experiments in your paper, please cite:
```
[Your paper citation]
```

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.
