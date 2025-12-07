# Understanding the Benchmark CSV Files

When you run `./scripts/benchmark_all_ab_combinations.sh`, you get three types of CSV files. Here's what each one shows:

---

## 1. **combinations_summary.csv** - Index of All Runs

**Location:** `benchmarks/TIMESTAMP_k25_ab_grid/combinations_summary.csv`

**Purpose:** Master index tracking all 64 A×B combinations

**Columns:**
- `pattern_a` - 8-bit pattern for A matrix (e.g., "11111111")
- `pattern_b` - 8-bit pattern for B matrix (e.g., "11110000")
- `density_a_pct` - A-matrix density percentage (100.0, 87.5, 75.0, ...)
- `density_b_pct` - B-matrix density percentage (100.0, 87.5, 75.0, ...)
- `output_dir` - Path to subdirectory with detailed results

**Example:**
```csv
pattern_a,pattern_b,density_a_pct,density_b_pct,output_dir
11111111,11111111,100.0,100.0,benchmarks/.../A_100pct_B_100pct
11111111,11110000,100.0,50.0,benchmarks/.../A_100pct_B_50pct
11110000,11110000,50.0,50.0,benchmarks/.../A_50pct_B_50pct
```

**Use case:** Quick lookup to find which subdirectory has which A/B combination

**Rows:** 64 (one per combination)

---

## 2. **Individual summary.csv** - Detailed Results Per Combination

**Location:** `benchmarks/TIMESTAMP_k25_ab_grid/A_XXpct_B_YYpct/summary.csv`

**Purpose:** Detailed benchmark metrics for ONE specific A×B combination

**Columns:**
- `kernel` - Which kernel ran:
  - `"PREPROCESS"` - Preprocessing kernels (2 rows: A and B preprocessors)
  - `"25"` - Main kernel K25 (1 row)
- `size` - Matrix dimension (e.g., 4096)
- `sparsity` - Label like "A100_B50"
- `pattern` - The 8-bit pattern used (shows pattern_a in consolidated view)
- `kernel_time_us` - **Execution time in microseconds** ⭐ KEY METRIC
- `memory_throughput_pct` - Memory bandwidth utilization (usually N/A without full NCU metrics)
- `compute_throughput_pct` - Compute utilization (usually N/A)
- `ncu_report` - Path to .ncu-rep file for detailed profiling
- `kernel_name` - Full CUDA kernel name

**Example (A=100%, B=50%):**
```csv
kernel,size,sparsity,pattern,kernel_time_us,memory_throughput_pct,compute_throughput_pct,ncu_report,kernel_name
PREPROCESS,4096,A100_B50,11111111,266.750,N/A,N/A,k25_4096_A100_B50.ncu-rep,"void preprocess_a_inline<...>"
PREPROCESS,4096,A100_B50,11111111,139.230,N/A,N/A,k25_4096_A100_B50.ncu-rep,"void preprocess_b_inline<...>"
25,4096,A100_B50,11111111,8105.980,N/A,N/A,k25_4096_A100_B50.ncu-rep,
```

**Understanding the rows:**
- Row 1: A-matrix preprocessor took 266.75 µs
- Row 2: B-matrix preprocessor took 139.23 µs
- Row 3: **Main kernel took 8,105.98 µs** ← This is the key number!

**Use case:** Detailed analysis of a specific combination, including preprocessing overhead

**Rows:** 3 per file (2 preprocessing + 1 main kernel)

**Total files:** 64 (one per subdirectory)

---

## 3. **consolidated_results.csv** - All Results in One File

**Location:** `benchmarks/TIMESTAMP_k25_ab_grid/consolidated_results.csv`
(Created by running `python3 scripts/consolidate_ab_results.py`)

**Purpose:** Combines all 64 individual summary.csv files into one master spreadsheet

**Columns:**
- `pattern_a`, `pattern_b` - The actual bit patterns (e.g., "11111111", "11110000")
- `density_a_pct`, `density_b_pct` - Density percentages (100.0, 87.5, 75.0, ...)
- `kernel` - "PREPROCESS" or "25" (main kernel)
- `size` - Matrix size (4096)
- `sparsity` - Combined label (e.g., "A100_B50")
- `pattern` - Shows pattern_a (for compatibility)
- `kernel_time_us` - **Execution time in microseconds** ⭐ KEY METRIC
- `memory_throughput_pct`, `compute_throughput_pct` - NCU metrics (usually N/A)
- `ncu_report` - Which .ncu-rep file
- `kernel_name` - Full kernel name

**Example:**
```csv
pattern_a,pattern_b,density_a_pct,density_b_pct,kernel,size,sparsity,pattern,kernel_time_us,...
11111111,11111111,100.0,100.0,PREPROCESS,4096,A100_B100,11111111,264.220,...
11111111,11111111,100.0,100.0,PREPROCESS,4096,A100_B100,11111111,137.570,...
11111111,11111111,100.0,100.0,25,4096,A100_B100,11111111,12251.790,...
11111111,10000000,100.0,12.5,PREPROCESS,4096,A100_B12,11111111,263.580,...
11111111,10000000,100.0,12.5,PREPROCESS,4096,A100_B12,11111111,137.730,...
11111111,10000000,100.0,12.5,25,4096,A100_B12,11111111,3881.310,...
...
```

**Use case:**
- **Easy analysis in Excel/Python/R** - all data in one file
- Filter to `kernel == "25"` to get only main kernel times
- Group by `density_a_pct` or `density_b_pct` to see trends
- Create heatmaps of kernel time vs (A density, B density)

**Rows:** 192 total (64 combinations × 3 rows each)

---

## Quick Analysis Examples

### Get only main kernel times (excluding preprocessing):
```bash
grep ",25," benchmarks/.../consolidated_results.csv > main_kernel_only.csv
```
This gives you 64 rows with just the main kernel performance.

### Find fastest combination:
```bash
grep ",25," benchmarks/.../consolidated_results.csv | sort -t, -k9 -n | head -1
```
Sorts by column 9 (kernel_time_us) numerically.

### Find slowest combination:
```bash
grep ",25," benchmarks/.../consolidated_results.csv | sort -t, -k9 -n | tail -1
```

### Analyze in Python:
```python
import pandas as pd

# Load consolidated results
df = pd.read_csv('benchmarks/.../consolidated_results.csv')

# Filter to main kernel only
main = df[df['kernel'] == '25'].copy()

# Create pivot table: A density × B density → kernel time
pivot = main.pivot_table(
    values='kernel_time_us',
    index='density_a_pct',
    columns='density_b_pct'
)

print(pivot)

# Find best combination
best = main.loc[main['kernel_time_us'].idxmin()]
print(f"Fastest: A={best['density_a_pct']}%, B={best['density_b_pct']}%")
print(f"Time: {best['kernel_time_us']:.1f} µs")
```

---

## Summary Table

| CSV File | Purpose | Rows | Key Use |
|----------|---------|------|---------|
| **combinations_summary.csv** | Index of all runs | 64 | Find which subdirectory has which A/B combo |
| **individual summary.csv** | One A/B combination | 3 | Detailed breakdown including preprocessing |
| **consolidated_results.csv** | All results combined | 192 | Easy analysis, plotting, statistics |

---

## Key Metrics to Analyze

### Main Kernel Time (`kernel == "25"`)
- **What it shows:** Total execution time of the sparse matrix multiplication kernel
- **Units:** Microseconds (µs)
- **Range in your data:** 137.6 µs (fastest) to 12,251.8 µs (slowest)
- **Lower is better**

### Preprocessing Time (`kernel == "PREPROCESS"`)
- **What it shows:** Time to analyze matrices and extract sparsity patterns
- **Two kernels:**
  - A-matrix preprocessor: ~260-270 µs (consistent)
  - B-matrix preprocessor: ~137-140 µs (consistent)
- **Note:** This is one-time overhead, amortized over multiple gemm calls

### Density Percentages
- **100%:** Fully dense (no sparsity)
- **87.5%:** 7/8 columns dense
- **75%:** 6/8 columns dense
- **62.5%:** 5/8 columns dense
- **50%:** 4/8 columns dense (half sparse)
- **37.5%:** 3/8 columns dense
- **25%:** 2/8 columns dense (highly sparse)
- **12.5%:** 1/8 columns dense (extremely sparse)

---

## Expected Patterns in Your Data

### Performance vs Sparsity
- **Dense matrices (100% × 100%):** Slowest (~12,250 µs in your data)
  - More computation required
  - Less opportunity to skip work

- **Sparse matrices (12.5% × 12.5%):** Potentially fastest
  - Less computation
  - More skipping of zero blocks

### Asymmetric Effects
- **A-sparse, B-dense:** May see different speedup than A-dense, B-sparse
- **Joint sparsity:** A=50%, B=50% might be faster than A=100%, B=25%
  - Depends on kernel's ability to exploit structure

### Preprocessing Overhead
- Preprocessing time is **constant** regardless of sparsity level
- For one-shot operations, preprocessing overhead matters
- For repeated operations (same matrices), preprocessing is amortized

---

## Troubleshooting

**"Why are there 192 rows instead of 64?"**
- Each combination has 3 rows: 2 preprocessing kernels + 1 main kernel
- Filter to `kernel == "25"` to get just the 64 main results

**"Why is memory_throughput_pct always N/A?"**
- Full NCU metric extraction requires parsing the .ncu-rep files in detail
- The current script extracts only kernel time (the most important metric)
- You can use `ncu --import <file>.ncu-rep` to see full metrics manually

**"How do I compare against cuBLAS?"**
- Run cuBLAS on same size: `cublasSgemm(4096, 4096, 4096)`
- Typical cuBLAS time: ~3,000-5,000 µs for 4096×4096 (depends on GPU)
- Your sparse kernel should be faster when sparsity is high

---

## Next Steps

1. **Load consolidated_results.csv** into your favorite tool (Excel, Python, R, MATLAB)

2. **Filter to main kernel:** `kernel == "25"`

3. **Create visualizations:**
   - Heatmap: A density (rows) × B density (cols) → kernel time (color)
   - Line plots: Fix A density, vary B density (or vice versa)
   - Scatter: Total density (A + B) vs kernel time

4. **Identify sweet spots:**
   - Which sparsity levels give best speedup?
   - Is there a minimum sparsity threshold needed?
   - Do asymmetric patterns (A≠B) help?

5. **Compare kernels:**
   - Re-run with different kernel numbers (K22, K24, K28)
   - See which kernel architecture handles sparsity best
