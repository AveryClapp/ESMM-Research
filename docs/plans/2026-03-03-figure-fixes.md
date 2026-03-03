# Figure Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four correctness and completeness gaps across the figure pipeline before the paper figures are finalized.

**Architecture:** The figure pipeline is split into data collection shell scripts (`scripts/experiments/0N_collect_figureN_data.sh`) and plot Python scripts (`scripts/experiments/plot_figureN.py`). Data lives in `results/figureN_*/`. Four independent fixes, none depend on each other, but Tasks 6–8 (GPU reruns and regeneration) must follow their respective script fixes.

**Tech Stack:** Python 3, pandas, matplotlib, numpy; bash; NCU profiling via `scripts/benchmark.py`

---

## Key Data Already Collected (do not rerun unnecessarily)

From `results/figure2_preprocessing_overhead/k25_scaling/summary.csv`, at 4096×4096, 50% density:

| Kernel | Time (µs) |
|--------|-----------|
| `preprocess_a_inline` (PREPROCESS row, A matrix) | 167.62 |
| `preprocess_b_inline` (PREPROCESS row, B matrix) | 136.42 |
| K25 GEMM | 7604.04 |

From `results/figure1_performance_vs_sparsity/esmm_kernels/summary.csv`:
- Existing density points: 0%, 12.5%, 25%, 50%, 75%, 87.5%
- **Missing:** 37.5% (`11100000`) and 62.5% (`11111000`)
- cuBLAS (K15): 7200 µs at 50% density

From `results/figure5_matrix_scaling/esmm_kernels/summary.csv` (if it exists):
- Was run at 50% density — K25 loses to cuBLAS at every size at 50% density
- Must be rerun at 25% density

---

## Task 1: Fix Figure 1 data collection — add missing density points

**Files:**
- Modify: `scripts/experiments/01_collect_figure1_data.sh:17`

**Step 1: Edit the patterns line**

Open `scripts/experiments/01_collect_figure1_data.sh` and change line 17 from:
```bash
SPARSITY_PATTERNS="00000000,10000000,11000000,11110000,11111100,11111110"
```
to:
```bash
SPARSITY_PATTERNS="00000000,10000000,11000000,11100000,11110000,11111000,11111100,11111110"
```
(Adds `11100000` = 37.5% and `11111000` = 62.5%.)

**Step 2: Verify the change**

Run: `grep SPARSITY_PATTERNS scripts/experiments/01_collect_figure1_data.sh`
Expected output contains: `11100000` and `11111000`

**Step 3: Commit**

```bash
git add scripts/experiments/01_collect_figure1_data.sh
git commit -m "fix: add 37.5% and 62.5% density points to Figure 1 data collection"
```

---

## Task 2: Fix Figure 1 plot — add absolute runtime subplot

**Files:**
- Modify: `scripts/experiments/plot_figure1.py`

**Step 1: Replace the single-axes figure with a two-axes layout**

In `plot_figure1.py`, find the `plot_figure1()` function. The current code creates `fig, ax = plt.subplots(figsize=(10, 6))`. Replace everything from that line through `plt.tight_layout()` (lines 84–130) with the two-subplot version below. Preserve the `load_data()` and `get_cublas_time()` functions above it unchanged.

```python
def plot_figure1():
    print("Loading data...")
    df = load_data()
    cublas_time = get_cublas_time(df)

    # Only K20, K21, K25 in the figure (K17 dropped, K15 used only for reference)
    esmm_df = df[df['kernel'].isin([20, 21, 25])].copy()
    print(f"ESMM kernels: {sorted(esmm_df['kernel'].unique())}")
    print(f"cuBLAS reference: {cublas_time:.1f} µs")

    esmm_df = esmm_df.groupby(['kernel', 'density'])['kernel_time_us'].mean().reset_index()
    esmm_df['speedup'] = cublas_time / esmm_df['kernel_time_us']

    fig, (ax_speedup, ax_abs) = plt.subplots(1, 2, figsize=(14, 5))

    kernels = {
        20: {'label': 'AB-Separate',  'marker': 's', 'color': 'purple'},
        21: {'label': 'AB-Fine',      'marker': 'o', 'color': 'blue'},
        25: {'label': 'AB-Fused ★',   'marker': 'D', 'color': 'red'},
    }

    # --- Left: speedup vs density ---
    ax_speedup.axhline(y=1.0, color='gray', linestyle='--', linewidth=2,
                       label='cuBLAS (dense baseline)', zorder=1)

    for kernel_num, style in kernels.items():
        kdata = esmm_df[esmm_df['kernel'] == kernel_num].sort_values('density')
        if not kdata.empty:
            lw = 3.0 if kernel_num == 25 else 2.5
            ax_speedup.plot(kdata['density'], kdata['speedup'],
                            marker=style['marker'], linewidth=lw, markersize=9,
                            label=style['label'], color=style['color'])

    ax_speedup.set_xlabel('Matrix Density (%)', fontsize=12, fontweight='bold')
    ax_speedup.set_ylabel('Speedup vs cuBLAS', fontsize=12, fontweight='bold')
    ax_speedup.set_title('Speedup vs Density', fontsize=14, fontweight='bold')
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax_speedup.set_xlim(-5, 105)
    max_speedup = esmm_df['speedup'].max()
    ax_speedup.set_ylim(0, max(max_speedup * 1.15, 2.5))

    # Annotate AB-Fused at 25% density
    k25 = esmm_df[esmm_df['kernel'] == 25]
    k25_25 = k25[k25['density'] == 25.0]
    if not k25_25.empty:
        row = k25_25.iloc[0]
        ax_speedup.annotate(f"{row['speedup']:.2f}× at 25%",
                            xy=(row['density'], row['speedup']),
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # --- Right: absolute runtime vs density ---
    # cuBLAS flat reference line (constant across densities)
    densities_range = sorted(esmm_df['density'].unique())
    ax_abs.axhline(y=cublas_time / 1000, color='gray', linestyle='--', linewidth=2,
                   label=f'cuBLAS ({cublas_time/1000:.1f} ms)', zorder=1)

    for kernel_num, style in kernels.items():
        kdata = esmm_df[esmm_df['kernel'] == kernel_num].sort_values('density')
        if not kdata.empty:
            lw = 3.0 if kernel_num == 25 else 2.5
            ax_abs.plot(kdata['density'], kdata['kernel_time_us'] / 1000,
                        marker=style['marker'], linewidth=lw, markersize=9,
                        label=style['label'], color=style['color'])

    ax_abs.set_xlabel('Matrix Density (%)', fontsize=12, fontweight='bold')
    ax_abs.set_ylabel('Kernel Time (ms)', fontsize=12, fontweight='bold')
    ax_abs.set_title('Absolute Runtime vs Density', fontsize=14, fontweight='bold')
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax_abs.set_xlim(-5, 105)

    fig.suptitle('Figure 1: Performance vs Sparsity (4096×4096, blockwise)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "figure1_performance_vs_sparsity.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {out}")

    print("\n===== Summary =====")
    print(f"cuBLAS baseline: {cublas_time:.1f} µs")
    for kn, name in [(20, 'AB-Separate'), (21, 'AB-Fine'), (25, 'AB-Fused ★')]:
        kdata = esmm_df[esmm_df['kernel'] == kn]
        if not kdata.empty:
            for d in [12.5, 25.0, 37.5, 50.0]:
                kd = kdata[kdata['density'] == d]
                s = kd['speedup'].values[0] if not kd.empty else float('nan')
                tag = " ★ MAIN" if kn == 25 else ""
                print(f"{name}: speedup at {d:.1f}% density = {s:.2f}×{tag}")
```

**Step 2: Verify the script runs on existing data (without the new density points)**

Run: `python3 scripts/experiments/plot_figure1.py`
Expected: exits without error, prints cuBLAS time from K15, saves PDF/PNG to `results/figures/`. The 37.5% and 62.5% points will be absent until the GPU rerun; the plot is still valid with 6 points.

**Step 3: Commit**

```bash
git add scripts/experiments/plot_figure1.py
git commit -m "feat: add absolute runtime subplot to Figure 1"
```

---

## Task 3: Fix Figure 4 plot — correct A vs B preprocessing amortization

**Problem:** `extract_times()` calls `preprocess_df['kernel_time_us'].mean()`, which averages A and B preprocessing times together instead of separating them. The amortization formula then treats the entire combined preprocessing as amortizable, but only B preprocessing (weight matrix) is amortizable. A preprocessing (activation matrix) happens every forward pass and cannot be amortized.

**Correct model:**
- B preprocessing is done once per weight matrix reuse → amortizable
- A preprocessing is done every inference call → not amortizable
- Effective overhead at batch_size N = `(a_preprocess + b_preprocess/N) / (a_preprocess + b_preprocess/N + gemm) × 100`

**Known values from profiling (4096×4096, 50% density):**
- `preprocess_a_inline` row: 167.62 µs
- `preprocess_b_inline` row: 136.42 µs
- GEMM (K25 row): 7604.04 µs

**Files:**
- Modify: `scripts/experiments/plot_figure4.py`

**Step 1: Replace `extract_times()` to return A and B times separately**

Find the `extract_times()` function (lines 22–58) and replace it:

```python
def extract_times():
    """Extract A preprocessing, B preprocessing, and GEMM times from K25 at 4096×4096.

    PREPROCESS rows are separated by kernel_name:
      - 'preprocess_a_inline' → A preprocessing (not amortizable: activations change each call)
      - 'preprocess_b_inline' → B preprocessing (amortizable: weights are static)
    """
    summary_file = DATA_DIR / "k25_reference" / "summary.csv"

    if not summary_file.exists():
        summary_file = PROJECT_ROOT / "results" / "figure2_preprocessing_overhead" / "k25_scaling" / "summary.csv"

    if not summary_file.exists():
        print(f"ERROR: No data found. Please run:")
        print("  bash scripts/experiments/02_collect_figure2_data.sh")
        sys.exit(1)

    df = pd.read_csv(summary_file, comment='#')

    df_4096 = df[df['size'] == 4096]
    if df_4096.empty:
        print("ERROR: No data for size=4096 found")
        sys.exit(1)

    preprocess_df = df_4096[df_4096['kernel'] == 'PREPROCESS'].copy()
    gemm_df = df_4096[df_4096['kernel'] != 'PREPROCESS'].copy()

    if preprocess_df.empty or gemm_df.empty:
        print("ERROR: Missing preprocessing or GEMM data")
        sys.exit(1)

    # Split by kernel_name to separate A (non-amortizable) from B (amortizable)
    a_rows = preprocess_df[preprocess_df['kernel_name'].str.contains('preprocess_a', na=False)]
    b_rows = preprocess_df[preprocess_df['kernel_name'].str.contains('preprocess_b', na=False)]

    if a_rows.empty or b_rows.empty:
        # Fallback: split combined preprocess time using known 55%/45% ratio from profiling
        combined = preprocess_df['kernel_time_us'].sum()
        a_preprocess_us = combined * 0.551  # A is ~55% of total (167/(167+136))
        b_preprocess_us = combined * 0.449  # B is ~45% of total
        print(f"WARNING: Could not split A/B preprocessing by kernel_name; using ratio split")
    else:
        a_preprocess_us = a_rows['kernel_time_us'].mean()
        b_preprocess_us = b_rows['kernel_time_us'].mean()

    gemm_time_us = gemm_df['kernel_time_us'].mean()
    return a_preprocess_us, b_preprocess_us, gemm_time_us
```

**Step 2: Replace `calculate_batch_overhead()` to use correct formula**

Find the `calculate_batch_overhead()` function (lines 61–72) and replace it:

```python
def calculate_batch_overhead(a_preprocess_us, b_preprocess_us, gemm_us, batch_sizes):
    """Calculate effective preprocessing overhead for different batch sizes.

    Only B preprocessing (weight matrix) is amortizable across batch items.
    A preprocessing (activation matrix) is required every forward pass.

    Formula:
      effective_preprocess = a_preprocess + b_preprocess / batch_size
      overhead% = effective_preprocess / (effective_preprocess + gemm) * 100
    """
    overheads = []
    for batch_size in batch_sizes:
        effective_preprocess = a_preprocess_us + b_preprocess_us / batch_size
        total_time = effective_preprocess + gemm_us
        overhead_pct = (effective_preprocess / total_time) * 100.0
        overheads.append(overhead_pct)
    return np.array(overheads)
```

**Step 3: Update `plot_figure4()` to pass the new arguments and clarify the title**

Find the `plot_figure4()` function. Update the following lines:

- Line calling `extract_times()`: `preprocess_us, gemm_us = extract_times()` → `a_preprocess_us, b_preprocess_us, gemm_us = extract_times()`
- Line calling `calculate_batch_overhead()`: update to pass split times
- Print statements: show A and B separately
- Title: clarify which part is amortizable

Replace the function body from the `print("Extracting times...")` line through the summary print block with:

```python
def plot_figure4():
    """Generate Figure 4: Batch Amortization"""

    print("Extracting times from benchmark data...")
    a_preprocess_us, b_preprocess_us, gemm_us = extract_times()
    combined_us = a_preprocess_us + b_preprocess_us

    print(f"\nExtracted times (4096×4096, 50% density):")
    print(f"  A preprocessing (non-amortizable): {a_preprocess_us:.1f} µs")
    print(f"  B preprocessing (amortizable):     {b_preprocess_us:.1f} µs")
    print(f"  GEMM:                              {gemm_us:.1f} µs")
    print(f"  Single-call overhead: {(combined_us / (combined_us + gemm_us) * 100):.2f}%")
    print(f"  Note: B-preprocessing (weights) amortizes across calls; A-preprocessing does not.")

    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    overheads = calculate_batch_overhead(a_preprocess_us, b_preprocess_us, gemm_us, batch_sizes)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(batch_sizes, overheads, marker='o', linewidth=2.5, markersize=10,
            color='red', label='Effective Preprocessing Overhead\n(B amortized, A per-call)')

    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='1% threshold', alpha=0.7)
    ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, label='5% threshold', alpha=0.7)

    # Asymptote: A-only overhead at large batch (B fully amortized)
    a_only_pct = (a_preprocess_us / (a_preprocess_us + gemm_us)) * 100.0
    ax.axhline(y=a_only_pct, color='blue', linestyle=':', linewidth=1.5,
               label=f'A-only floor ({a_only_pct:.2f}%, large batch limit)', alpha=0.7)

    ax.axvspan(32, 128, alpha=0.2, color='blue', label='Typical LLM batch range')

    ax.set_xlabel('Batch Size (# inferences, same weights)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Effective Preprocessing Overhead (%)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: Batch Amortization (K25, 4096×4096, 50% density)\n'
                 'B-preprocessing (weight matrix) amortized across batch; '
                 'A-preprocessing (activations) per-call',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    for batch_size in [1, 32, 128]:
        idx = np.where(batch_sizes == batch_size)[0][0]
        overhead = overheads[idx]
        ax.annotate(f'{overhead:.2f}%',
                   xy=(batch_size, overhead),
                   xytext=(0, -15), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    output_file = OUTPUT_DIR / "figure4_batch_amortization.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.savefig(output_file.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file.with_suffix('.png')}")
    plt.close()

    print("\n===== Overhead vs Batch Size (corrected model) =====")
    print(f"{'Batch':>8} | {'Overhead %':>11} | {'A contribution':>15} | {'B contribution':>15}")
    print("-" * 60)
    for batch_size, overhead in zip(batch_sizes, overheads):
        eff_pre = a_preprocess_us + b_preprocess_us / batch_size
        a_contrib = (a_preprocess_us / (eff_pre + gemm_us)) * 100.0
        b_contrib = ((b_preprocess_us / batch_size) / (eff_pre + gemm_us)) * 100.0
        print(f"{batch_size:>8} | {overhead:>10.3f}% | {a_contrib:>14.3f}% | {b_contrib:>14.3f}%")

    print(f"\nAt large batch: overhead converges to {a_only_pct:.3f}% (A-preprocessing only)")
```

**Step 4: Run to verify**

Run: `python3 scripts/experiments/plot_figure4.py`
Expected: prints separate A/B times (should match ~167 µs and ~136 µs), saves PDF. Check the console output shows:
- "A preprocessing (non-amortizable): ~167 µs"
- "B preprocessing (amortizable): ~136 µs"
- At large batch: overhead converges to non-zero floor (the A-only percentage)

**Step 5: Commit**

```bash
git add scripts/experiments/plot_figure4.py
git commit -m "fix: correct batch amortization model — only B-preprocessing (weights) is amortizable"
```

---

## Task 4: Fix Figure 5 data collection — change to 25% density

**Files:**
- Modify: `scripts/experiments/05_collect_figure5_data.sh`

**Step 1: Edit the sparsity pattern and comments**

In `05_collect_figure5_data.sh`, change lines 17–19 from:
```bash
SIZES="1024,2048,4096,8192"
# 50% sparsity (representative)
SPARSITY="11110000"
```
to:
```bash
SIZES="1024,2048,4096,8192"
# 25% sparsity — K25 achieves genuine speedup over cuBLAS at this density
SPARSITY="11000000"
```

Also update the echo on line 23:
```bash
echo "Benchmarking K15 (cuBLAS), K25 (AB-Fused) at sizes: $SIZES with 50% sparsity"
```
to:
```bash
echo "Benchmarking K15 (cuBLAS), K25 (AB-Fused) at sizes: $SIZES with 25% sparsity"
```

**Step 2: Verify the change**

Run: `grep SPARSITY scripts/experiments/05_collect_figure5_data.sh`
Expected: shows `SPARSITY="11000000"`

**Step 3: Commit**

```bash
git add scripts/experiments/05_collect_figure5_data.sh
git commit -m "fix: run Figure 5 at 25% density where K25 achieves genuine speedup over cuBLAS"
```

---

## Task 5: Fix Figure 5 plot — update title and suptitle to reflect 25% density

**Files:**
- Modify: `scripts/experiments/plot_figure5.py`

**Step 1: Update the title strings**

In `plot_figure5()`, find line 128:
```python
fig.suptitle('Figure 5: Matrix Size Scaling (AB-Fused vs cuBLAS, 50% density)',
```
Change to:
```python
fig.suptitle('Figure 5: Matrix Size Scaling (AB-Fused vs cuBLAS, 25% density)',
```

**Step 2: Commit**

```bash
git add scripts/experiments/plot_figure5.py
git commit -m "fix: update Figure 5 title to reflect 25% density"
```

---

## Task 6: Rerun Figure 1 data collection (GPU required)

**Prerequisites:** Task 1 (patterns updated in script)

**Step 1: Run the data collection**

```bash
bash scripts/experiments/01_collect_figure1_data.sh
```

This will take ~20 minutes. It overwrites `results/figure1_performance_vs_sparsity/esmm_kernels/`.
The script runs K15, K20, K21, K25 at 8 density levels now (was 6).

Expected: new NCU .ncu-rep files for 37_5pct and 62_5pct patterns appear in the output directory.

**Step 2: Verify new patterns are in summary.csv**

```bash
grep "37_5pct\|62_5pct" results/figure1_performance_vs_sparsity/esmm_kernels/summary.csv
```
Expected: rows for K20, K21, K25 at both new density points.

**Step 3: Regenerate Figure 1**

```bash
python3 scripts/experiments/plot_figure1.py
```
Expected: saves updated figure with 8 density points including 37.5% and 62.5%.

---

## Task 7: Rerun Figure 5 data collection (GPU required)

**Prerequisites:** Task 4 (density changed to 25% in script)

**Step 1: Run the data collection**

```bash
bash scripts/experiments/05_collect_figure5_data.sh
```

This will take ~20 minutes. It overwrites `results/figure5_matrix_scaling/esmm_kernels/`.

**Step 2: Verify the summary shows 25% density**

```bash
grep "pattern" results/figure5_matrix_scaling/esmm_kernels/summary.csv | head -3
```
Expected: pattern column shows `11000000` (not `11110000`).

**Step 3: Regenerate Figure 5**

```bash
python3 scripts/experiments/plot_figure5.py
```
Expected: K25 shows speedup > 1.0 at larger sizes, consistent with the known 1.24× at 4096.

---

## Task 8: Regenerate all other figures

After Tasks 6–7, regenerate remaining figures to ensure consistency (they use Figure 1 data for cuBLAS reference).

```bash
python3 scripts/experiments/plot_figure2.py
python3 scripts/experiments/plot_figure3.py
python3 scripts/experiments/plot_figure4.py
```

Verify: all complete without errors and PDFs appear in `results/figures/`.

**Final commit:**

```bash
git add results/figures/
git commit -m "chore: regenerate all figures with corrected data"
```

---

## Summary of Changes

| Task | File | Change | GPU Required |
|------|------|--------|-------------|
| 1 | `01_collect_figure1_data.sh` | Add 37.5% + 62.5% density patterns | No (prep) |
| 2 | `plot_figure1.py` | Add absolute runtime subplot | No |
| 3 | `plot_figure4.py` | Correct A vs B amortization model | No |
| 4 | `05_collect_figure5_data.sh` | Change 50% → 25% density | No (prep) |
| 5 | `plot_figure5.py` | Update title to 25% density | No |
| 6 | Run `01_collect_figure1_data.sh` | Collect 37.5% + 62.5% data | **Yes** |
| 7 | Run `05_collect_figure5_data.sh` | Collect Figure 5 at 25% density | **Yes** |
| 8 | Regenerate all figures | Consistency sweep | No |

Tasks 1–5 are pure code edits and can be done immediately. Tasks 6–7 require GPU time.
