#!/bin/bash
# Resume script - picks up from Figure 1 cuBLAS step (ESMM kernels already done)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "ESMM Paper: Resuming Experiment Pipeline"
echo "=========================================="
echo ""
echo "Figure 1 ESMM kernels: ALREADY DONE (24/24)"
echo "Resuming from: Figure 1 cuBLAS baseline"
echo ""

cd "$PROJECT_ROOT"

SIZE=4096

# ============================================================
# Figure 1: cuBLAS + cuSPARSE baselines (ESMM kernels done)
# ============================================================
echo "===== Figure 1: Running cuBLAS baseline ====="
OUTPUT_DIR="$PROJECT_ROOT/results/figure1_performance_vs_sparsity"

python3 scripts/run_cublas_baseline.py --size $SIZE --output "$OUTPUT_DIR/cublas_baseline.csv"

echo ""
echo "Step 3: Running cuSPARSE baseline (if available)"
if [ -f "$PROJECT_ROOT/scripts/run_cusparse_baseline.py" ]; then
  python3 scripts/run_cusparse_baseline.py \
    --size $SIZE \
    --sparsity-levels 0.0,0.125,0.25,0.5,0.75,0.875 \
    --output "$OUTPUT_DIR/cusparse_baseline.csv"
else
  echo "WARNING: cuSPARSE baseline not available, skipping."
fi

echo ""
echo "===== Figure 1 data collection complete ====="

# ============================================================
# Figure 2: Preprocessing Overhead
# ============================================================
echo ""
echo "[2/5] Collecting data for Figure 2 (Preprocessing Overhead)..."
bash scripts/experiments/02_collect_figure2_data.sh

# ============================================================
# Figure 3: Granularity Tradeoff
# ============================================================
echo ""
echo "[3/5] Collecting data for Figure 3 (Granularity Tradeoff)..."
bash scripts/experiments/03_collect_figure3_data.sh

# ============================================================
# Figure 4: Batch Amortization
# ============================================================
echo ""
echo "[4/5] Collecting data for Figure 4 (Batch Amortization)..."
bash scripts/experiments/04_collect_figure4_data.sh

# ============================================================
# Figure 5: Matrix Size Scaling
# ============================================================
echo ""
echo "[5/5] Collecting data for Figure 5 (Matrix Scaling)..."
bash scripts/experiments/05_collect_figure5_data.sh

# ============================================================
# Phase 2: Generate all figures
# ============================================================
echo ""
echo "======================================"
echo "Phase 2: Figure Generation"
echo "======================================"
echo ""

mkdir -p "$PROJECT_ROOT/results/figures"

echo "[1/5] Generating Figure 1..."
python3 scripts/experiments/plot_figure1.py

echo ""
echo "[2/5] Generating Figure 2..."
python3 scripts/experiments/plot_figure2.py

echo ""
echo "[3/5] Generating Figure 3..."
python3 scripts/experiments/plot_figure3.py

echo ""
echo "[4/5] Generating Figure 4..."
python3 scripts/experiments/plot_figure4.py

echo ""
echo "[5/5] Generating Figure 5..."
python3 scripts/experiments/plot_figure5.py

echo ""
echo "=========================================="
echo "✓ All experiments complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Raw data:  results/figure*/"
echo "  - Figures:   results/figures/*.pdf"
echo ""
echo "Generated figures:"
ls -lh results/figures/*.pdf 2>/dev/null || echo "  (none yet - check for errors above)"
echo ""
