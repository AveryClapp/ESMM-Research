#!/bin/bash
# Resume from Figure 3 - Figures 1 and 2 data already collected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "ESMM Paper: Resuming from Figure 3"
echo "=========================================="
echo "Figure 1: DONE (24/24 ESMM + cuBLAS baseline)"
echo "Figure 2: DONE (4 sizes: 1024, 2048, 4096, 8192)"
echo ""

cd "$PROJECT_ROOT"

run_step() {
    local desc="$1"
    shift
    echo ""
    echo ">>> $desc"
    "$@" || echo "WARNING: '$desc' had partial failures, continuing..."
}

# Figure 3: Fusion vs Granularity
run_step "[3/5] Figure 3: Granularity Tradeoff" \
    bash scripts/experiments/03_collect_figure3_data.sh

# Figure 4: Batch Amortization
run_step "[4/5] Figure 4: Batch Amortization" \
    bash scripts/experiments/04_collect_figure4_data.sh

# Figure 5: Matrix Size Scaling
run_step "[5/5] Figure 5: Matrix Scaling" \
    bash scripts/experiments/05_collect_figure5_data.sh

# ============================================================
# Generate all figures
# ============================================================
echo ""
echo "======================================"
echo "Phase 2: Figure Generation"
echo "======================================"

mkdir -p "$PROJECT_ROOT/results/figures"

run_step "[1/5] Generating Figure 1" python3 scripts/experiments/plot_figure1.py
run_step "[2/5] Generating Figure 2" python3 scripts/experiments/plot_figure2.py
run_step "[3/5] Generating Figure 3" python3 scripts/experiments/plot_figure3.py
run_step "[4/5] Generating Figure 4" python3 scripts/experiments/plot_figure4.py
run_step "[5/5] Generating Figure 5" python3 scripts/experiments/plot_figure5.py

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo ""
echo "Generated figures:"
ls -lh results/figures/*.pdf 2>/dev/null || echo "  (check for errors above)"
