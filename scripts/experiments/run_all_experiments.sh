#!/bin/bash
# Master script to run all experiments and generate all figures for the paper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "ESMM Paper: Complete Experiment Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Collect benchmark data for all 5 figures"
echo "  2. Generate PDF/PNG plots for each figure"
echo ""
echo "Expected runtime: ~2 hours (depends on GPU, cold-start enabled)"
echo "Output: results/figures/*.pdf"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

cd "$PROJECT_ROOT"

# Make sure executable is built
if [ ! -f "./exec_dev" ]; then
    echo "ERROR: ./exec_dev not found. Please run 'make dev' first."
    exit 1
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if python3 -c "import pandas, matplotlib, numpy" 2>/dev/null; then
    echo "✓ Python dependencies OK"
else
    echo "WARNING: Missing Python dependencies. Installing..."
    pip install pandas matplotlib numpy || {
        echo "ERROR: Failed to install dependencies. Please run manually:"
        echo "  pip install pandas matplotlib numpy"
        exit 1
    }
    echo "✓ Python dependencies installed"
fi

# Check for NCU (needed for benchmark.py)
echo "Checking for NCU (NVIDIA Nsight Compute)..."
NCU_FOUND=false

if command -v ncu &> /dev/null; then
    echo "✓ NCU found in PATH: $(which ncu)"
    NCU_FOUND=true
elif [ -f "/usr/local/cuda-12.1/bin/ncu" ]; then
    echo "✓ NCU found at /usr/local/cuda-12.1/bin/ncu"
    NCU_FOUND=true
elif [ -f "/usr/local/cuda/bin/ncu" ]; then
    echo "✓ NCU found at /usr/local/cuda/bin/ncu"
    NCU_FOUND=true
fi

if [ "$NCU_FOUND" = false ]; then
    echo "ERROR: NCU not found. Please install CUDA toolkit with Nsight Compute"
    echo "Expected locations: /usr/local/cuda-12.1/bin/ncu or /usr/local/cuda/bin/ncu"
    exit 1
fi

# Make all experiment scripts executable
chmod +x scripts/experiments/*.sh
chmod +x scripts/experiments/*.py

echo ""
echo "======================================"
echo "Phase 1: Data Collection"
echo "======================================"
echo ""

# Figure 1: Performance vs Sparsity
echo "[1/5] Collecting data for Figure 1 (Performance vs Sparsity)..."
bash scripts/experiments/01_collect_figure1_data.sh

echo ""
echo "[2/5] Collecting data for Figure 2 (Preprocessing Overhead)..."
bash scripts/experiments/02_collect_figure2_data.sh

echo ""
echo "[3/5] Collecting data for Figure 3 (Granularity Tradeoff)..."
bash scripts/experiments/03_collect_figure3_data.sh

echo ""
echo "[4/5] Collecting data for Figure 4 (Batch Amortization)..."
bash scripts/experiments/04_collect_figure4_data.sh

echo ""
echo "[5/5] Collecting data for Figure 5 (Matrix Scaling - Optional)..."
read -p "Run Figure 5 experiment? (takes ~30 min) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash scripts/experiments/05_collect_figure5_data.sh
else
    echo "Skipping Figure 5."
fi

echo ""
echo "======================================"
echo "Phase 2: Figure Generation"
echo "======================================"
echo ""

# Generate all plots
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

if [ -d "results/figure5_matrix_scaling" ]; then
    echo ""
    echo "[5/5] Generating Figure 5..."
    python3 scripts/experiments/plot_figure5.py
else
    echo ""
    echo "[5/5] Skipping Figure 5 (no data collected)."
fi

echo ""
echo "=========================================="
echo "✓ All experiments complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Raw data:  results/figure*/"
echo "  - Figures:   results/figures/*.pdf"
echo ""

# List generated figures
echo "Generated figures:"
ls -lh results/figures/*.pdf 2>/dev/null || echo "  (none yet - check for errors above)"

echo ""
echo "Next steps:"
echo "  1. Review figures in results/figures/"
echo "  2. Check raw data in results/figure*/ directories"
echo "  3. Re-run specific experiments if needed (scripts/experiments/XX_*.sh)"
echo ""
