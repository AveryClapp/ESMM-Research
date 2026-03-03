#!/bin/bash
# Experiment 5: Matrix Size Scaling (Figure 5)
# Compares AB-Fused (K25) and cuBLAS (K15) across matrix sizes at 25% sparsity.
# K15 is profiled via NCU alongside K25 for methodology consistency.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure5_matrix_scaling"

echo "===== Figure 5: Matrix Size Scaling ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Matrix sizes
SIZES="1024,2048,4096,8192"
# 25% sparsity — K25 achieves genuine speedup over cuBLAS at this density
SPARSITY="11000000"

echo ""
echo "Benchmarking K15 (cuBLAS), K25 (AB-Fused) at sizes: $SIZES with 25% sparsity"
echo "This will take ~20 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 15,25 \
  --sizes "$SIZES" \
  --sparsity "$SPARSITY" \
  --cold-start \
  --parallel 1 \
  -o "$OUTPUT_DIR/esmm_kernels"

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Note: K15 rows in summary.csv provide the NCU-measured cuBLAS baseline."
echo "      Plot scripts load cuBLAS time from kernel==15 rows for consistency."
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure5.py"
echo "  2. Output will be: results/figures/figure5_matrix_scaling.pdf"
