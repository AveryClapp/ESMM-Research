#!/bin/bash
# Experiment 1: Performance vs Sparsity (Figure 1)
# Collects data for K15 (cuBLAS), K20, K21, K25 across sparsity levels at 4096×4096
# K15 is profiled via NCU alongside ESMM kernels for methodology consistency.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure1_performance_vs_sparsity"

echo "===== Figure 1: Performance vs Sparsity ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Sparsity patterns (8-bit binary)
SPARSITY_PATTERNS="00000000,10000000,11000000,11100000,11110000,11111000,11111100,11111110"
SIZE=4096

echo ""
echo "Benchmarking K15 (cuBLAS), K20 (AB-Separate), K21 (AB-Fine), K25 (AB-Fused) at size=$SIZE"
echo "Patterns: $SPARSITY_PATTERNS"
echo "This will take ~15 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 15,20,21,25 \
  --sizes $SIZE \
  --sparsity "$SPARSITY_PATTERNS" \
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
echo "  1. Run: python3 scripts/experiments/plot_figure1.py"
echo "  2. Output will be: results/figures/figure1_performance_vs_sparsity.pdf"
