#!/bin/bash
# Experiment 2: Preprocessing Overhead (Figure 2)
# Collects K28 preprocessing + GEMM times across matrix sizes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure2_preprocessing_overhead"

echo "===== Figure 2: Preprocessing Overhead ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Matrix sizes
SIZES="1024,2048,4096,8192,16384"
# 50% sparsity (representative)
SPARSITY="11110000"

echo ""
echo "Benchmarking K25 (MAIN - fused) at sizes: $SIZES with 50% sparsity"
echo "This will take ~20 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 25 \
  --sizes "$SIZES" \
  --sparsity "$SPARSITY" \
  --cold-start \
  --parallel 1 \
  -o "$OUTPUT_DIR/k25_scaling"

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Note: benchmark.py already separates preprocessing time from GEMM time"
echo "The summary.csv will have both PREPROCESS and main kernel times"
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure2.py"
echo "  2. Output will be: results/figures/figure2_preprocessing_overhead.pdf"
