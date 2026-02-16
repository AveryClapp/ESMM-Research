#!/bin/bash
# Experiment 4: Batch Amortization (Figure 4)
# This is arithmetic only - no actual benchmarking needed
# We calculate effective overhead for different batch sizes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure4_batch_amortization"

echo "===== Figure 4: Batch Amortization ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "This experiment uses arithmetic calculation only."
echo "We need preprocessing time and GEMM time from K25 at 4096×4096, 50% sparsity"
echo ""

# Check if we have the data from Figure 2
FIGURE2_DATA="$PROJECT_ROOT/results/figure2_preprocessing_overhead/k25_scaling"
if [ -d "$FIGURE2_DATA" ]; then
  echo "Using existing data from Figure 2 experiment..."
  # Find the summary.csv for size 4096
  SUMMARY_FILE=$(find "$FIGURE2_DATA" -name "summary.csv" | head -1)

  if [ -f "$SUMMARY_FILE" ]; then
    echo "Found: $SUMMARY_FILE"
    echo "Extracting preprocessing and GEMM times for size=4096..."

    # Copy relevant data
    cp "$SUMMARY_FILE" "$OUTPUT_DIR/k28_4096_times.csv"
    echo "Data ready for plotting."
  else
    echo "ERROR: summary.csv not found in $FIGURE2_DATA"
    echo "Please run 02_collect_figure2_data.sh first."
    exit 1
  fi
else
  echo "Figure 2 data not found. Running minimal benchmark for K25 at 4096×4096..."

  cd "$PROJECT_ROOT"
  python3 scripts/benchmark.py \
    --kernel 25 \
    --sizes 4096 \
    --sparsity "11110000" \
    --cold-start \
    --parallel 1 \
    -o "$OUTPUT_DIR/k25_reference"
fi

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure4.py"
echo "  2. This will calculate: overhead = preprocess / (preprocess + GEMM * batch_size)"
echo "  3. Output will be: results/figures/figure4_batch_amortization.pdf"
