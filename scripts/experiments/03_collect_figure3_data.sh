#!/bin/bash
# Experiment 3: Granularity Tradeoff (Figure 3)
# Compares K24 (64-row) vs K28 (8-row) across sparsity levels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure3_granularity_tradeoff"

echo "===== Figure 3: Fusion vs Granularity Tradeoff ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Full sparsity range: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
# Compare fusion approach (K25) vs fine granularity (K21) vs separate (K20)
SPARSITY_PATTERNS="00000000,10000000,11000000,11100000,11110000,11111000,11111100,11111110"
SIZE=4096

echo ""
echo "Benchmarking K20 (64-row separate), K21 (8×32), K25 (64×32 FUSED) at size=$SIZE"
echo "Patterns: $SPARSITY_PATTERNS"
echo "This will take ~25 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 20,21,25 \
  --sizes $SIZE \
  --sparsity "$SPARSITY_PATTERNS" \
  --cold-start \
  --parallel 1 \
  -o "$OUTPUT_DIR/fusion_vs_granularity"

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure3.py"
echo "  2. Output will be: results/figures/figure3_granularity_tradeoff.pdf"
echo ""
echo "Note: Metadata size is calculated from formulas, not measured"
echo "  K24: (M/64) × (K/8) bytes for A patterns"
echo "  K28: (M/8) × (K/8) bytes for A patterns"
