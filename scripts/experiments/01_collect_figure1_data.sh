#!/bin/bash
# Experiment 1: Performance vs Sparsity (Figure 1)
# Collects data for K17, K24, K28 across sparsity levels at 4096×4096

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure1_performance_vs_sparsity"

echo "===== Figure 1: Performance vs Sparsity ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Sparsity patterns (8-bit binary)
# 0% = 00000000, 12.5% = 10000000, 25% = 11000000, 50% = 11110000, 75% = 11111100, 87.5% = 11111110
SPARSITY_PATTERNS="00000000,10000000,11000000,11110000,11111100,11111110"
SIZE=4096

echo ""
echo "Step 1: Benchmarking K17 (B-only), K20 (separate), K21 (8×32), K25 (MAIN - fused) at size=$SIZE"
echo "Patterns: $SPARSITY_PATTERNS"
echo "This will take ~15 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 17,20,21,25 \
  --sizes $SIZE \
  --sparsity "$SPARSITY_PATTERNS" \
  --cold-start \
  --parallel 1 \
  -o "$OUTPUT_DIR/esmm_kernels"

echo ""
echo "Step 2: Running cuBLAS baseline (dense only)"
echo "cuBLAS doesn't exploit sparsity, so we run once and use same time for all sparsity levels"

# Run cuBLAS via kernel 10 (which internally uses cuBLAS for verification reference)
# Or we need a standalone cuBLAS runner - check if exists
if [ -f "$PROJECT_ROOT/scripts/run_cublas_baseline.py" ]; then
  python3 scripts/run_cublas_baseline.py --size $SIZE --output "$OUTPUT_DIR/cublas_baseline.csv"
else
  echo "WARNING: cuBLAS baseline script not found. You'll need to run cuBLAS separately."
  echo "Expected: scripts/run_cublas_baseline.py"
  echo "For now, using K10 (dense GEMM) as reference..."

  python3 scripts/benchmark.py \
    --kernel 10 \
    --sizes $SIZE \
    --sparsity "11111111" \
    --cold-start \
    --parallel 1 \
    -o "$OUTPUT_DIR/cublas_reference"
fi

echo ""
echo "Step 3: Running cuSPARSE baseline (if available)"
if [ -f "$PROJECT_ROOT/scripts/run_cusparse_baseline.py" ]; then
  # Run cuSPARSE at each sparsity level (include format conversion time)
  python3 scripts/run_cusparse_baseline.py \
    --size $SIZE \
    --sparsity-levels 0.0,0.125,0.25,0.5,0.75,0.875 \
    --output "$OUTPUT_DIR/cusparse_baseline.csv"
else
  echo "WARNING: cuSPARSE baseline script not found."
  echo "Expected: scripts/run_cusparse_baseline.py"
  echo "Skipping cuSPARSE comparison for now."
fi

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure1.py"
echo "  2. Output will be: results/figures/figure1_performance_vs_sparsity.pdf"
