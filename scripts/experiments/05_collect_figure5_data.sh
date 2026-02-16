#!/bin/bash
# Experiment 5: Matrix Size Scaling (Figure 5 - Optional)
# Compares K17, K28, cuBLAS across matrix sizes at 50% sparsity

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/figure5_matrix_scaling"

echo "===== Figure 5: Matrix Size Scaling ====="
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Matrix sizes
SIZES="1024,2048,4096,8192,16384"
# 50% sparsity (representative)
SPARSITY="11110000"

echo ""
echo "Step 1: Benchmarking K17 (B-only), K25 (MAIN - fused) at sizes: $SIZES with 50% sparsity"
echo "This will take ~20 minutes..."

cd "$PROJECT_ROOT"
python3 scripts/benchmark.py \
  --kernel 17,25 \
  --sizes "$SIZES" \
  --sparsity "$SPARSITY" \
  --cold-start \
  --parallel 1 \
  -o "$OUTPUT_DIR/esmm_kernels"

echo ""
echo "Step 2: Running cuBLAS baseline across sizes (dense)"

if [ -f "$PROJECT_ROOT/scripts/run_cublas_baseline.py" ]; then
  python3 scripts/run_cublas_baseline.py --sizes "$SIZES" --output "$OUTPUT_DIR/cublas_baseline.csv"
else
  echo "WARNING: cuBLAS baseline script not found. Using K10 (dense GEMM) as reference..."

  python3 scripts/benchmark.py \
    --kernel 10 \
    --sizes "$SIZES" \
    --sparsity "11111111" \
    --cold-start \
    --parallel 1 \
    -o "$OUTPUT_DIR/cublas_reference"
fi

echo ""
echo "===== Data collection complete ====="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python3 scripts/experiments/plot_figure5.py"
echo "  2. Output will be: results/figures/figure5_matrix_scaling.pdf"
