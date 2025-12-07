#!/bin/bash
# Run benchmark across all combinations of A and B sparsity patterns
#
# Usage: ./scripts/benchmark_all_ab_combinations.sh [kernel] [size] [options]
# Example: ./scripts/benchmark_all_ab_combinations.sh 25 4096 --cold-start

set -e

KERNEL=${1:-25}
SIZE=${2:-4096}
shift 2 || true
EXTRA_ARGS="$@"

# Define sparsity patterns to test
# Pattern format: "name:binary_pattern"
# Density % = (count of 1s / 8) * 100
declare -a PATTERNS=(
    "100pct:11111111"  # 8/8 = 100% density
    "87pct:11111110"   # 7/8 = 87.5% density
    "75pct:11111100"   # 6/8 = 75% density
    "62pct:11111000"   # 5/8 = 62.5% density
    "50pct:11110000"   # 4/8 = 50% density
    "37pct:11100000"   # 3/8 = 37.5% density
    "25pct:11000000"   # 2/8 = 25% density
    "12pct:10000000"   # 1/8 = 12.5% density
)

echo "========================================"
echo "Benchmarking all A×B sparsity combinations"
echo "Kernel: $KERNEL"
echo "Size: $SIZE"
echo "Extra args: $EXTRA_ARGS"
echo "========================================"
echo ""

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
OUTPUT_BASE="benchmarks/${TIMESTAMP}_k${KERNEL}_ab_grid"
mkdir -p "$OUTPUT_BASE"

# Create CSV to track all runs
SUMMARY_FILE="${OUTPUT_BASE}/combinations_summary.csv"
echo "pattern_a,pattern_b,density_a_pct,density_b_pct,output_dir" > "$SUMMARY_FILE"

TOTAL_COMBINATIONS=$(( ${#PATTERNS[@]} * ${#PATTERNS[@]} ))
CURRENT=0

for pattern_a_entry in "${PATTERNS[@]}"; do
    IFS=':' read -r label_a pattern_a <<< "$pattern_a_entry"
    density_a=$(echo "$pattern_a" | tr -cd '1' | wc -c)
    density_a_pct=$(echo "scale=1; $density_a * 100 / 8" | bc)

    for pattern_b_entry in "${PATTERNS[@]}"; do
        IFS=':' read -r label_b pattern_b <<< "$pattern_b_entry"
        density_b=$(echo "$pattern_b" | tr -cd '1' | wc -c)
        density_b_pct=$(echo "scale=1; $density_b * 100 / 8" | bc)

        CURRENT=$((CURRENT + 1))

        echo ""
        echo "[$CURRENT/$TOTAL_COMBINATIONS] Running A=${label_a} (${density_a_pct}%), B=${label_b} (${density_b_pct}%)"
        echo "  Pattern A: $pattern_a"
        echo "  Pattern B: $pattern_b"

        OUTPUT_DIR="A_${label_a}_B_${label_b}"

        # Run benchmark
        ./scripts/benchmark.py \
            -k "$KERNEL" \
            --sizes "$SIZE" \
            -pa "$pattern_a" \
            -pb "$pattern_b" \
            --blockwise \
            -o "${OUTPUT_BASE}/${OUTPUT_DIR}" \
            $EXTRA_ARGS

        # Log to summary
        echo "${pattern_a},${pattern_b},${density_a_pct},${density_b_pct},${OUTPUT_BASE}/${OUTPUT_DIR}" >> "$SUMMARY_FILE"

        echo "  ✓ Complete: ${OUTPUT_BASE}/${OUTPUT_DIR}"
    done
done

echo ""
echo "========================================"
echo "All combinations complete!"
echo "  Total runs: $TOTAL_COMBINATIONS"
echo "  Output: $OUTPUT_BASE"
echo "  Summary: $SUMMARY_FILE"
echo "========================================"
