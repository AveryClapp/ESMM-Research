#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$PROJECT_ROOT/cusparse_bench"
OUTPUT_FILE="$PROJECT_ROOT/results/cusparse_benchmark/cusparse_results.csv"
RUNS=20

# Verify binary exists
if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: cusparse_bench binary not found or not executable at: $BENCH" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

SIZES=(1024 2048 4096)
DENSITIES=(0.125 0.25 0.5)

# Write header fresh (overwrite any existing file)
"$BENCH" --header > "$OUTPUT_FILE"

total=${#SIZES[@]}
total=$(( total * ${#DENSITIES[@]} ))
count=0

for SIZE in "${SIZES[@]}"; do
    for DENSITY in "${DENSITIES[@]}"; do
        count=$(( count + 1 ))
        echo "[${count}/${total}] size=${SIZE} density=${DENSITY}" >&2
        "$BENCH" --size "$SIZE" --density "$DENSITY" --runs "$RUNS" >> "$OUTPUT_FILE"
    done
done

echo "Done. Results in: $OUTPUT_FILE" >&2
