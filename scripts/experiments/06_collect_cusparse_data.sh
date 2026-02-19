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

# Write to a temp file; move into place only on clean completion to avoid
# leaving a partial CSV if the benchmark fails mid-run.
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

# Write header (explicit error message if --header invocation fails)
"$BENCH" --header > "$TMP_FILE" || { echo "ERROR: --header invocation failed" >&2; exit 1; }

total=$(( ${#SIZES[@]} * ${#DENSITIES[@]} ))
count=0

for SIZE in "${SIZES[@]}"; do
    for DENSITY in "${DENSITIES[@]}"; do
        count=$(( count + 1 ))
        echo "[${count}/${total}] size=${SIZE} density=${DENSITY}" >&2
        "$BENCH" --size "$SIZE" --density "$DENSITY" --runs "$RUNS" >> "$TMP_FILE"
    done
done

mv "$TMP_FILE" "$OUTPUT_FILE"
echo "Done. Results in: $OUTPUT_FILE" >&2
