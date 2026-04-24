#!/bin/bash
# Run q_proj × k_proj benchmark across all pruner/group/perm/sparsity combinations.
# Each sparsity dir is benchmarked independently; results are merged at the end.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/ab_qk_sweep"
FINAL="$REPO_ROOT/results/ab_real_qk_all.csv"

mkdir -p "$OUT_DIR"

DIRS=$(find "$REPO_ROOT/weight_permutations" -type d -name "sparsity_*" | sort)
TOTAL=$(echo "$DIRS" | wc -l)
COUNT=0

for DIR in $DIRS; do
    COUNT=$((COUNT + 1))
    LABEL=$(echo "$DIR" | sed "s|$REPO_ROOT/weight_permutations/||")
    SLUG=$(echo "$LABEL" | tr '/' '_')
    OUT_FILE="$OUT_DIR/${SLUG}.csv"

    echo "[$COUNT/$TOTAL] $LABEL"

    if [ -f "$OUT_FILE" ]; then
        echo "  [SKIP] already done"
        continue
    fi

    python3.8 "$REPO_ROOT/scripts/benchmark_ab_real.py" \
        --weights-a "$DIR" \
        --weights-b "$DIR" \
        --filter-a q_proj \
        --filter-b k_proj \
        --kernels 15,29 \
        --out "$OUT_FILE"
done

echo ""
echo "Merging CSVs -> $FINAL"
python3.8 - <<'EOF'
import csv, glob, os
from pathlib import Path

out_dir = os.environ.get("OUT_DIR", "results/ab_qk_sweep")
final   = os.environ.get("FINAL",   "results/ab_real_qk_all.csv")

files = sorted(glob.glob(f"{out_dir}/*.csv"))
rows, fieldnames = [], None
for f in files:
    with open(f) as fp:
        reader = csv.DictReader(fp)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)

if not rows:
    print("No rows to merge.")
else:
    with open(final, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(files)} files -> {final}")
EOF
