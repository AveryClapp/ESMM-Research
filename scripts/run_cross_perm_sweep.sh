#!/bin/bash
# Cross-permutation sweep: A from col-perm dirs × B from row-perm dirs, and vice versa.
# Tests whether mixing permutation types (A_col × B_row, A_row × B_col) beats same-type
# pairing (col×col, row×row) for joint AB sparsity exploitation in ESMM.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/ab_cross_perm_sweep"
FINAL="$REPO_ROOT/results/ab_cross_perm_all.csv"

mkdir -p "$OUT_DIR"

# Iterate over all col_perm sparsity dirs; derive the matching row_perm counterpart by path substitution.
COL_DIRS=$(find "$REPO_ROOT/weight_permutations" -type d -name "sparsity_*" | grep "col_perm" | sort)
TOTAL=$(echo "$COL_DIRS" | wc -l)
COUNT=0

for COL_DIR in $COL_DIRS; do
    ROW_DIR=$(echo "$COL_DIR" | sed 's/col_perm/row_perm/')

    if [ ! -d "$ROW_DIR" ]; then
        echo "[SKIP] No matching row_perm dir for $COL_DIR"
        continue
    fi

    COUNT=$((COUNT + 1))
    LABEL=$(echo "$COL_DIR" | sed "s|$REPO_ROOT/weight_permutations/||")

    # Slug for output files
    BASE_SLUG=$(echo "$LABEL" | tr '/' '_' | sed 's/_col_perm//')

    OUT_A_COL="$OUT_DIR/${BASE_SLUG}_Acol_Brow.csv"
    OUT_A_ROW="$OUT_DIR/${BASE_SLUG}_Arow_Bcol.csv"

    echo "[$COUNT/$TOTAL] $LABEL"

    # A=col_perm (q_proj), B=row_perm (k_proj)
    if [ -f "$OUT_A_COL" ]; then
        echo "  [SKIP] A_col×B_row already done"
    else
        echo "  Running A_col × B_row ..."
        python3.8 "$REPO_ROOT/scripts/benchmark_ab_real.py" \
            --weights-a "$COL_DIR" \
            --weights-b "$ROW_DIR" \
            --filter-a q_proj \
            --filter-b k_proj \
            --kernels 15,29 \
            --out "$OUT_A_COL"
    fi

    # A=row_perm (q_proj), B=col_perm (k_proj)
    if [ -f "$OUT_A_ROW" ]; then
        echo "  [SKIP] A_row×B_col already done"
    else
        echo "  Running A_row × B_col ..."
        python3.8 "$REPO_ROOT/scripts/benchmark_ab_real.py" \
            --weights-a "$ROW_DIR" \
            --weights-b "$COL_DIR" \
            --filter-a q_proj \
            --filter-b k_proj \
            --kernels 15,29 \
            --out "$OUT_A_ROW"
    fi
done

echo ""
echo "Merging CSVs -> $FINAL"
python3.8 - <<'EOF'
import csv, glob, os

out_dir = os.environ.get("OUT_DIR", "results/ab_cross_perm_sweep")
final   = os.environ.get("FINAL",   "results/ab_cross_perm_all.csv")

files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
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
    print("Merged {} rows from {} files -> {}".format(len(rows), len(files), final))
EOF
