#!/usr/bin/env python3
"""
Merge two real_weights benchmark CSVs that have different kernel columns.
Joins on (layer, pruner, group_size, perm_type, sparsity, orig_shape).
"""
import csv
import sys
from pathlib import Path

KEY_COLS = ["layer", "pruner", "group_size", "perm_type", "sparsity", "orig_shape",
            "elem_sparsity_pct", "block_sparsity_pct"]

def load(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row[k] for k in KEY_COLS)
            rows[key] = row
    return rows

def merge(path_a, path_b, out_path):
    a = load(path_a)
    b = load(path_b)

    # Determine all columns from both files
    all_keys_a = list(next(iter(a.values())).keys()) if a else []
    all_keys_b = list(next(iter(b.values())).keys()) if b else []
    extra_a = [c for c in all_keys_a if c not in KEY_COLS]
    extra_b = [c for c in all_keys_b if c not in KEY_COLS]
    # Deduplicate extras preserving order
    seen = set()
    extras = []
    for c in extra_a + extra_b:
        if c not in seen:
            extras.append(c)
            seen.add(c)

    all_cols = KEY_COLS + extras
    all_keys = set(a.keys()) | set(b.keys())

    rows_out = []
    for key in sorted(all_keys):
        row = {k: "" for k in all_cols}
        base = (a.get(key) or b.get(key))
        for k in KEY_COLS:
            row[k] = base[k]
        if key in a:
            for c in extra_a:
                row[c] = a[key].get(c, "")
        if key in b:
            for c in extra_b:
                row[c] = b[key].get(c, "")
        rows_out.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Merged {len(rows_out)} rows → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: merge_real_weights_csv.py <csv_a> <csv_b> <out>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
