#!/usr/bin/env python3
"""
Benchmark ESMM kernels vs cuBLAS on real LLM weight pairs (both A and B loaded).

Finds all compatible A×B pairs from two directories, benchmarks each under NCU,
and writes results to CSV.

Usage:
  python3 scripts/benchmark_ab_real.py \
    --weights-a weight_permutations/.../sparsity_0.99/ \
    --weights-b weight_permutations/.../sparsity_0.99/ \
    --kernels 15,29
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
_candidates = ["exec_prod", "exec_dev", "exec"]
EXEC = next((REPO_ROOT / e for e in _candidates if (REPO_ROOT / e).exists()), REPO_ROOT / "exec_dev")

NCU_PATH = "/usr/local/cuda-12.1/bin/ncu"
NCU_ENV = {**os.environ, "LD_LIBRARY_PATH": f"/usr/local/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"}

BM = 32
BK = 8
MAX_K = 1024 * BK  # 8192 — K29 smem limit


def pad_to_tile(arr: np.ndarray) -> np.ndarray:
    M, K = arr.shape
    M_pad = ((M + BM - 1) // BM) * BM
    K_pad = ((K + BK - 1) // BK) * BK
    if M_pad == M and K_pad == K:
        return arr
    out = np.zeros((M_pad, K_pad), dtype=arr.dtype)
    out[:M, :K] = arr
    return out


def compute_block_sparsity(arr: np.ndarray) -> float:
    M, K = arr.shape
    tiles = arr.reshape(M // BM, BM, K // BK, BK)
    all_zero = np.abs(tiles).max(axis=(1, 3)) == 0
    return float(all_zero.mean())


def parse_config(pt_path: Path) -> dict:
    parts = pt_path.parts
    config = {"pruner": "", "group_size": "", "perm_type": "", "sparsity": ""}
    for p in parts:
        if p in ("wanda", "sparsegpt"):
            config["pruner"] = p
        if "grp_8" in p:
            config["group_size"] = "8"
        elif "grp_16" in p:
            config["group_size"] = "16"
        if "col_perm" in p or "columns_permuted" in p:
            config["perm_type"] = "col"
        elif "row_perm" in p or "rows_permuted" in p:
            config["perm_type"] = "row"
        if "sparsity_" in p:
            config["sparsity"] = p.split("sparsity_")[1].rstrip("/")
    return config


def _parse_csv_line(line: str) -> list:
    """Parse a single CSV line handling quoted commas."""
    parts = []
    current = ""
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            parts.append(current.strip('"'))
            current = ""
            continue
        current += char
    parts.append(current.strip('"'))
    return parts


def find_compatible_pairs(a_files: list, b_files: list) -> list:
    """Return (a_path, b_path) tuples where A.cols == B.rows and K <= MAX_K."""
    b_shapes = {}
    for b_path in b_files:
        try:
            b_obj = torch.load(b_path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [SKIP B] {b_path.name}: {e}")
            continue
        if not isinstance(b_obj, torch.Tensor) or b_obj.dim() != 2:
            print(f"  [SKIP B] {b_path.name}: not a 2D tensor")
            continue
        b_shapes[b_path] = b_obj.shape

    pairs = []
    for a_path in a_files:
        try:
            a_obj = torch.load(a_path, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if not isinstance(a_obj, torch.Tensor) or a_obj.dim() != 2:
            continue
        a_K = a_obj.shape[1]
        if a_K > MAX_K:
            print(f"  [SKIP A] {a_path.name}: K={a_K} > {MAX_K}")
            continue
        for b_path, b_shape in b_shapes.items():
            if a_path == b_path:
                continue
            if b_shape[0] == a_K:
                pairs.append((a_path, b_path))
    return pairs


def run_pair(kernel: int, a_bin: str, b_bin: str, rows: int, inners: int, cols: int):
    """Run kernel under NCU with both A and B loaded from binary files.

    Returns (compute_ms, preprocess_ms) or (None, None) on failure.
    """
    tmp_dir = tempfile.mkdtemp()
    ncu_base = os.path.join(tmp_dir, "profile")
    ncu_rep = ncu_base + ".ncu-rep"

    ncu_cmd = [
        "sudo", NCU_PATH,
        "--set", "basic",
        "--target-processes", "all",
        "--export", ncu_base,
        "--force-overwrite",
        str(EXEC), str(kernel),
        "--load-a", a_bin,
        "--load-b", b_bin,
        "--dims", str(rows), str(inners), str(cols),
        "--no-check",
    ]

    try:
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300, env=NCU_ENV)
        if result.returncode != 0:
            print(f"    [NCU ERROR] kernel={kernel}: {result.stderr[:300]}")
            return None, None

        import_cmd = [
            "sudo", NCU_PATH,
            "--import", ncu_rep,
            "--csv", "--page", "details",
        ]
        imp = subprocess.run(import_cmd, capture_output=True, text=True, timeout=60, env=NCU_ENV)
        if imp.returncode != 0:
            print(f"    [NCU IMPORT ERROR] kernel={kernel}: {imp.stderr[:200]}")
            return None, None

        compute_us = 0.0
        preprocess_us = 0.0

        lines = imp.stdout.strip().split("\n")
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = _parse_csv_line(line)
            if len(parts) < 15:
                continue

            kernel_name = parts[4]
            metric_name = parts[12]
            metric_unit = parts[13]
            metric_value_str = parts[14]

            if metric_name != "Duration":
                continue

            try:
                val = float(metric_value_str.replace(",", ""))
            except ValueError:
                continue

            if metric_unit == "second":
                val_us = val * 1e6
            elif metric_unit == "msecond":
                val_us = val * 1e3
            else:
                val_us = val  # usecond

            kn_lower = kernel_name.lower()
            if "preprocess" in kn_lower or "analyze" in kn_lower:
                preprocess_us += val_us
            else:
                compute_us += val_us

        if compute_us == 0.0:
            print(f"    [WARN] No Duration for kernel={kernel}. stdout[:300]:\n{imp.stdout[:300]}")
            return None, None

        return compute_us / 1000.0, preprocess_us / 1000.0

    except subprocess.TimeoutExpired:
        print(f"    [NCU TIMEOUT] kernel={kernel}")
        return None, None
    except Exception as e:
        print(f"    [ERROR] kernel={kernel}: {e}")
        return None, None
    finally:
        try:
            os.unlink(ncu_rep)
        except OSError:
            pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def benchmark_pair(a_path: Path, b_path: Path, kernels: list):
    """Load, pad, and benchmark one A×B weight pair. Returns a CSV row dict or None."""
    a_cfg = parse_config(a_path)
    b_cfg = parse_config(b_path)

    a_obj = torch.load(a_path, map_location="cpu", weights_only=True)
    b_obj = torch.load(b_path, map_location="cpu", weights_only=True)

    if not isinstance(a_obj, torch.Tensor) or a_obj.dim() != 2:
        return None
    if not isinstance(b_obj, torch.Tensor) or b_obj.dim() != 2:
        return None

    a_arr = pad_to_tile(a_obj.float().numpy())
    b_arr = pad_to_tile(b_obj.float().numpy())

    a_orig = tuple(a_obj.shape)
    b_orig = tuple(b_obj.shape)
    M_pad, K_pad = a_arr.shape
    _, N_pad = b_arr.shape

    a_elem_sp = float((a_obj == 0).float().mean())
    b_elem_sp = float((b_obj == 0).float().mean())
    a_block_sp = compute_block_sparsity(a_arr)
    b_block_sp = compute_block_sparsity(b_arr)

    print(f"\n{a_path.stem[:35]} × {b_path.stem[:35]}")
    print(f"  A: shape={a_orig}  elem={a_elem_sp*100:.1f}%  block={a_block_sp*100:.1f}%")
    print(f"  B: shape={b_orig}  elem={b_elem_sp*100:.1f}%  block={b_block_sp*100:.1f}%")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fa:
        a_tmp = fa.name
        a_arr.astype(np.float32).tofile(fa)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fb:
        b_tmp = fb.name
        b_arr.astype(np.float32).tofile(fb)

    try:
        timings = {}
        for k in kernels:
            compute_ms, preprocess_ms = run_pair(k, a_tmp, b_tmp, M_pad, K_pad, N_pad)
            timings[k] = (compute_ms, preprocess_ms)
            label_k = "cuBLAS" if k == 15 else f"K{k}"
            if compute_ms is not None:
                total = compute_ms + (preprocess_ms or 0.0)
                print(f"  {label_k}: compute={compute_ms:.3f}ms  "
                      f"preprocess={preprocess_ms:.3f}ms  total={total:.3f}ms")
            else:
                print(f"  {label_k}: FAILED")
    finally:
        os.unlink(a_tmp)
        os.unlink(b_tmp)

    row = {
        "a_file": a_path.stem,
        "b_file": b_path.stem,
        "a_pruner": a_cfg["pruner"],
        "b_pruner": b_cfg["pruner"],
        "a_sparsity": a_cfg["sparsity"],
        "b_sparsity": b_cfg["sparsity"],
        "a_block_sparsity_pct": f"{a_block_sp*100:.1f}",
        "b_block_sparsity_pct": f"{b_block_sp*100:.1f}",
        "a_shape": f"{a_orig[0]}x{a_orig[1]}",
        "b_shape": f"{b_orig[0]}x{b_orig[1]}",
    }

    for k in kernels:
        compute_ms, preprocess_ms = timings[k]
        row[f"k{k}_compute_ms"] = f"{compute_ms:.3f}" if compute_ms is not None else "N/A"
        row[f"k{k}_preprocess_ms"] = f"{preprocess_ms:.3f}" if preprocess_ms is not None else "N/A"

    if 15 in kernels and 29 in kernels:
        c15, _ = timings.get(15, (None, None))
        c29, p29 = timings.get(29, (None, None))
        if c15 and c29:
            row["k29_compute_speedup"] = f"{c15/c29:.3f}"
            total29 = c29 + (p29 or 0.0)
            row["k29_total_speedup"] = f"{c15/total29:.3f}"
        else:
            row["k29_compute_speedup"] = "N/A"
            row["k29_total_speedup"] = "N/A"

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ESMM on real A×B LLM weight pairs."
    )
    parser.add_argument("--weights-a", required=True,
                        help="Dir containing *_permuted.pt files for A matrices")
    parser.add_argument("--weights-b", required=True,
                        help="Dir containing *_permuted.pt files for B matrices")
    parser.add_argument("--kernels", default="15,29",
                        help="Comma-separated kernel IDs (default: 15,29)")
    parser.add_argument("--filter-a", default=None,
                        help="Only use A files whose name contains this substring")
    parser.add_argument("--filter-b", default=None,
                        help="Only use B files whose name contains this substring")
    parser.add_argument("--out", default="results/ab_real_benchmark.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    kernels = [int(k) for k in args.kernels.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not EXEC.exists():
        print(f"Error: exec not found at {EXEC}. Run 'make release'.")
        sys.exit(1)
    if not Path(NCU_PATH).exists():
        print(f"Error: ncu not found at {NCU_PATH}.")
        sys.exit(1)

    a_files = sorted(Path(args.weights_a).rglob("*_permuted.pt"))
    b_files = sorted(Path(args.weights_b).rglob("*_permuted.pt"))

    if args.filter_a:
        a_files = [p for p in a_files if args.filter_a in p.name]
    if args.filter_b:
        b_files = [p for p in b_files if args.filter_b in p.name]

    if not a_files:
        print(f"No A files found in {args.weights_a}" +
              (f" matching '{args.filter_a}'" if args.filter_a else ""))
        sys.exit(1)
    if not b_files:
        print(f"No B files found in {args.weights_b}" +
              (f" matching '{args.filter_b}'" if args.filter_b else ""))
        sys.exit(1)

    pairs = find_compatible_pairs(a_files, b_files)
    if not pairs:
        print("No compatible A×B pairs found (check shapes and K limit).")
        sys.exit(1)

    print(f"A files: {len(a_files)} | B files: {len(b_files)} | "
          f"Compatible pairs: {len(pairs)} | kernels={kernels} | exec={EXEC.name} | ncu=ON")
    print("─" * 100)

    rows_out = []
    fieldnames = None
    for a_path, b_path in pairs:
        row = benchmark_pair(a_path, b_path, kernels)
        if row:
            rows_out.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)

    if 15 in kernels and 29 in kernels:
        compute_speedups, total_speedups = [], []
        for r in rows_out:
            try:
                compute_speedups.append(float(r["k29_compute_speedup"]))
            except (KeyError, ValueError):
                pass
            try:
                total_speedups.append(float(r["k29_total_speedup"]))
            except (KeyError, ValueError):
                pass
        if compute_speedups:
            print(f"\n{'─'*60}")
            print(f"K29 vs cuBLAS — {len(compute_speedups)} pairs (NCU timing):")
            print(f"  Compute-only: min={min(compute_speedups):.2f}x  "
                  f"max={max(compute_speedups):.2f}x  "
                  f"mean={sum(compute_speedups)/len(compute_speedups):.2f}x")
            print(f"  Total (incl. preprocess): min={min(total_speedups):.2f}x  "
                  f"max={max(total_speedups):.2f}x  "
                  f"mean={sum(total_speedups)/len(total_speedups):.2f}x")

    print(f"\nResults saved to {out_path}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
