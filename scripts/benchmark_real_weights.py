#!/usr/bin/env python3
"""
Benchmark ESMM kernels vs cuBLAS on real LLM weight tensors.

Loads .pt files directly, writes a temp binary per tensor, benchmarks with NCU,
then deletes. No persistent binary storage needed — only ~200MB temp space at a time.

Usage:
  python3 scripts/benchmark_real_weights.py --weights weight_permutations/
  python3 scripts/benchmark_real_weights.py --weights weight_permutations/ --kernels 15,29
"""
import argparse
import csv
import os
import re
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


def run_kernel(kernel: int, bin_path: str, rows: int, inners: int, cols: int):
    """Run kernel under NCU profiling.

    Returns (compute_ms, preprocess_ms) or (None, None) on failure.
    compute_ms: main ESMM/cuBLAS kernel time only.
    preprocess_ms: sum of all preprocessing kernel times.
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
        "--load-b", bin_path,
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
            print(f"    [WARN] NCU returned no Duration for kernel={kernel}. stdout[:300]:\n{imp.stdout[:300]}")
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


def benchmark_tensor(pt_path: Path, kernels: list):
    config = parse_config(pt_path)

    obj = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor) or obj.dim() != 2:
        return None

    arr = pad_to_tile(obj.float().numpy())
    orig_shape = tuple(obj.shape)
    M_w, K_w = arr.shape

    elem_sp = float((obj == 0).float().mean())
    block_sp = compute_block_sparsity(arr)

    # GEMM: A (rows × M_w, dense synthetic) * B (M_w × K_w, loaded weight)
    rows = 4096  # representative batch, divisible by BM=32

    # K29 smem dispatch supports up to MAX_K_BLOCKS=1024 (K ≤ 8192)
    MAX_K = 1024 * BK  # = 8192
    if M_w > MAX_K:
        print(f"  [SKIP] K dimension {M_w} > {MAX_K} (K29 smem limit)")
        return None

    label = (f"{config['pruner']}/grp{config['group_size']}/"
             f"{config['perm_type']}/sp{config['sparsity']}/{pt_path.stem[:35]}")
    print(f"\n{label}")
    print(f"  shape={orig_shape}  elem={elem_sp*100:.1f}%  block={block_sp*100:.1f}%")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = f.name
        arr.astype(np.float32).tofile(f)

    try:
        # timings[k] = (compute_ms, preprocess_ms)
        timings = {}
        for k in kernels:
            compute_ms, preprocess_ms = run_kernel(k, tmp_path, rows, M_w, K_w)
            timings[k] = (compute_ms, preprocess_ms)
            label_k = "cuBLAS" if k == 15 else f"K{k}"
            if compute_ms is not None:
                total = compute_ms + (preprocess_ms or 0.0)
                print(f"  {label_k}: compute={compute_ms:.3f}ms  "
                      f"preprocess={preprocess_ms:.3f}ms  total={total:.3f}ms")
            else:
                print(f"  {label_k}: FAILED")
    finally:
        os.unlink(tmp_path)

    row = {
        "layer": pt_path.stem,
        "pruner": config["pruner"],
        "group_size": config["group_size"],
        "perm_type": config["perm_type"],
        "sparsity": config["sparsity"],
        "orig_shape": f"{orig_shape[0]}x{orig_shape[1]}",
        "elem_sparsity_pct": f"{elem_sp*100:.1f}",
        "block_sparsity_pct": f"{block_sp*100:.1f}",
    }

    for k in kernels:
        compute_ms, preprocess_ms = timings[k]
        row[f"k{k}_compute_ms"] = f"{compute_ms:.3f}" if compute_ms is not None else "N/A"
        row[f"k{k}_preprocess_ms"] = f"{preprocess_ms:.3f}" if preprocess_ms is not None else "N/A"

    # Speedup columns for K29 vs cuBLAS (K15)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weight_permutations/",
                        help="Root dir containing *_permuted.pt files")
    parser.add_argument("--kernels", default="15,29",
                        help="Comma-separated kernel IDs (default: 15,29)")
    parser.add_argument("--out", default="results/real_weights_benchmark.csv")
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    kernels = [int(k) for k in args.kernels.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not EXEC.exists():
        print(f"Error: exec not found at {EXEC}. Run 'make release'.")
        sys.exit(1)

    if not Path(NCU_PATH).exists():
        print(f"Error: ncu not found at {NCU_PATH}.")
        sys.exit(1)

    pt_files = sorted(weights_dir.rglob("*_permuted.pt"))
    if not pt_files:
        print(f"No *_permuted.pt files found in {weights_dir}")
        sys.exit(1)

    print(f"Found {len(pt_files)} weight tensors | kernels={kernels} | exec={EXEC.name} | ncu=ON")
    print("─" * 100)

    rows_out = []
    fieldnames = None
    for pt in pt_files:
        row = benchmark_tensor(pt, kernels)
        if row:
            rows_out.append(row)
            if fieldnames is None:
                fieldnames = list(row.keys())
            # Flush after every tensor so partial results are safe
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
            print(f"K29 vs cuBLAS — {len(compute_speedups)} configs (NCU timing):")
            print(f"  Compute-only: min={min(compute_speedups):.2f}x  "
                  f"max={max(compute_speedups):.2f}x  "
                  f"mean={sum(compute_speedups)/len(compute_speedups):.2f}x")
            print(f"  Total (incl. preprocess): min={min(total_speedups):.2f}x  "
                  f"max={max(total_speedups):.2f}x  "
                  f"mean={sum(total_speedups)/len(total_speedups):.2f}x")

    print(f"\nResults saved to {out_path}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
