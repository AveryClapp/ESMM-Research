#!/usr/bin/env python3
"""
Benchmark ESMM kernels vs cuBLAS on real LLM weight tensors.

Loads .pt files directly, writes a temp binary per tensor, benchmarks, deletes.
No persistent binary storage needed — only ~200MB temp space at a time.

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


def run_kernel(kernel: int, bin_path: str, rows: int, inners: int, cols: int):
    """Run kernel on binary file, return kernel time in ms. None on error."""
    cmd = [
        str(EXEC), str(kernel),
        "--load-b", bin_path,
        "--dims", str(rows), str(inners), str(cols),
        "--no-check",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        m = re.search(r"\(avg:\s*([\d.]+)\s*ms\)", output)
        if m:
            return float(m.group(1))
        m = re.search(r"Kernel time:\s*([\d.]+)\s*ms", output)
        if m:
            return float(m.group(1))
        print(f"    [WARN] No timing:\n{output[:300]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] kernel={kernel}")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


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
        timings = {}
        for k in kernels:
            t = run_kernel(k, tmp_path, rows, M_w, K_w)
            timings[k] = t
            label_k = "cuBLAS" if k == 15 else f"K{k}"
            print(f"  {label_k}: {f'{t:.3f} ms' if t is not None else 'FAILED'}")
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
        row[f"k{k}_ms"] = f"{timings[k]:.3f}" if timings[k] is not None else "N/A"
    if timings.get(15) and timings.get(29):
        row["k29_speedup"] = f"{timings[15]/timings[29]:.3f}"
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

    pt_files = sorted(weights_dir.rglob("*_permuted.pt"))
    if not pt_files:
        print(f"No *_permuted.pt files found in {weights_dir}")
        sys.exit(1)

    print(f"Found {len(pt_files)} weight tensors | kernels={kernels} | exec={EXEC.name}")
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

    speedups = []
    for r in rows_out:
        try:
            speedups.append(float(r["k29_speedup"]))
        except (KeyError, ValueError):
            pass

    if speedups:
        print(f"\n{'─'*60}")
        print(f"K29 vs cuBLAS — {len(speedups)} configs:")
        print(f"  min={min(speedups):.2f}x  max={max(speedups):.2f}x  "
              f"mean={sum(speedups)/len(speedups):.2f}x")

    print(f"\nResults saved to {out_path}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
